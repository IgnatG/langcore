"""Pydantic schema validation and retry for the extraction pipeline.

When a Pydantic ``schema`` is passed to ``extract()`` /
``async_extract()``, this module validates each extraction against the
schema after the initial LLM pass.  Extractions that fail validation
are collected with their error messages, and a correction prompt is
built asking the LLM to re-extract with the feedback.  Valid
extractions from the first pass are preserved; the retry only attempts
to fix the invalid ones.

This implements the Instructor-style "validate → re-ask" pattern
described in the Phase 1 plan.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pydantic

from langcore.core import data
from langcore.core.schema_utils import find_primary_text_field

if TYPE_CHECKING:
    from langcore import annotation as annotation_mod
    from langcore import hooks as hooks_lib
    from langcore import resolver as resolver_mod
    from langcore.core import tokenizer as tokenizer_lib

__all__: list[str] = []

logger = logging.getLogger(__name__)

# ── Correction prompt template ────────────────────────────────────

_CORRECTION_CONTEXT = (
    "\n\n--- VALIDATION FEEDBACK ---\n"
    "Your previous extraction attempt produced outputs that failed "
    "schema validation. Please fix the following issues:\n\n"
    "{errors}\n\n"
    "Re-extract the entities, ensuring each one conforms to the "
    "required schema. Return ALL valid extractions, not just the "
    "corrected ones."
)


# ── Validation helpers ────────────────────────────────────────────


def _extraction_to_field_data(
    extraction: data.Extraction,
    schema: type[pydantic.BaseModel],
    primary_field: str,
) -> dict[str, Any]:
    """Map an ``Extraction`` to a dict suitable for ``model_validate``.

    Uses the same mapping logic as ``AnnotatedDocument.to_pydantic()``.
    """
    field_data: dict[str, Any] = {
        primary_field: extraction.extraction_text,
    }
    if extraction.attributes:
        for key, value in extraction.attributes.items():
            if key in schema.model_fields:
                field_data[key] = value
    return field_data


def validate_extractions(
    result: data.AnnotatedDocument,
    schema: type[pydantic.BaseModel],
) -> tuple[list[data.Extraction], list[tuple[data.Extraction, str]]]:
    """Validate extractions against a Pydantic schema.

    Parameters:
        result: The annotated document whose extractions to validate.
        schema: The Pydantic model class to validate against.

    Returns:
        A ``(valid, invalid)`` tuple where *valid* is a list of
        extractions that passed validation and *invalid* is a list
        of ``(extraction, error_message)`` pairs for failures.
    """
    if not result.extractions:
        return [], []

    model_name = schema.__name__
    primary_field = find_primary_text_field(schema)
    valid: list[data.Extraction] = []
    invalid: list[tuple[data.Extraction, str]] = []

    for extraction in result.extractions:
        # Only validate extractions that match the schema class
        if (
            extraction.extraction_class != model_name
            and extraction.extraction_class.lower() != model_name.lower()
        ):
            valid.append(extraction)
            continue

        field_data = _extraction_to_field_data(extraction, schema, primary_field)
        try:
            schema.model_validate(field_data)
            valid.append(extraction)
        except pydantic.ValidationError as exc:
            error_msg = (
                f"Extraction '{extraction.extraction_text}' "
                f"(class={extraction.extraction_class}): {exc}"
            )
            invalid.append((extraction, error_msg))

    return valid, invalid


def build_correction_context(
    invalid: list[tuple[data.Extraction, str]],
) -> str:
    """Build additional context describing validation failures.

    Parameters:
        invalid: List of ``(extraction, error_message)`` pairs.

    Returns:
        A string to append as ``additional_context`` for the retry.
    """
    error_lines = []
    for i, (_, error_msg) in enumerate(invalid, 1):
        error_lines.append(f"{i}. {error_msg}")
    return _CORRECTION_CONTEXT.format(errors="\n".join(error_lines))


# ── Sync retry ────────────────────────────────────────────────────


def pydantic_retry(
    result: data.AnnotatedDocument,
    schema: type[pydantic.BaseModel],
    annotator: annotation_mod.Annotator,
    res: resolver_mod.AbstractResolver,
    *,
    max_char_buffer: int,
    batch_length: int,
    additional_context: str | None,
    debug: bool,
    extraction_passes: int,
    context_window_chars: int | None,
    show_progress: bool,
    max_workers: int,
    tokenizer: tokenizer_lib.Tokenizer | None,
    alignment_kwargs: dict[str, Any],
    hooks: hooks_lib.Hooks,
    max_retries: int = 1,
) -> data.AnnotatedDocument:
    """Validate extractions and re-run extraction on failures.

    Valid extractions from the original result are always preserved.
    The retry pass re-extracts with a correction prompt and its valid
    results are merged in.

    Parameters:
        result: Original extraction result.
        schema: Pydantic model class for validation.
        annotator: Annotator instance for re-extraction.
        res: Resolver instance.
        max_retries: Number of retry attempts (default 1).
        Additional parameters mirror ``extract()`` arguments.

    Returns:
        The (possibly updated) ``AnnotatedDocument``.
    """
    if result.text is None:
        return result

    for attempt in range(max_retries):
        valid, invalid = validate_extractions(result, schema)
        if not invalid:
            logger.debug("Pydantic validation passed on attempt %d", attempt)
            return result

        logger.info(
            "Pydantic validation: %d valid, %d invalid extractions (retry %d/%d)",
            len(valid),
            len(invalid),
            attempt + 1,
            max_retries,
        )

        correction = build_correction_context(invalid)
        combined_context = (
            f"{additional_context}\n{correction}" if additional_context else correction
        )

        retry_result = annotator.annotate_text(
            text=result.text,
            resolver=res,
            max_char_buffer=max_char_buffer,
            batch_length=batch_length,
            additional_context=combined_context,
            debug=debug,
            extraction_passes=extraction_passes,
            context_window_chars=context_window_chars,
            show_progress=show_progress,
            max_workers=max_workers,
            tokenizer=tokenizer,
            **alignment_kwargs,
        )

        # Validate the retry result and keep only the valid ones
        retry_valid, retry_invalid = validate_extractions(retry_result, schema)

        if retry_invalid:
            logger.debug(
                "Retry attempt %d still has %d invalid extractions",
                attempt + 1,
                len(retry_invalid),
            )

        # Merge: original valid + retry valid.  Keep still-invalid
        # extractions from the retry so the next iteration can try
        # to fix them again.
        still_invalid_exts = [ext for ext, _ in retry_invalid]
        merged = valid + retry_valid + still_invalid_exts
        result = data.AnnotatedDocument(
            extractions=merged if merged else None,
            text=result.text,
            usage=result.usage,
        )
        result.document_id = retry_result.document_id

    return result


# ── Async retry ───────────────────────────────────────────────────


async def async_pydantic_retry(
    result: data.AnnotatedDocument,
    schema: type[pydantic.BaseModel],
    annotator: annotation_mod.Annotator,
    res: resolver_mod.AbstractResolver,
    *,
    max_char_buffer: int,
    batch_length: int,
    additional_context: str | None,
    debug: bool,
    extraction_passes: int,
    context_window_chars: int | None,
    show_progress: bool,
    max_workers: int,
    tokenizer: tokenizer_lib.Tokenizer | None,
    alignment_kwargs: dict[str, Any],
    hooks: hooks_lib.Hooks,
    max_retries: int = 1,
) -> data.AnnotatedDocument:
    """Async version of :func:`pydantic_retry`."""
    if result.text is None:
        return result

    for attempt in range(max_retries):
        valid, invalid = validate_extractions(result, schema)
        if not invalid:
            logger.debug("Pydantic validation passed on attempt %d", attempt)
            return result

        logger.info(
            "Pydantic validation: %d valid, %d invalid extractions (retry %d/%d)",
            len(valid),
            len(invalid),
            attempt + 1,
            max_retries,
        )

        correction = build_correction_context(invalid)
        combined_context = (
            f"{additional_context}\n{correction}" if additional_context else correction
        )

        retry_result = await annotator.async_annotate_text(
            text=result.text,
            resolver=res,
            max_char_buffer=max_char_buffer,
            batch_length=batch_length,
            additional_context=combined_context,
            debug=debug,
            extraction_passes=extraction_passes,
            context_window_chars=context_window_chars,
            show_progress=show_progress,
            max_workers=max_workers,
            tokenizer=tokenizer,
            **alignment_kwargs,
        )

        retry_valid, retry_invalid = validate_extractions(retry_result, schema)

        if retry_invalid:
            logger.debug(
                "Retry attempt %d still has %d invalid extractions",
                attempt + 1,
                len(retry_invalid),
            )

        # Merge: original valid + retry valid.  Keep still-invalid
        # extractions from the retry so the next iteration can try
        # to fix them again.
        still_invalid_exts = [ext for ext, _ in retry_invalid]
        merged = valid + retry_valid + still_invalid_exts
        result = data.AnnotatedDocument(
            extractions=merged if merged else None,
            text=result.text,
            usage=result.usage,
        )
        result.document_id = retry_result.document_id

    return result
