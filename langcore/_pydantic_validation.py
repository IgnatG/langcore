"""Pydantic schema validation and retry for the extraction pipeline.

When a Pydantic ``schema`` is passed to ``extract()`` /
``async_extract()``, this module validates each extraction against the
schema after the initial LLM pass.  Extractions that fail validation
are collected with their error messages, and a correction prompt is
built asking the LLM to re-extract with the feedback.  Valid
extractions from the first pass are preserved; the retry only attempts
to fix the invalid ones.

**Chunk-level retry** — instead of re-extracting the entire document,
the retry identifies the minimal text regions around failing
extractions and only re-sends those chunks to the LLM.  This reduces
retry cost from ``O(document_size)`` to ``O(failing_chunks)`` and is
critical for large documents (e.g. 500K+ character texts).

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

#: Minimum padding (in characters) around a failing extraction's
#: ``char_interval`` when building the retry text region.
_RETRY_REGION_PADDING = 200


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


# ── Chunk-level retry helpers ─────────────────────────────────────


def _build_retry_regions(
    invalid: list[tuple[data.Extraction, str]],
    full_text: str,
    max_char_buffer: int,
) -> list[tuple[int, int]]:
    """Compute minimal text regions that cover all failing extractions.

    For each invalid extraction with a ``char_interval``, a region of
    ``max_char_buffer`` characters centred on the extraction is created.
    Overlapping/adjacent regions are merged so the LLM receives the
    fewest possible re-extraction calls.

    Extractions **without** a ``char_interval`` are assigned the full
    document range so the fallback is correct (but expensive).

    Parameters:
        invalid: List of ``(extraction, error_message)`` pairs.
        full_text: The complete document text.
        max_char_buffer: The chunk size used during extraction (also
            used as the region size for retries).

    Returns:
        Sorted, merged list of ``(start, end)`` character ranges.
    """
    text_len = len(full_text)
    if text_len == 0:
        return []

    raw_regions: list[tuple[int, int]] = []

    for ext, _ in invalid:
        ci = ext.char_interval
        if ci is not None and ci.start_pos is not None and ci.end_pos is not None:
            mid = (ci.start_pos + ci.end_pos) // 2
            half = max(max_char_buffer // 2, _RETRY_REGION_PADDING)
            start = max(0, mid - half)
            end = min(text_len, mid + half)
            raw_regions.append((start, end))
        else:
            # No positional info — must fall back to full text.
            raw_regions.append((0, text_len))

    if not raw_regions:
        return []

    # Sort and merge overlapping / adjacent regions.
    raw_regions.sort()
    merged: list[tuple[int, int]] = [raw_regions[0]]
    for start, end in raw_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping or touching — extend.
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _merge_usage_pair(
    original: dict[str, int] | None,
    retry: dict[str, int] | None,
) -> dict[str, int] | None:
    """Sum two token-usage dicts, returning ``None`` if both are empty."""
    if original is None and retry is None:
        return None
    merged: dict[str, int] = {}
    for u in (original, retry):
        if u is not None:
            for k, v in u.items():
                merged[k] = merged.get(k, 0) + v
    return merged or None


def _offset_extractions(
    extractions: list[data.Extraction],
    char_offset: int,
) -> list[data.Extraction]:
    """Shift each extraction's ``char_interval`` by *char_offset*.

    When we re-extract a sub-region of the full text, the returned
    extractions have positions relative to that sub-region.  This
    helper maps them back to the full-document coordinate space.
    """
    for ext in extractions:
        if ext.char_interval is not None:
            s = ext.char_interval.start_pos
            e = ext.char_interval.end_pos
            if s is not None:
                ext.char_interval.start_pos = s + char_offset
            if e is not None:
                ext.char_interval.end_pos = e + char_offset
    return extractions


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

    Uses **chunk-level retry**: instead of re-extracting the entire
    document, only the text regions surrounding failing extractions are
    re-sent to the LLM.  This reduces cost from ``O(document_size)``
    to ``O(failing_chunks)``.

    Valid extractions from the original result are always preserved.

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

    original_document_id = result.document_id
    accumulated_usage = result.usage

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

        # ── Chunk-level retry: only re-extract failing regions ────
        regions = _build_retry_regions(invalid, result.text, max_char_buffer)

        retry_extractions: list[data.Extraction] = []
        for region_start, region_end in regions:
            region_text = result.text[region_start:region_end]
            logger.debug(
                "Retry region [%d:%d] (%d chars)",
                region_start,
                region_end,
                len(region_text),
            )

            region_result = annotator.annotate_text(
                text=region_text,
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

            accumulated_usage = _merge_usage_pair(
                accumulated_usage, region_result.usage
            )

            if region_result.extractions:
                _offset_extractions(region_result.extractions, region_start)
                retry_extractions.extend(region_result.extractions)

        # Build a temporary doc to validate the retry extractions.
        retry_doc = data.AnnotatedDocument(
            extractions=retry_extractions or None,
            text=result.text,
        )
        retry_valid, retry_invalid = validate_extractions(retry_doc, schema)

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
            usage=accumulated_usage,
        )
        result.document_id = original_document_id

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
    """Async version of :func:`pydantic_retry`.

    Uses **chunk-level retry** — see :func:`pydantic_retry` docstring
    for details.
    """
    if result.text is None:
        return result

    original_document_id = result.document_id
    accumulated_usage = result.usage

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

        # ── Chunk-level retry: only re-extract failing regions ────
        regions = _build_retry_regions(invalid, result.text, max_char_buffer)

        retry_extractions: list[data.Extraction] = []
        for region_start, region_end in regions:
            region_text = result.text[region_start:region_end]
            logger.debug(
                "Retry region [%d:%d] (%d chars)",
                region_start,
                region_end,
                len(region_text),
            )

            region_result = await annotator.async_annotate_text(
                text=region_text,
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

            accumulated_usage = _merge_usage_pair(
                accumulated_usage, region_result.usage
            )

            if region_result.extractions:
                _offset_extractions(region_result.extractions, region_start)
                retry_extractions.extend(region_result.extractions)

        # Build a temporary doc to validate the retry extractions.
        retry_doc = data.AnnotatedDocument(
            extractions=retry_extractions or None,
            text=result.text,
        )
        retry_valid, retry_invalid = validate_extractions(retry_doc, schema)

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
            usage=accumulated_usage,
        )
        result.document_id = original_document_id

    return result
