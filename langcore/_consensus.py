"""Multi-model consensus extraction.

When ``consensus_models`` is provided to ``extract()`` /
``async_extract()``, this module runs the extraction independently
with each model and merges the results.  The merge reuses the
existing ``_merge_non_overlapping_extractions()`` logic from
``annotation.py``, treating each model's output like a separate
extraction pass.

Each string in ``consensus_models`` is resolved through the provider
router, so any model ID accepted by ``model_id`` works here:

- Built-in: ``"gemini-2.5-flash"``, ``"gpt-4o"``, ``"llama3.2:1b"``
- LiteLLM plugin: ``"litellm/anthropic/claude-sonnet-4"``,
  ``"litellm/gpt-4o"``, ``"litellm/bedrock/anthropic.claude-3"``
- Custom entry-point plugins: any registered pattern.

Providers can be freely mixed in a single consensus run.

Consensus confidence is computed as:
  ``agreement_ratio x alignment_confidence``
where ``agreement_ratio = models_that_found_it / total_models``.
This rewards extractions confirmed by multiple providers.

Each extraction is also tagged with the ``model_id`` that produced it
via the ``_consensus_model_id`` key in ``attributes``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Iterable
from typing import Any

from langcore import annotation as annotation_mod
from langcore.core import data

__all__: list[str] = []

logger = logging.getLogger(__name__)

#: Attribute key used to tag each extraction with the model that produced it.
MODEL_ID_KEY = "_consensus_model_id"


def _tag_extractions(
    result: data.AnnotatedDocument, model_id: str
) -> data.AnnotatedDocument:
    """Add ``MODEL_ID_KEY`` to each extraction's attributes."""
    if result.extractions:
        for ext in result.extractions:
            if ext.attributes is None:
                ext.attributes = {}
            ext.attributes[MODEL_ID_KEY] = model_id
    return result


#: Standard token-usage keys.  After merging, only these keys are
#: guaranteed to be present — extra provider-specific keys are summed
#: but moved into an ``"other"`` sub-dict so downstream consumers
#: see a consistent shape.
_STANDARD_USAGE_KEYS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})


def _merge_usage(
    usage_list: list[dict[str, int] | None],
) -> dict[str, int] | None:
    """Sum token usage dicts from multiple model runs.

    The result is normalised to have at most the standard keys
    (``prompt_tokens``, ``completion_tokens``, ``total_tokens``)
    at the top level.  Non-standard keys are still summed but
    kept separately so the merged dict has a predictable shape.
    """
    merged: dict[str, int] = {}
    for u in usage_list:
        if u is not None:
            for k, v in u.items():
                merged[k] = merged.get(k, 0) + v
    if not merged:
        return None

    # Ensure all standard keys are present (default 0).
    for key in _STANDARD_USAGE_KEYS:
        merged.setdefault(key, 0)

    # Recalculate total_tokens as sum of prompt + completion when
    # they're both present, to avoid double-counting across models.
    if "prompt_tokens" in merged and "completion_tokens" in merged:
        merged["total_tokens"] = merged["prompt_tokens"] + merged["completion_tokens"]

    return merged


def merge_consensus_results(
    results: list[data.AnnotatedDocument],
    text: str | None,
    *,
    document_id: str | None = None,
) -> data.AnnotatedDocument:
    """Merge extraction results from multiple models into a consensus.

    Uses ``_merge_non_overlapping_extractions`` with
    ``total_passes = len(results)`` so that confidence is computed as
    ``agreement_ratio x alignment_confidence``.

    Parameters:
        results: One ``AnnotatedDocument`` per model.
        text: The original source text.
        document_id: Optional document ID to forward to the merged
            result.  When not provided the ID from the first result is
            used (if available), ensuring that the original document
            identity is preserved through the consensus pipeline.

    Returns:
        A single ``AnnotatedDocument`` with merged extractions.
    """
    if not results:
        return data.AnnotatedDocument(text=text)

    if len(results) == 1:
        return results[0]

    extraction_lists: list[list[data.Extraction]] = []
    for r in results:
        extraction_lists.append(r.extractions or [])

    merged_extractions = annotation_mod._merge_non_overlapping_extractions(
        extraction_lists,
        total_passes=len(results),
    )

    usage = _merge_usage([r.usage for r in results])

    effective_id = document_id or results[0]._document_id

    merged = data.AnnotatedDocument(
        document_id=effective_id,
        extractions=merged_extractions or None,
        text=text,
        usage=usage,
    )
    return merged


def consensus_extract(
    text: str,
    model_ids: list[str],
    *,
    build_components_fn: Any,
    build_kwargs: dict[str, Any],
    annotate_kwargs: dict[str, Any],
    max_workers: int | None = None,
    fail_fast: bool = True,
) -> data.AnnotatedDocument:
    """Run sync extraction with each model **in parallel** and merge results.

    Models are extracted concurrently using a
    :class:`~concurrent.futures.ThreadPoolExecutor`.  Each model
    builds its own annotator/resolver via ``build_components_fn``, so
    there is no shared mutable state between threads.

    Parameters:
        text: The source text.
        model_ids: List of model IDs to extract with.
        build_components_fn: Reference to ``_build_extraction_components``.
        build_kwargs: Keyword arguments for ``_build_extraction_components``
            (without ``model_id``).
        annotate_kwargs: Keyword arguments for ``annotator.annotate_text``
            (without ``text`` and ``resolver``).
        max_workers: Maximum number of threads.  Defaults to
            ``min(len(model_ids), 4)``.
        fail_fast: If ``True`` (default), any model failure raises
            immediately.  If ``False``, failed models are logged and
            skipped; the consensus is computed from successful results
            only.  Raises :class:`RuntimeError` if **all** models fail.

    Returns:
        A single ``AnnotatedDocument`` with consensus extractions.
    """

    def _run_one(mid: str) -> data.AnnotatedDocument:
        logger.info("Consensus extraction: running model %s", mid)
        _, annotator, res, alignment_kwargs = build_components_fn(
            model_id=mid, **build_kwargs
        )
        result = annotator.annotate_text(
            text=text,
            resolver=res,
            **annotate_kwargs,
            **alignment_kwargs,
        )
        _tag_extractions(result, mid)
        return result

    effective_workers = max_workers or min(len(model_ids), 4)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers
    ) as executor:
        futures = {executor.submit(_run_one, mid): mid for mid in model_ids}
        results: list[data.AnnotatedDocument] = []
        for future in concurrent.futures.as_completed(futures):
            mid = futures[future]
            try:
                results.append(future.result())
            except Exception:
                if fail_fast:
                    logger.exception("Consensus extraction failed for model %s", mid)
                    raise
                logger.warning(
                    "Consensus extraction failed for model %s (skipping)",
                    mid,
                    exc_info=True,
                )

    if not results:
        raise RuntimeError(
            f"All consensus models failed. Models attempted: {', '.join(model_ids)}"
        )

    return merge_consensus_results(results, text=text)


async def async_consensus_extract(
    text: str,
    model_ids: list[str],
    *,
    build_components_fn: Any,
    build_kwargs: dict[str, Any],
    annotate_kwargs: dict[str, Any],
    fail_fast: bool = True,
) -> data.AnnotatedDocument:
    """Run async extraction with each model and merge results.

    Models are extracted concurrently using ``asyncio.gather``.

    Parameters:
        text: The source text.
        model_ids: List of model IDs to extract with.
        build_components_fn: Reference to ``_build_extraction_components``.
        build_kwargs: Keyword arguments for ``_build_extraction_components``
            (without ``model_id``).
        annotate_kwargs: Keyword arguments for ``annotator.annotate_text``
            (without ``text`` and ``resolver``).
        fail_fast: If ``True`` (default), any model failure raises
            immediately.  If ``False``, failed models are logged and
            skipped.  Raises :class:`RuntimeError` if **all** models
            fail.

    Returns:
        A single ``AnnotatedDocument`` with consensus extractions.
    """

    async def _run_one(mid: str) -> data.AnnotatedDocument:
        logger.info("Consensus extraction: running model %s", mid)
        _, annotator, res, alignment_kwargs = build_components_fn(
            model_id=mid, **build_kwargs
        )
        result = await annotator.async_annotate_text(
            text=text,
            resolver=res,
            **annotate_kwargs,
            **alignment_kwargs,
        )
        _tag_extractions(result, mid)
        return result

    raw_results = await asyncio.gather(
        *[_run_one(mid) for mid in model_ids],
        return_exceptions=not fail_fast,
    )

    results: list[data.AnnotatedDocument] = []
    for mid, res_or_exc in zip(model_ids, raw_results):
        if isinstance(res_or_exc, BaseException):
            logger.warning(
                "Consensus extraction failed for model %s (skipping): %s",
                mid,
                res_or_exc,
            )
        else:
            results.append(res_or_exc)

    if not results:
        raise RuntimeError(
            f"All consensus models failed. Models attempted: {', '.join(model_ids)}"
        )

    return merge_consensus_results(list(results), text=text)


# ── Document-list consensus ──────────────────────────────────────


def consensus_extract_documents(
    documents: Iterable[data.Document],
    model_ids: list[str],
    *,
    build_components_fn: Any,
    build_kwargs: dict[str, Any],
    annotate_kwargs: dict[str, Any],
    max_workers: int | None = None,
    fail_fast: bool = True,
) -> list[data.AnnotatedDocument]:
    """Run sync consensus extraction over a list of documents.

    Each document is extracted independently by every model, then
    per-document results are merged.  The original ``document_id`` of
    each :class:`~langcore.core.data.Document` is forwarded to the
    merged :class:`~langcore.core.data.AnnotatedDocument`.

    Parameters:
        documents: Iterable of :class:`~langcore.core.data.Document`.
        model_ids: List of model IDs to extract with.
        build_components_fn: Reference to ``_build_extraction_components``.
        build_kwargs: Keyword arguments for ``_build_extraction_components``
            (without ``model_id`` and ``text_or_documents``).
        annotate_kwargs: Keyword arguments for ``annotator.annotate_text``
            (without ``text`` and ``resolver``).
        max_workers: Maximum parallel threads (forwarded to
            :func:`consensus_extract`).
        fail_fast: If ``True`` (default), any model failure raises
            immediately.  See :func:`consensus_extract`.

    Returns:
        A list of ``AnnotatedDocument``, one per input document.
    """
    merged_results: list[data.AnnotatedDocument] = []
    for doc in documents:
        # Override text_or_documents for this specific document.
        doc_build_kwargs = {**build_kwargs, "text_or_documents": doc.text}
        result = consensus_extract(
            text=doc.text,
            model_ids=model_ids,
            build_components_fn=build_components_fn,
            build_kwargs=doc_build_kwargs,
            annotate_kwargs=annotate_kwargs,
            max_workers=max_workers,
            fail_fast=fail_fast,
        )
        result.document_id = doc.document_id
        merged_results.append(result)
    return merged_results


async def async_consensus_extract_documents(
    documents: Iterable[data.Document],
    model_ids: list[str],
    *,
    build_components_fn: Any,
    build_kwargs: dict[str, Any],
    annotate_kwargs: dict[str, Any],
    fail_fast: bool = True,
) -> list[data.AnnotatedDocument]:
    """Async version of :func:`consensus_extract_documents`.

    Each document is processed sequentially (models run concurrently
    per document via :func:`async_consensus_extract`).

    Parameters:
        documents: Iterable of :class:`~langcore.core.data.Document`.
        model_ids: List of model IDs to extract with.
        build_components_fn: Reference to ``_build_extraction_components``.
        build_kwargs: Keyword arguments for ``_build_extraction_components``.
        annotate_kwargs: Keyword arguments for ``annotator.annotate_text``.
        fail_fast: See :func:`async_consensus_extract`.

    Returns:
        A list of ``AnnotatedDocument``, one per input document.
    """
    merged_results: list[data.AnnotatedDocument] = []
    for doc in documents:
        doc_build_kwargs = {**build_kwargs, "text_or_documents": doc.text}
        result = await async_consensus_extract(
            text=doc.text,
            model_ids=model_ids,
            build_components_fn=build_components_fn,
            build_kwargs=doc_build_kwargs,
            annotate_kwargs=annotate_kwargs,
            fail_fast=fail_fast,
        )
        result.document_id = doc.document_id
        merged_results.append(result)
    return merged_results
