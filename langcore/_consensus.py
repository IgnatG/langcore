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


def _merge_usage(
    usage_list: list[dict[str, int] | None],
) -> dict[str, int] | None:
    """Sum token usage dicts from multiple model runs."""
    merged: dict[str, int] = {}
    for u in usage_list:
        if u is not None:
            for k, v in u.items():
                merged[k] = merged.get(k, 0) + v
    return merged or None


def merge_consensus_results(
    results: list[data.AnnotatedDocument],
    text: str | None,
) -> data.AnnotatedDocument:
    """Merge extraction results from multiple models into a consensus.

    Uses ``_merge_non_overlapping_extractions`` with
    ``total_passes = len(results)`` so that confidence is computed as
    ``agreement_ratio x alignment_confidence``.

    Parameters:
        results: One ``AnnotatedDocument`` per model.
        text: The original source text.

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

    merged = data.AnnotatedDocument(
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
                logger.exception("Consensus extraction failed for model %s", mid)
                raise

    return merge_consensus_results(results, text=text)


async def async_consensus_extract(
    text: str,
    model_ids: list[str],
    *,
    build_components_fn: Any,
    build_kwargs: dict[str, Any],
    annotate_kwargs: dict[str, Any],
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

    results = await asyncio.gather(*[_run_one(mid) for mid in model_ids])
    return merge_consensus_results(list(results), text=text)
