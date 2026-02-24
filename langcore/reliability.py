"""Extraction Reliability Score — composite quality metric.

Combines multiple extraction quality signals into a single
``reliability_score`` in ``[0.0, 1.0]``:

- **Confidence** — alignment-based confidence (already on ``Extraction``).
- **Schema validity** — did the extraction pass Pydantic validation?
- **Field completeness** — are all required schema fields non-empty?
- **Source grounding** — does the extraction have a ``char_interval``
  covering a meaningful span?

The weights are configurable via ``ReliabilityConfig``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pydantic

from langcore.core import data

__all__ = [
    "ReliabilityConfig",
    "compute_reliability_score",
    "compute_reliability_scores",
]


@dataclasses.dataclass(frozen=True)
class ReliabilityConfig:
    """Weights for the composite reliability score.

    Each weight controls how much a particular signal contributes to
    the final ``reliability_score``.  Weights are normalised internally
    so they do not need to sum to 1.0.

    Attributes:
        w_confidence: Weight for the alignment-based confidence score.
        w_schema_valid: Weight for schema (Pydantic) validity.
        w_completeness: Weight for required-field completeness ratio.
        w_grounding: Weight for source-grounding strength.
    """

    w_confidence: float = 0.4
    w_schema_valid: float = 0.2
    w_completeness: float = 0.2
    w_grounding: float = 0.2


# ── Helpers ───────────────────────────────────────────────────────


def _schema_validity_score(
    extraction: data.Extraction,
    schema: type[pydantic.BaseModel] | None,
    primary_field: str | None,
    *,
    _validity_cache: dict[int, float] | None = None,
) -> float:
    """Return 1.0 if the extraction validates against *schema*, else 0.0.

    When *schema* is ``None`` the signal is neutral (1.0) so it does
    not penalise extractions that were not schema-driven.

    Parameters:
        _validity_cache: Optional ``{id(extraction): score}`` cache.
            When provided, results are stored/retrieved to avoid
            redundant ``model_validate`` calls.
    """
    if schema is None:
        return 1.0

    ext_id = id(extraction)
    if _validity_cache is not None and ext_id in _validity_cache:
        return _validity_cache[ext_id]

    model_name = schema.__name__
    if (
        extraction.extraction_class != model_name
        and extraction.extraction_class.lower() != model_name.lower()
    ):
        # Not the right class — treat as neutral rather than penalising
        score = 1.0
        if _validity_cache is not None:
            _validity_cache[ext_id] = score
        return score

    field_data: dict[str, Any] = {}
    if primary_field is not None:
        field_data[primary_field] = extraction.extraction_text
    if extraction.attributes:
        for key, value in extraction.attributes.items():
            if key in schema.model_fields:
                field_data[key] = value

    try:
        schema.model_validate(field_data)
        score = 1.0
    except pydantic.ValidationError:
        score = 0.0

    if _validity_cache is not None:
        _validity_cache[ext_id] = score
    return score


def _field_completeness_score(
    extraction: data.Extraction,
    schema: type[pydantic.BaseModel] | None,
    primary_field: str | None,
    *,
    required_fields: list[str] | None = None,
) -> float:
    """Ratio of non-empty required fields to total required fields.

    Returns 1.0 when no schema is provided (neutral).

    Parameters:
        required_fields: Pre-computed list of required field names.
            When provided, avoids re-introspecting the schema for
            every extraction (batch optimisation).
    """
    if schema is None:
        return 1.0

    if required_fields is None:
        required_fields = [
            name
            for name, field_info in schema.model_fields.items()
            if field_info.is_required()
        ]

    if not required_fields:
        return 1.0

    # Build the data dict the same way to_pydantic / validate does
    present: dict[str, Any] = {}
    if primary_field is not None:
        present[primary_field] = extraction.extraction_text
    if extraction.attributes:
        present.update(extraction.attributes)

    filled = 0
    for field_name in required_fields:
        value = present.get(field_name)
        if value is not None and value != "" and value != []:
            filled += 1

    return filled / len(required_fields)


def _grounding_score(extraction: data.Extraction) -> float:
    """Score in ``[0.0, 1.0]`` indicating source-grounding strength.

    - 1.0  — extraction has a valid non-zero ``char_interval``
    - 0.5  — extraction has a ``char_interval`` but it's degenerate
             (zero-length or missing positions)
    - 0.0  — no ``char_interval`` at all
    """
    ci = extraction.char_interval
    if ci is None:
        return 0.0
    if ci.start_pos is None or ci.end_pos is None:
        return 0.5
    if ci.end_pos <= ci.start_pos:
        return 0.5
    return 1.0


# ── Public API ────────────────────────────────────────────────────


def compute_reliability_score(
    extraction: data.Extraction,
    *,
    schema: type[pydantic.BaseModel] | None = None,
    primary_field: str | None = None,
    config: ReliabilityConfig | None = None,
    pre_validated: bool | None = None,
    required_fields: list[str] | None = None,
    _validity_cache: dict[int, float] | None = None,
) -> float:
    """Compute the composite reliability score for a single extraction.

    Parameters:
        extraction: The extraction to score.
        schema: Optional Pydantic model class.  When provided, schema
            validity and field completeness signals are evaluated.
        primary_field: The primary text field name for the schema
            (as returned by ``find_primary_text_field``).  Required
            when *schema* is provided; otherwise ignored.
        config: Weight configuration.  Uses default weights when
            ``None``.
        pre_validated: When ``True``, the extraction is assumed to
            have already passed Pydantic validation and the schema
            validity signal is set to 1.0 without calling
            ``model_validate``.  When ``None`` (default), validation
            is performed normally.
        required_fields: Pre-computed list of required field names
            for the schema (batch optimisation).
        _validity_cache: Internal per-batch cache mapping
            ``id(extraction)`` → validity score.

    Returns:
        A float in ``[0.0, 1.0]``.
    """
    cfg = config or ReliabilityConfig()

    # 1. Confidence signal
    confidence = (
        extraction.confidence_score if extraction.confidence_score is not None else 0.0
    )

    # 2. Schema validity signal
    if pre_validated is True:
        schema_valid = 1.0
    else:
        schema_valid = _schema_validity_score(
            extraction, schema, primary_field, _validity_cache=_validity_cache
        )

    # 3. Field completeness signal
    completeness = _field_completeness_score(
        extraction, schema, primary_field, required_fields=required_fields
    )

    # 4. Source-grounding signal
    grounding = _grounding_score(extraction)

    # Weighted combination — normalise weights so they sum to 1
    total_weight = (
        cfg.w_confidence + cfg.w_schema_valid + cfg.w_completeness + cfg.w_grounding
    )
    if total_weight == 0:
        return 0.0

    score = (
        cfg.w_confidence * confidence
        + cfg.w_schema_valid * schema_valid
        + cfg.w_completeness * completeness
        + cfg.w_grounding * grounding
    ) / total_weight

    return round(score, 4)


def compute_reliability_scores(
    result: data.AnnotatedDocument,
    *,
    schema: type[pydantic.BaseModel] | None = None,
    config: ReliabilityConfig | None = None,
    pre_validated: bool | None = None,
) -> None:
    """Compute and set ``reliability_score`` on every extraction in *result*.

    Mutates each ``Extraction`` in *result.extractions* in-place.

    Pre-computes the required-field list once for the batch and
    maintains a per-batch validation cache so that
    ``model_validate`` is called at most once per extraction.

    Parameters:
        result: The annotated document to score.
        schema: Optional Pydantic schema for validity / completeness signals.
        config: Weight configuration.
        pre_validated: When ``True``, all extractions are assumed to
            have already passed Pydantic validation.  The schema
            validity signal is set to 1.0 for every extraction
            without calling ``model_validate``.  Use this when
            ``pydantic_retry`` has already validated the result.
    """
    if not result.extractions:
        return

    primary_field: str | None = None
    required_fields: list[str] | None = None
    if schema is not None:
        from langcore.core.schema_utils import find_primary_text_field

        primary_field = find_primary_text_field(schema)
        required_fields = [
            name
            for name, field_info in schema.model_fields.items()
            if field_info.is_required()
        ]

    # Per-batch cache: avoids duplicate model_validate calls for the
    # same extraction object.
    validity_cache: dict[int, float] = {}

    for extraction in result.extractions:
        extraction.reliability_score = compute_reliability_score(
            extraction,
            schema=schema,
            primary_field=primary_field,
            config=config,
            pre_validated=pre_validated,
            required_fields=required_fields,
            _validity_cache=validity_cache,
        )
