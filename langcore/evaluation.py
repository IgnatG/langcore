"""Quality metrics and evaluation for LangCore extractions.

Provides ``ExtractionMetrics`` for computing precision, recall, F1, and
accuracy between predicted and ground-truth extractions.  Supports both
class-level and field-level (per-attribute) evaluation, with an optional
Pydantic schema for structured field-level breakdowns.

Typical usage::

    from langcore.evaluation import ExtractionMetrics

    metrics = ExtractionMetrics()
    report = metrics.evaluate(predictions=results, ground_truth=expected)
    print(report.f1)          # 0.92
    print(report.per_field)   # {"invoice_number": 0.98, "amount": 0.88}

Averaging modes::

    # Macro (default) — compute P/R/F1 across all extractions at once
    metrics = ExtractionMetrics(averaging="macro")

    # Micro — compute P/R/F1 per document, then average
    metrics = ExtractionMetrics(averaging="micro")

    # Weighted — per-document P/R/F1 weighted by ground-truth count
    metrics = ExtractionMetrics(averaging="weighted")

Fuzzy matching::

    # Allow near-matches with ≥80% similarity
    metrics = ExtractionMetrics(fuzzy_threshold=0.8)
"""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import logging
from collections.abc import Sequence
from typing import Literal

import pydantic

from langcore.core.data import AnnotatedDocument, Extraction

# Type alias for the averaging strategy.
AveragingMode = Literal["macro", "micro", "weighted"]

__all__ = [
    "AveragingMode",
    "EvaluationReport",
    "ExtractionMetrics",
    "FieldReport",
]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #


@dataclasses.dataclass
class FieldReport:
    """Per-field evaluation metrics.

    Attributes:
        field_name: Name of the field (attribute key or
            ``extraction_class``).
        precision: Precision for this field.
        recall: Recall for this field.
        f1: F1 score for this field.
        support: Number of ground-truth instances for this field.
    """

    field_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclasses.dataclass
class EvaluationReport:
    """Full evaluation report with aggregate and per-field metrics.

    Attributes:
        precision: Aggregate precision.
        recall: Aggregate recall.
        f1: Aggregate F1 score.
        accuracy: Fraction of ground-truth extractions exactly
            matched by predictions.
        total_predictions: Total predicted extractions evaluated.
        total_ground_truth: Total ground-truth extractions evaluated.
        true_positives: Number of matching extraction pairs.
        averaging: The averaging strategy used (``"macro"``,
            ``"micro"``, or ``"weighted"``).
        per_field: Per-field ``FieldReport`` breakdown, keyed by
            field name.  Empty when no schema is provided.
        per_document: Optional list of per-document ``(precision,
            recall, f1)`` tuples in the same order as the inputs.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    total_predictions: int
    total_ground_truth: int
    true_positives: int
    averaging: str = "macro"
    per_field: dict[str, FieldReport] = dataclasses.field(
        default_factory=dict,
    )
    per_document: list[dict[str, float]] = dataclasses.field(
        default_factory=list,
    )


# ------------------------------------------------------------------ #
# Key-building helpers
# ------------------------------------------------------------------ #


def _extraction_key(ext: Extraction) -> str:
    """Build a normalised comparison key for an ``Extraction``.

    The key is ``class|text`` lower-cased and whitespace-collapsed so
    that minor formatting differences don't cause false negatives.

    Results are memorised on the ``Extraction`` instance via a
    ``_cached_key`` attribute to avoid re-computing the same string
    operations when the same object is passed through multiple metrics.
    """
    cached = getattr(ext, "_cached_key", None)
    if cached is not None:
        return cached
    cls = (ext.extraction_class or "").strip().lower()
    txt = " ".join((ext.extraction_text or "").split()).lower()
    key = f"{cls}|{txt}"
    with contextlib.suppress(AttributeError, TypeError):
        ext._cached_key = key  # type: ignore[attr-defined]
    return key


def _extraction_key_with_attrs(ext: Extraction) -> str:
    """Build a key that also includes sorted attribute pairs.

    Used for stricter matching when field-level metrics are requested.

    Like ``_extraction_key``, results are memoised on the instance.
    """
    cached = getattr(ext, "_cached_key_attrs", None)
    if cached is not None:
        return cached
    base = _extraction_key(ext)
    if not ext.attributes:
        return base
    attr_parts = sorted(
        f"{k}={str(v).strip().lower()}" for k, v in ext.attributes.items()
    )
    key = f"{base}||{'|'.join(attr_parts)}"
    with contextlib.suppress(AttributeError, TypeError):
        ext._cached_key_attrs = key  # type: ignore[attr-defined]
    return key


def _field_keys(
    ext: Extraction,
) -> dict[str, str]:
    """Return ``{field_name: normalised_value}`` for an extraction.

    Each extraction contributes:
    * ``extraction_class`` → the class string
    * ``extraction_text`` → the text string
    * One entry per attribute key → its value

    This powers per-field precision / recall.
    """
    fields: dict[str, str] = {
        "extraction_class": (ext.extraction_class or "").strip().lower(),
        "extraction_text": " ".join((ext.extraction_text or "").split()).lower(),
    }
    if ext.attributes:
        for k, v in ext.attributes.items():
            fields[k] = str(v).strip().lower()
    return fields


# ------------------------------------------------------------------ #
# Flat-list helpers
# ------------------------------------------------------------------ #


def _flatten(
    data: (
        Sequence[Extraction]
        | Sequence[list[Extraction]]
        | AnnotatedDocument
        | Sequence[AnnotatedDocument]
    ),
) -> list[Extraction]:
    """Normalise various input shapes into a flat ``list[Extraction]``.

    Accepted shapes:
    * A single ``AnnotatedDocument``
    * A list of ``AnnotatedDocument``
    * A flat list of ``Extraction``
    * A list of lists of ``Extraction``
    """
    if isinstance(data, AnnotatedDocument):
        return list(data.extractions or [])

    flat: list[Extraction] = []
    for item in data:
        if isinstance(item, AnnotatedDocument):
            flat.extend(item.extractions or [])
        elif isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, Extraction):
            flat.append(item)
        else:
            raise TypeError(
                f"Unsupported element type {type(item).__name__} in "
                "predictions/ground_truth sequence."
            )
    return flat


def _flatten_per_document(
    data: (
        Sequence[Extraction]
        | Sequence[list[Extraction]]
        | AnnotatedDocument
        | Sequence[AnnotatedDocument]
    ),
) -> list[list[Extraction]]:
    """Normalise input into a ``list[list[Extraction]]`` per document.

    When given a flat ``list[Extraction]`` or a single
    ``AnnotatedDocument``, wraps in a single-element outer list.
    """
    if isinstance(data, AnnotatedDocument):
        return [list(data.extractions or [])]

    groups: list[list[Extraction]] = []
    for item in data:
        if isinstance(item, AnnotatedDocument):
            groups.append(list(item.extractions or []))
        elif isinstance(item, list):
            groups.append(item)
        elif isinstance(item, Extraction):
            # flat list — wrap everything in one group
            return [list(data)]  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported element type {type(item).__name__}.")
    return groups


# ------------------------------------------------------------------ #
# Core metric computation
# ------------------------------------------------------------------ #


def _compute_prf(
    pred_keys: list[str],
    gt_keys: list[str],
    *,
    fuzzy_threshold: float | None = None,
) -> tuple[float, float, float, int]:
    """Compute precision, recall, F1 and true-positive count.

    Uses multiset (bag) matching: each ground-truth key can be matched
    at most once, preserving correct counts when duplicates exist.

    When *fuzzy_threshold* is set (a float in ``(0, 1]``), keys are
    compared using ``difflib.SequenceMatcher``.  A prediction matches a
    ground-truth key when the similarity ratio meets or exceeds the
    threshold.  Exact matching (the default) uses plain ``==``.

    Returns:
        ``(precision, recall, f1, true_positives)``
    """
    gt_remaining = list(gt_keys)
    tp = 0
    if fuzzy_threshold is not None and fuzzy_threshold < 1.0:
        for pk in pred_keys:
            best_idx: int | None = None
            best_ratio = 0.0
            for i, gk in enumerate(gt_remaining):
                ratio = difflib.SequenceMatcher(None, pk, gk).ratio()
                if ratio >= fuzzy_threshold and ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
            if best_idx is not None:
                gt_remaining.pop(best_idx)
                tp += 1
    else:
        for pk in pred_keys:
            if pk in gt_remaining:
                gt_remaining.remove(pk)
                tp += 1

    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(gt_keys) if gt_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1, tp


# ------------------------------------------------------------------ #
# ExtractionMetrics
# ------------------------------------------------------------------ #


class ExtractionMetrics:
    """Compute quality metrics between predicted and expected extractions.

    Optionally accepts a Pydantic ``schema`` to enable per-field
    breakdown.  When no schema is provided, metrics are computed at the
    extraction level (class + text matching).

    Parameters:
        schema: Optional Pydantic ``BaseModel`` subclass. When set,
            the report includes a ``per_field`` breakdown mapping each
            model field to its own precision / recall / F1.
        strict_attributes: If ``True``, matching also considers
            attribute values (not just class + text).  Defaults to
            ``False``, so extractions match as long as their
            ``extraction_class`` and ``extraction_text`` agree (after
            normalisation).  Set ``True`` when you need to verify that
            attribute values also match exactly.
        averaging: Aggregation strategy for multi-document evaluation:

            * ``\"macro\"`` (default) — compute P/R/F1 across *all*
              extractions pooled together (flat).
            * ``\"micro\"`` — compute P/R/F1 per document first, then
              take the unweighted arithmetic mean.
            * ``\"weighted\"`` — compute P/R/F1 per document, then
              take a weighted mean where each document's weight is its
              ground-truth count.
        fuzzy_threshold: When set to a float in ``(0, 1]``, enables
            fuzzy string matching instead of exact equality.  Two
            extraction keys are considered a match when their
            ``difflib.SequenceMatcher`` ratio meets or exceeds this
            threshold.  ``None`` (default) uses exact matching.
    """

    def __init__(
        self,
        schema: type[pydantic.BaseModel] | None = None,
        *,
        strict_attributes: bool = False,
        averaging: AveragingMode = "macro",
        fuzzy_threshold: float | None = None,
    ) -> None:
        self._schema = schema
        self._strict = strict_attributes
        self._averaging: AveragingMode = averaging
        if fuzzy_threshold is not None and not 0 < fuzzy_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_threshold must be in (0, 1] or None, got {fuzzy_threshold}"
            )
        self._fuzzy = fuzzy_threshold

    # ------------------------------------------------------------------ #
    # Convenience static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def precision(
        predictions: Sequence[Extraction] | Sequence[list[Extraction]],
        ground_truth: Sequence[Extraction] | Sequence[list[Extraction]],
    ) -> float:
        """Compute precision between predictions and ground truth.

        Parameters:
            predictions: Predicted extractions (flat or per-document).
            ground_truth: Expected extractions (flat or per-document).

        Returns:
            Precision as a float in ``[0.0, 1.0]``.
        """
        pred = [_extraction_key(e) for e in _flatten(predictions)]
        gt = [_extraction_key(e) for e in _flatten(ground_truth)]
        p, _, _, _ = _compute_prf(pred, gt)
        return round(p, 4)

    @staticmethod
    def recall(
        predictions: Sequence[Extraction] | Sequence[list[Extraction]],
        ground_truth: Sequence[Extraction] | Sequence[list[Extraction]],
    ) -> float:
        """Compute recall between predictions and ground truth.

        Parameters:
            predictions: Predicted extractions (flat or per-document).
            ground_truth: Expected extractions (flat or per-document).

        Returns:
            Recall as a float in ``[0.0, 1.0]``.
        """
        pred = [_extraction_key(e) for e in _flatten(predictions)]
        gt = [_extraction_key(e) for e in _flatten(ground_truth)]
        _, r, _, _ = _compute_prf(pred, gt)
        return round(r, 4)

    @staticmethod
    def f1(
        predictions: Sequence[Extraction] | Sequence[list[Extraction]],
        ground_truth: Sequence[Extraction] | Sequence[list[Extraction]],
    ) -> float:
        """Compute F1 score between predictions and ground truth.

        Parameters:
            predictions: Predicted extractions (flat or per-document).
            ground_truth: Expected extractions (flat or per-document).

        Returns:
            F1 as a float in ``[0.0, 1.0]``.
        """
        pred = [_extraction_key(e) for e in _flatten(predictions)]
        gt = [_extraction_key(e) for e in _flatten(ground_truth)]
        _, _, f, _ = _compute_prf(pred, gt)
        return round(f, 4)

    @staticmethod
    def accuracy(
        predictions: Sequence[Extraction] | Sequence[list[Extraction]],
        ground_truth: Sequence[Extraction] | Sequence[list[Extraction]],
    ) -> float:
        """Compute accuracy (exact-match ratio against ground truth).

        Accuracy is defined as the fraction of ground-truth extractions
        that have at least one matching prediction.

        Parameters:
            predictions: Predicted extractions (flat or per-document).
            ground_truth: Expected extractions (flat or per-document).

        Returns:
            Accuracy as a float in ``[0.0, 1.0]``.
        """
        pred = [_extraction_key(e) for e in _flatten(predictions)]
        gt = [_extraction_key(e) for e in _flatten(ground_truth)]
        if not gt:
            return 1.0 if not pred else 0.0
        _, _, _, tp = _compute_prf(pred, gt)
        return round(tp / len(gt), 4)

    # ------------------------------------------------------------------ #
    # Full evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        predictions: (
            Sequence[Extraction]
            | Sequence[list[Extraction]]
            | AnnotatedDocument
            | Sequence[AnnotatedDocument]
        ),
        ground_truth: (
            Sequence[Extraction]
            | Sequence[list[Extraction]]
            | AnnotatedDocument
            | Sequence[AnnotatedDocument]
        ),
    ) -> EvaluationReport:
        """Run a full evaluation and return an ``EvaluationReport``.

        The ``averaging`` strategy (set in the constructor) controls how
        multi-document metrics are aggregated:

        * **macro** — pool all extractions, compute P/R/F1 once.
        * **micro** — compute P/R/F1 per document, average equally.
        * **weighted** — compute P/R/F1 per document, weighted by
          ground-truth count.

        Parameters:
            predictions: Predicted extractions in any supported shape
                (flat list, per-document lists, or ``AnnotatedDocument``).
            ground_truth: Expected extractions in any supported shape.

        Returns:
            An ``EvaluationReport`` containing aggregate and optional
            per-field metrics.
        """
        key_fn = _extraction_key_with_attrs if self._strict else _extraction_key
        fuzzy = self._fuzzy

        # ---- flat lists for totals ---- #
        pred_flat = _flatten(predictions)
        gt_flat = _flatten(ground_truth)

        # ---- per-document breakdown (always computed) ---- #
        pred_docs = _flatten_per_document(predictions)
        gt_docs = _flatten_per_document(ground_truth)

        per_doc: list[dict[str, float]] = []
        per_doc_tp = 0
        for pd_list, gt_list in zip(pred_docs, gt_docs):
            pk = [key_fn(e) for e in pd_list]
            gk = [key_fn(e) for e in gt_list]
            dp, dr, df, d_tp = _compute_prf(pk, gk, fuzzy_threshold=fuzzy)
            per_doc_tp += d_tp
            per_doc.append(
                {
                    "precision": round(dp, 4),
                    "recall": round(dr, 4),
                    "f1": round(df, 4),
                }
            )

        # ---- aggregate metrics ---- #
        if self._averaging == "macro":
            pred_keys = [key_fn(e) for e in pred_flat]
            gt_keys = [key_fn(e) for e in gt_flat]
            prec, rec, f1_val, tp = _compute_prf(
                pred_keys, gt_keys, fuzzy_threshold=fuzzy
            )
        elif self._averaging in ("micro", "weighted"):
            # Per-document average
            if not per_doc:
                prec, rec, f1_val, tp = 0.0, 0.0, 0.0, 0
            else:
                if self._averaging == "micro":
                    # Unweighted mean of per-document scores
                    prec = sum(d["precision"] for d in per_doc) / len(per_doc)
                    rec = sum(d["recall"] for d in per_doc) / len(per_doc)
                    f1_val = sum(d["f1"] for d in per_doc) / len(per_doc)
                else:
                    # Weighted by ground-truth count
                    total_weight = sum(len(gt_list) for gt_list in gt_docs)
                    if total_weight == 0:
                        prec, rec, f1_val = 0.0, 0.0, 0.0
                    else:
                        prec = (
                            sum(
                                d["precision"] * len(gt_list)
                                for d, gt_list in zip(per_doc, gt_docs)
                            )
                            / total_weight
                        )
                        rec = (
                            sum(
                                d["recall"] * len(gt_list)
                                for d, gt_list in zip(per_doc, gt_docs)
                            )
                            / total_weight
                        )
                        f1_val = (
                            sum(
                                d["f1"] * len(gt_list)
                                for d, gt_list in zip(per_doc, gt_docs)
                            )
                            / total_weight
                        )
                tp = per_doc_tp
        else:
            raise ValueError(f"Unknown averaging mode: {self._averaging!r}")

        acc = tp / len(gt_flat) if gt_flat else (1.0 if not pred_flat else 0.0)

        # ---- per-field ---- #
        per_field = self._compute_per_field(pred_flat, gt_flat)

        return EvaluationReport(
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1_val, 4),
            accuracy=round(acc, 4),
            total_predictions=len(pred_flat),
            total_ground_truth=len(gt_flat),
            true_positives=tp,
            averaging=self._averaging,
            per_field=per_field,
            per_document=per_doc,
        )

    # ------------------------------------------------------------------ #
    # Per-field breakdown
    # ------------------------------------------------------------------ #

    def _compute_per_field(
        self,
        pred_flat: list[Extraction],
        gt_flat: list[Extraction],
    ) -> dict[str, FieldReport]:
        """Compute per-field precision / recall / F1.

        When a Pydantic schema is set, only the schema's fields (plus
        ``extraction_text``) are evaluated.  Otherwise, all attribute
        keys that appear in the ground truth are included.
        """
        # Determine which field names to evaluate
        if self._schema is not None:
            field_names = list(self._schema.model_fields.keys())
            # Always include extraction_text (the primary text field)
            if "extraction_text" not in field_names:
                field_names = ["extraction_text", *field_names]
        else:
            # Collect every attribute key that appears in predictions
            # or ground truth, plus the standard fields.
            attr_keys: set[str] = set()
            for ext in (*pred_flat, *gt_flat):
                if ext.attributes:
                    attr_keys.update(ext.attributes.keys())
            field_names = [
                "extraction_class",
                "extraction_text",
                *sorted(attr_keys),
            ]

        reports: dict[str, FieldReport] = {}
        for fname in field_names:
            pred_vals = self._field_values(pred_flat, fname)
            gt_vals = self._field_values(gt_flat, fname)
            p, r, f, _tp = _compute_prf(pred_vals, gt_vals)
            reports[fname] = FieldReport(
                field_name=fname,
                precision=round(p, 4),
                recall=round(r, 4),
                f1=round(f, 4),
                support=len(gt_vals),
            )

        return reports

    @staticmethod
    def _field_values(
        extractions: list[Extraction],
        field_name: str,
    ) -> list[str]:
        """Extract normalised values for a single field name.

        Maps ``extraction_text`` and ``extraction_class`` to the
        corresponding ``Extraction`` attribute; all other names are
        looked up in ``extraction.attributes``.
        """
        values: list[str] = []
        for ext in extractions:
            if field_name == "extraction_text":
                val = " ".join((ext.extraction_text or "").split()).lower()
            elif field_name == "extraction_class":
                val = (ext.extraction_class or "").strip().lower()
            elif ext.attributes and field_name in ext.attributes:
                val = str(ext.attributes[field_name]).strip().lower()
            else:
                continue  # field not present on this extraction
            values.append(val)
        return values
