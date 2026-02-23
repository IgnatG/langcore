"""Tests for langcore.evaluation module."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from langcore.core.data import AnnotatedDocument, Extraction
from langcore.evaluation import (
    EvaluationReport,
    ExtractionMetrics,
    FieldReport,
    _compute_prf,
    _extraction_key,
    _extraction_key_with_attrs,
    _flatten,
    _flatten_per_document,
)

# ================================================================== #
# Fixtures / helpers
# ================================================================== #


def _ext(cls: str, text: str, **attrs: str) -> Extraction:
    """Shorthand for building an ``Extraction``."""
    return Extraction(
        cls,
        text,
        attributes=attrs if attrs else None,
    )


class Invoice(BaseModel):
    """Sample Pydantic schema for field-level tests."""

    invoice_number: str = Field(description="Invoice ID")
    amount: str = Field(description="Total amount")
    due_date: str = Field(description="Due date YYYY-MM-DD")


# ================================================================== #
# _extraction_key
# ================================================================== #


class TestExtractionKey:
    """Unit tests for the key-building helpers."""

    def test_basic_key(self) -> None:
        ext = _ext("Invoice", "INV-001")
        assert _extraction_key(ext) == "invoice|inv-001"

    def test_whitespace_normalisation(self) -> None:
        ext = _ext("  Invoice ", "  INV  001  ")
        assert _extraction_key(ext) == "invoice|inv 001"

    def test_case_insensitive(self) -> None:
        a = _ext("Entity", "Hello World")
        b = _ext("ENTITY", "hello world")
        assert _extraction_key(a) == _extraction_key(b)

    def test_key_with_attrs(self) -> None:
        ext = _ext("Invoice", "INV-001", amount="500", due="2024-01-01")
        key = _extraction_key_with_attrs(ext)
        assert "amount=500" in key
        assert "due=2024-01-01" in key
        # Attrs are sorted
        assert key.index("amount") < key.index("due")

    def test_key_with_attrs_no_attributes(self) -> None:
        ext = _ext("Invoice", "INV-001")
        assert _extraction_key_with_attrs(ext) == _extraction_key(ext)


# ================================================================== #
# _flatten
# ================================================================== #


class TestFlatten:
    """Unit tests for _flatten and _flatten_per_document."""

    def test_flat_list(self) -> None:
        exts = [_ext("A", "a"), _ext("B", "b")]
        assert _flatten(exts) == exts

    def test_nested_lists(self) -> None:
        doc1 = [_ext("A", "a")]
        doc2 = [_ext("B", "b"), _ext("C", "c")]
        flat = _flatten([doc1, doc2])
        assert len(flat) == 3

    def test_annotated_document(self) -> None:
        doc = AnnotatedDocument(
            extractions=[_ext("A", "a"), _ext("B", "b")],
        )
        assert len(_flatten(doc)) == 2

    def test_list_of_annotated_documents(self) -> None:
        docs = [
            AnnotatedDocument(extractions=[_ext("A", "a")]),
            AnnotatedDocument(extractions=[_ext("B", "b")]),
        ]
        assert len(_flatten(docs)) == 2

    def test_annotated_document_none_extractions(self) -> None:
        doc = AnnotatedDocument(extractions=None)
        assert _flatten(doc) == []

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            _flatten(["not_an_extraction"])  # type: ignore[arg-type]

    def test_flatten_per_document_flat_list(self) -> None:
        exts = [_ext("A", "a"), _ext("B", "b")]
        groups = _flatten_per_document(exts)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_flatten_per_document_nested(self) -> None:
        doc1 = [_ext("A", "a")]
        doc2 = [_ext("B", "b")]
        groups = _flatten_per_document([doc1, doc2])
        assert len(groups) == 2


# ================================================================== #
# _compute_prf
# ================================================================== #


class TestComputePRF:
    """Unit tests for the core P/R/F1 computation."""

    def test_perfect_match(self) -> None:
        keys = ["a", "b", "c"]
        p, r, f, tp = _compute_prf(keys, keys)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0
        assert tp == 3

    def test_no_match(self) -> None:
        p, r, f, tp = _compute_prf(["a"], ["b"])
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0
        assert tp == 0

    def test_partial_match(self) -> None:
        p, r, _f, tp = _compute_prf(["a", "b"], ["a", "c"])
        assert tp == 1
        assert p == 0.5
        assert r == 0.5

    def test_empty_predictions(self) -> None:
        p, r, f, _tp = _compute_prf([], ["a"])
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

    def test_empty_ground_truth(self) -> None:
        p, r, _f, _tp = _compute_prf(["a"], [])
        assert p == 0.0
        assert r == 0.0

    def test_both_empty(self) -> None:
        p, r, f, tp = _compute_prf([], [])
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0
        assert tp == 0

    def test_duplicate_handling(self) -> None:
        """Each GT key matched at most once (bag matching)."""
        p, r, _f, tp = _compute_prf(["a", "a"], ["a"])
        assert tp == 1
        assert p == 0.5  # 1/2 predictions matched
        assert r == 1.0  # 1/1 GT matched


# ================================================================== #
# ExtractionMetrics static helpers
# ================================================================== #


class TestStaticHelpers:
    """Test the convenience static methods on ExtractionMetrics."""

    def test_precision_perfect(self) -> None:
        exts = [_ext("A", "a")]
        assert ExtractionMetrics.precision(exts, exts) == 1.0

    def test_recall_perfect(self) -> None:
        exts = [_ext("A", "a")]
        assert ExtractionMetrics.recall(exts, exts) == 1.0

    def test_f1_perfect(self) -> None:
        exts = [_ext("A", "a")]
        assert ExtractionMetrics.f1(exts, exts) == 1.0

    def test_accuracy_perfect(self) -> None:
        exts = [_ext("A", "a")]
        assert ExtractionMetrics.accuracy(exts, exts) == 1.0

    def test_precision_no_match(self) -> None:
        pred = [_ext("A", "a")]
        gt = [_ext("B", "b")]
        assert ExtractionMetrics.precision(pred, gt) == 0.0

    def test_accuracy_no_gt(self) -> None:
        """No GT and no predictions → accuracy 1.0."""
        assert ExtractionMetrics.accuracy([], []) == 1.0

    def test_accuracy_extra_predictions(self) -> None:
        """No GT but predictions → accuracy 0.0."""
        assert ExtractionMetrics.accuracy([_ext("A", "a")], []) == 0.0


# ================================================================== #
# ExtractionMetrics.evaluate — aggregate
# ================================================================== #


class TestEvaluateAggregate:
    """Test full evaluate() aggregate metrics."""

    def test_perfect_match(self) -> None:
        preds = [_ext("Invoice", "INV-001"), _ext("Invoice", "INV-002")]
        gt = [_ext("Invoice", "INV-001"), _ext("Invoice", "INV-002")]

        metrics = ExtractionMetrics()
        report = metrics.evaluate(preds, gt)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.accuracy == 1.0
        assert report.true_positives == 2

    def test_no_match(self) -> None:
        preds = [_ext("Invoice", "INV-001")]
        gt = [_ext("Invoice", "INV-999")]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert report.f1 == 0.0
        assert report.true_positives == 0

    def test_partial_match(self) -> None:
        preds = [_ext("A", "x"), _ext("A", "y")]
        gt = [_ext("A", "x"), _ext("A", "z")]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert report.true_positives == 1
        assert report.precision == 0.5
        assert report.recall == 0.5

    def test_annotated_document_input(self) -> None:
        pred_doc = AnnotatedDocument(
            extractions=[_ext("A", "x"), _ext("A", "y")],
        )
        gt_doc = AnnotatedDocument(
            extractions=[_ext("A", "x"), _ext("A", "y")],
        )
        report = ExtractionMetrics().evaluate(pred_doc, gt_doc)
        assert report.f1 == 1.0

    def test_list_of_annotated_documents(self) -> None:
        preds = [
            AnnotatedDocument(extractions=[_ext("A", "x")]),
            AnnotatedDocument(extractions=[_ext("B", "y")]),
        ]
        gt = [
            AnnotatedDocument(extractions=[_ext("A", "x")]),
            AnnotatedDocument(extractions=[_ext("B", "y")]),
        ]
        report = ExtractionMetrics().evaluate(preds, gt)
        assert report.f1 == 1.0
        assert report.total_predictions == 2
        assert report.total_ground_truth == 2

    def test_strict_attributes_mode(self) -> None:
        pred = [_ext("A", "x", color="red")]
        gt = [_ext("A", "x", color="blue")]

        # Without strict: matches on class+text only
        report = ExtractionMetrics().evaluate(pred, gt)
        assert report.true_positives == 1

        # With strict: also checks attributes
        report_strict = ExtractionMetrics(
            strict_attributes=True,
        ).evaluate(pred, gt)
        assert report_strict.true_positives == 0

    def test_counts(self) -> None:
        preds = [_ext("A", "1"), _ext("A", "2"), _ext("A", "3")]
        gt = [_ext("A", "1"), _ext("A", "2")]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert report.total_predictions == 3
        assert report.total_ground_truth == 2


# ================================================================== #
# ExtractionMetrics.evaluate — per-document
# ================================================================== #


class TestEvaluatePerDocument:
    """Test per-document breakdown in the report."""

    def test_per_document_breakdown(self) -> None:
        preds = [[_ext("A", "x")], [_ext("B", "y"), _ext("B", "z")]]
        gt = [[_ext("A", "x")], [_ext("B", "y")]]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert len(report.per_document) == 2
        assert report.per_document[0]["f1"] == 1.0
        # Doc 2: pred has extra "z"
        assert report.per_document[1]["recall"] == 1.0
        assert report.per_document[1]["precision"] == 0.5


# ================================================================== #
# ExtractionMetrics.evaluate — per-field
# ================================================================== #


class TestEvaluatePerField:
    """Test per-field breakdown."""

    def test_per_field_no_schema(self) -> None:
        """Without schema, fields derive from attribute keys."""
        preds = [_ext("Invoice", "INV-001", amount="500")]
        gt = [_ext("Invoice", "INV-001", amount="500")]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert "extraction_class" in report.per_field
        assert "extraction_text" in report.per_field
        assert "amount" in report.per_field
        assert report.per_field["amount"].f1 == 1.0

    def test_per_field_with_schema(self) -> None:
        """With a Pydantic schema, only schema fields appear."""
        preds = [
            _ext(
                "Invoice",
                "INV-001",
                invoice_number="INV-001",
                amount="500",
                due_date="2024-01-01",
            ),
        ]
        gt = [
            _ext(
                "Invoice",
                "INV-001",
                invoice_number="INV-001",
                amount="500",
                due_date="2024-01-01",
            ),
        ]

        report = ExtractionMetrics(schema=Invoice).evaluate(preds, gt)
        # Schema fields + extraction_text
        assert "invoice_number" in report.per_field
        assert "amount" in report.per_field
        assert "due_date" in report.per_field
        assert "extraction_text" in report.per_field
        # Should NOT include extraction_class (not in schema)
        assert "extraction_class" not in report.per_field

        for fr in report.per_field.values():
            assert fr.f1 == 1.0

    def test_per_field_partial_match(self) -> None:
        """Field that mismatches should have lower scores."""
        preds = [
            _ext("Invoice", "INV-001", amount="500", due_date="2024-01-01"),
        ]
        gt = [
            _ext("Invoice", "INV-001", amount="999", due_date="2024-01-01"),
        ]

        report = ExtractionMetrics().evaluate(preds, gt)
        # extraction_text matches
        assert report.per_field["extraction_text"].f1 == 1.0
        # amount differs
        assert report.per_field["amount"].f1 == 0.0
        # due_date matches
        assert report.per_field["due_date"].f1 == 1.0

    def test_per_field_support_count(self) -> None:
        gt = [
            _ext("A", "x", color="red"),
            _ext("A", "y", color="blue"),
        ]
        preds = [_ext("A", "x", color="red")]

        report = ExtractionMetrics().evaluate(preds, gt)
        assert report.per_field["color"].support == 2


# ================================================================== #
# EvaluationReport / FieldReport dataclasses
# ================================================================== #


class TestDataclasses:
    """Basic tests for the report dataclasses."""

    def test_evaluation_report_defaults(self) -> None:
        report = EvaluationReport(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            accuracy=0.8,
            total_predictions=10,
            total_ground_truth=10,
            true_positives=8,
        )
        assert report.per_field == {}
        assert report.per_document == []

    def test_field_report_attrs(self) -> None:
        fr = FieldReport(
            field_name="amount",
            precision=0.95,
            recall=0.90,
            f1=0.924,
            support=20,
        )
        assert fr.field_name == "amount"
        assert fr.support == 20


# ================================================================== #
# lx.evaluate convenience
# ================================================================== #


class TestLxEvaluate:
    """Test the top-level lx.evaluate() convenience function."""

    def test_lx_evaluate_basic(self) -> None:
        import langcore as lx

        preds = [_ext("A", "x")]
        gt = [_ext("A", "x")]
        report = lx.evaluate(predictions=preds, ground_truth=gt)
        assert isinstance(report, EvaluationReport)
        assert report.f1 == 1.0

    def test_lx_evaluate_with_schema(self) -> None:
        import langcore as lx

        preds = [_ext("I", "inv", invoice_number="001", amount="5", due_date="d")]
        gt = [_ext("I", "inv", invoice_number="001", amount="5", due_date="d")]
        report = lx.evaluate(
            predictions=preds,
            ground_truth=gt,
            schema=Invoice,
        )
        assert "invoice_number" in report.per_field

    def test_lx_evaluate_strict(self) -> None:
        import langcore as lx

        preds = [_ext("A", "x", color="red")]
        gt = [_ext("A", "x", color="blue")]
        report = lx.evaluate(
            predictions=preds,
            ground_truth=gt,
            strict_attributes=True,
        )
        assert report.true_positives == 0
