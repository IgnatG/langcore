"""Tests for composite reliability scoring (``reliability.py``).

Covers:
- ``compute_reliability_score`` per-extraction scoring
- ``compute_reliability_scores`` batch scoring on AnnotatedDocument
- ``AnnotatedDocument.average_reliability`` property
- ``ReliabilityConfig`` custom weights
- Schema validity and field completeness signals
- Source grounding signal
"""

from __future__ import annotations

import pydantic
from absl.testing import absltest, parameterized

from langcore.core import data
from langcore.reliability import (
    ReliabilityConfig,
    compute_reliability_score,
    compute_reliability_scores,
)

# ── Pydantic test schemas ────────────────────────────────────────


class Invoice(pydantic.BaseModel):
    text: str
    amount: float
    currency: str = "USD"  # optional with default


class Person(pydantic.BaseModel):
    name: str
    age: int


# ── Helpers ───────────────────────────────────────────────────────


def _make_extraction(
    *,
    text: str = "test",
    cls: str = "Invoice",
    confidence: float | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    alignment: data.AlignmentStatus | None = None,
    attributes: dict | None = None,
) -> data.Extraction:
    ci = None
    if char_start is not None or char_end is not None:
        ci = data.CharInterval(start_pos=char_start, end_pos=char_end)
    return data.Extraction(
        extraction_class=cls,
        extraction_text=text,
        confidence_score=confidence,
        char_interval=ci,
        alignment_status=alignment,
        attributes=attributes,
    )


# ═══════════════════════════════════════════════════════════════════
# Individual signal tests
# ═══════════════════════════════════════════════════════════════════


class GroundingSignalTest(parameterized.TestCase):
    """Test the source-grounding component of reliability."""

    def test_no_char_interval_gives_zero_grounding(self):
        ext = _make_extraction(confidence=0.9)
        # With only grounding weight
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=0, w_grounding=1.0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_valid_char_interval_gives_full_grounding(self):
        ext = _make_extraction(confidence=0.9, char_start=0, char_end=10)
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=0, w_grounding=1.0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_degenerate_interval_gives_partial_grounding(self):
        ext = _make_extraction(confidence=0.9, char_start=5, char_end=5)
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=0, w_grounding=1.0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_none_positions_gives_partial_grounding(self):
        ext = _make_extraction(confidence=0.9)
        ext.char_interval = data.CharInterval(start_pos=None, end_pos=None)
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=0, w_grounding=1.0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.5, places=4)


class SchemaValiditySignalTest(parameterized.TestCase):
    """Test the schema-validity component of reliability."""

    def test_valid_extraction_scores_1(self):
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            attributes={"amount": "42.0", "currency": "EUR"},
        )
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=1.0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_invalid_extraction_scores_0(self):
        # amount must be float — "not-a-number" will fail
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            attributes={"amount": "not-a-number"},
        )
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=1.0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_no_schema_is_neutral(self):
        ext = _make_extraction(text="test")
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=1.0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(ext, schema=None, config=cfg)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_mismatched_class_is_neutral(self):
        ext = _make_extraction(text="test", cls="Receipt")
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=1.0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        self.assertAlmostEqual(score, 1.0, places=4)


class FieldCompletenessSignalTest(parameterized.TestCase):
    """Test the field-completeness component of reliability."""

    def test_all_required_fields_present(self):
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            attributes={"amount": "42.0"},
        )
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=1.0, w_grounding=0
        )
        # Invoice has 2 required: text (via primary), amount
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_missing_required_field(self):
        # Missing 'amount' — only text is provided
        ext = _make_extraction(text="widget", cls="Invoice")
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=1.0, w_grounding=0
        )
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        # text present, amount missing → 1/2 = 0.5
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_empty_string_counts_as_missing(self):
        ext = _make_extraction(text="widget", cls="Invoice", attributes={"amount": ""})
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=1.0, w_grounding=0
        )
        score = compute_reliability_score(
            ext, schema=Invoice, primary_field="text", config=cfg
        )
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_no_schema_is_neutral(self):
        ext = _make_extraction(text="test")
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=1.0, w_grounding=0
        )
        score = compute_reliability_score(ext, schema=None, config=cfg)
        self.assertAlmostEqual(score, 1.0, places=4)


class ConfidenceSignalTest(parameterized.TestCase):
    """Test the confidence component of reliability."""

    def test_confidence_propagated(self):
        ext = _make_extraction(confidence=0.85)
        cfg = ReliabilityConfig(
            w_confidence=1.0, w_schema_valid=0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.85, places=4)

    def test_none_confidence_treated_as_zero(self):
        ext = _make_extraction(confidence=None)
        cfg = ReliabilityConfig(
            w_confidence=1.0, w_schema_valid=0, w_completeness=0, w_grounding=0
        )
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.0, places=4)


# ═══════════════════════════════════════════════════════════════════
# Composite scoring tests
# ═══════════════════════════════════════════════════════════════════


class CompositeReliabilityTest(parameterized.TestCase):
    """Test the combined reliability score with default weights."""

    def test_perfect_extraction(self):
        """All signals maxed → score = 1.0."""
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            confidence=1.0,
            char_start=0,
            char_end=10,
            attributes={"amount": "42.0", "currency": "EUR"},
        )
        score = compute_reliability_score(ext, schema=Invoice, primary_field="text")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_no_confidence_no_grounding(self):
        """Confidence=0, grounding=0, but schema valid and complete."""
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            confidence=None,
            attributes={"amount": "42.0"},
        )
        # Default weights: 0.4*0 + 0.2*1 + 0.2*1 + 0.2*0 = 0.4
        score = compute_reliability_score(ext, schema=Invoice, primary_field="text")
        self.assertAlmostEqual(score, 0.4, places=4)

    def test_default_weights_sum(self):
        """Verify default weights give expected composite."""
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            confidence=0.8,
            char_start=0,
            char_end=10,
            attributes={"amount": "42.0"},
        )
        # conf=0.8, valid=1, complete=1, grounded=1
        # 0.4*0.8 + 0.2*1 + 0.2*1 + 0.2*1 = 0.32 + 0.6 = 0.92
        score = compute_reliability_score(ext, schema=Invoice, primary_field="text")
        self.assertAlmostEqual(score, 0.92, places=4)


class CustomWeightsTest(parameterized.TestCase):
    """Test ReliabilityConfig with custom weights."""

    def test_confidence_only(self):
        cfg = ReliabilityConfig(
            w_confidence=1.0, w_schema_valid=0, w_completeness=0, w_grounding=0
        )
        ext = _make_extraction(confidence=0.75)
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.75, places=4)

    def test_equal_weights(self):
        cfg = ReliabilityConfig(
            w_confidence=1, w_schema_valid=1, w_completeness=1, w_grounding=1
        )
        ext = _make_extraction(
            text="John",
            cls="Person",
            confidence=0.6,
            char_start=0,
            char_end=4,
            attributes={"age": "30"},
        )
        # conf=0.6, valid=1, complete=1, grounded=1 → (0.6+1+1+1)/4 = 0.9
        score = compute_reliability_score(
            ext, schema=Person, primary_field="name", config=cfg
        )
        self.assertAlmostEqual(score, 0.9, places=4)

    def test_all_zero_weights(self):
        cfg = ReliabilityConfig(
            w_confidence=0, w_schema_valid=0, w_completeness=0, w_grounding=0
        )
        ext = _make_extraction(confidence=0.9)
        score = compute_reliability_score(ext, config=cfg)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_unnormalised_weights(self):
        """Weights that don't sum to 1 should be normalised."""
        cfg = ReliabilityConfig(
            w_confidence=2.0, w_schema_valid=0, w_completeness=0, w_grounding=0
        )
        ext = _make_extraction(confidence=0.5)
        score = compute_reliability_score(ext, config=cfg)
        # 2.0*0.5 / 2.0 = 0.5
        self.assertAlmostEqual(score, 0.5, places=4)


# ═══════════════════════════════════════════════════════════════════
# Batch scoring & AnnotatedDocument
# ═══════════════════════════════════════════════════════════════════


class ComputeReliabilityScoresTest(absltest.TestCase):
    """Test ``compute_reliability_scores`` on AnnotatedDocument."""

    def test_scores_set_on_all_extractions(self):
        ext1 = _make_extraction(confidence=0.9, char_start=0, char_end=5)
        ext2 = _make_extraction(confidence=0.5, char_start=10, char_end=20)
        doc = data.AnnotatedDocument(extractions=[ext1, ext2])
        compute_reliability_scores(doc)
        self.assertIsNotNone(ext1.reliability_score)
        self.assertIsNotNone(ext2.reliability_score)
        # Higher confidence → higher reliability
        self.assertGreater(ext1.reliability_score, ext2.reliability_score)

    def test_empty_extractions(self):
        doc = data.AnnotatedDocument(extractions=[])
        compute_reliability_scores(doc)  # Should not raise

    def test_none_extractions(self):
        doc = data.AnnotatedDocument(extractions=None)
        compute_reliability_scores(doc)  # Should not raise

    def test_with_schema(self):
        ext = _make_extraction(
            text="widget",
            cls="Invoice",
            confidence=0.8,
            char_start=0,
            char_end=10,
            attributes={"amount": "42.0"},
        )
        doc = data.AnnotatedDocument(extractions=[ext])
        compute_reliability_scores(doc, schema=Invoice)
        self.assertIsNotNone(ext.reliability_score)
        self.assertGreater(ext.reliability_score, 0.5)


class AverageReliabilityTest(absltest.TestCase):
    """Test ``AnnotatedDocument.average_reliability`` property."""

    def test_average_of_scored_extractions(self):
        ext1 = _make_extraction(confidence=1.0, char_start=0, char_end=5)
        ext2 = _make_extraction(confidence=0.5, char_start=10, char_end=20)
        doc = data.AnnotatedDocument(extractions=[ext1, ext2])
        compute_reliability_scores(doc)
        avg = doc.average_reliability
        self.assertIsNotNone(avg)
        self.assertGreater(avg, 0.0)
        self.assertLessEqual(avg, 1.0)

    def test_none_when_no_extractions(self):
        doc = data.AnnotatedDocument(extractions=None)
        self.assertIsNone(doc.average_reliability)

    def test_none_when_empty_extractions(self):
        doc = data.AnnotatedDocument(extractions=[])
        self.assertIsNone(doc.average_reliability)

    def test_none_when_no_scores_set(self):
        ext = _make_extraction(confidence=0.5)
        doc = data.AnnotatedDocument(extractions=[ext])
        # Don't call compute_reliability_scores
        self.assertIsNone(doc.average_reliability)


class ReliabilityScoreFieldTest(absltest.TestCase):
    """Test the reliability_score field on Extraction."""

    def test_default_is_none(self):
        ext = data.Extraction(extraction_class="Test", extraction_text="test")
        self.assertIsNone(ext.reliability_score)

    def test_can_set_in_constructor(self):
        ext = data.Extraction(
            extraction_class="Test",
            extraction_text="test",
            reliability_score=0.75,
        )
        self.assertAlmostEqual(ext.reliability_score, 0.75)

    def test_can_set_after_construction(self):
        ext = data.Extraction(extraction_class="Test", extraction_text="test")
        ext.reliability_score = 0.42
        self.assertAlmostEqual(ext.reliability_score, 0.42)


if __name__ == "__main__":
    absltest.main()
