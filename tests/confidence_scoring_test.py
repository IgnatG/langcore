"""Tests for per-extraction confidence scoring (Phase 2).

Covers:
- ``compute_alignment_confidence`` (resolver.py)
- ``Resolver.align`` confidence propagation
- ``AnnotatedDocument.average_confidence`` property
- Multi-pass confidence augmentation
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from langcore import annotation, resolver
from langcore.core import data
from langcore.core import tokenizer as tokenizer_lib


class ComputeAlignmentConfidenceTest(parameterized.TestCase):
    """Unit tests for ``compute_alignment_confidence``."""

    @parameterized.parameters(
        (data.AlignmentStatus.MATCH_EXACT, 1.0),
        (data.AlignmentStatus.MATCH_LESSER, 0.8),
        (data.AlignmentStatus.MATCH_GREATER, 0.7),
        (data.AlignmentStatus.MATCH_FUZZY, 0.5),
        (None, 0.2),
    )
    def test_alignment_quality_without_token_interval(
        self,
        status: data.AlignmentStatus | None,
        expected_quality: float,
    ):
        """When no token interval is set, confidence equals the
        weighted sum using alignment quality for both components."""
        ext = data.Extraction(
            extraction_class="entity",
            extraction_text="hello world",
            alignment_status=status,
        )
        score = resolver.compute_alignment_confidence(ext)
        # overlap_ratio defaults to alignment_quality when no token info
        expected = round(0.7 * expected_quality + 0.3 * expected_quality, 4)
        self.assertAlmostEqual(score, expected, places=4)

    def test_exact_match_with_perfect_token_overlap(self):
        """MATCH_EXACT + tokens span that matches extraction tokens
        exactly should produce confidence ~1.0."""
        ext = data.Extraction(
            extraction_class="entity",
            extraction_text="hello world",
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
            token_interval=tokenizer_lib.TokenInterval(start_index=0, end_index=2),
        )
        score = resolver.compute_alignment_confidence(ext)
        # alignment_quality = 1.0, overlap = 2/2 = 1.0
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_exact_match_with_wider_token_span(self):
        """When the source span is wider than extraction tokens, the
        overlap ratio decreases."""
        ext = data.Extraction(
            extraction_class="entity",
            extraction_text="hello",
            alignment_status=data.AlignmentStatus.MATCH_EXACT,
            token_interval=tokenizer_lib.TokenInterval(start_index=0, end_index=4),
        )
        score = resolver.compute_alignment_confidence(ext)
        # alignment_quality = 1.0, overlap = 1/4 = 0.25
        expected = round(0.7 * 1.0 + 0.3 * 0.25, 4)
        self.assertAlmostEqual(score, expected, places=4)

    def test_fuzzy_match_with_token_overlap(self):
        """Fuzzy match with moderate overlap."""
        ext = data.Extraction(
            extraction_class="entity",
            extraction_text="hello world",
            alignment_status=data.AlignmentStatus.MATCH_FUZZY,
            token_interval=tokenizer_lib.TokenInterval(start_index=0, end_index=3),
        )
        score = resolver.compute_alignment_confidence(ext)
        # alignment_quality = 0.5, overlap = 2/3 ≈ 0.6667
        expected = round(0.7 * 0.5 + 0.3 * (2 / 3), 4)
        self.assertAlmostEqual(score, expected, places=4)

    def test_unaligned_extraction(self):
        """Extraction with no alignment status gets low confidence."""
        ext = data.Extraction(
            extraction_class="entity",
            extraction_text="unknown",
            alignment_status=None,
        )
        score = resolver.compute_alignment_confidence(ext)
        expected = round(0.7 * 0.2 + 0.3 * 0.2, 4)
        self.assertAlmostEqual(score, expected, places=4)

    def test_score_always_between_0_and_1(self):
        """Confidence score is always in [0.0, 1.0]."""
        for status in [*list(data.AlignmentStatus), None]:
            for span_size in [1, 2, 5, 10]:
                ext = data.Extraction(
                    extraction_class="e",
                    extraction_text="a",
                    alignment_status=status,
                    token_interval=tokenizer_lib.TokenInterval(
                        start_index=0, end_index=span_size
                    ),
                )
                score = resolver.compute_alignment_confidence(ext)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


class ResolverAlignConfidenceTest(absltest.TestCase):
    """Integration tests for confidence being set during Resolver.align."""

    def test_align_sets_confidence_on_extractions(self):
        """After alignment, every extraction has a non-null confidence."""
        source_text = "The quick brown fox jumps over the lazy dog"
        extractions = [
            data.Extraction(
                extraction_class="animal",
                extraction_text="fox",
            ),
            data.Extraction(
                extraction_class="animal",
                extraction_text="dog",
            ),
        ]
        res = resolver.Resolver()
        aligned = list(
            res.align(
                extractions,
                source_text,
                token_offset=0,
                char_offset=0,
            )
        )
        self.assertLen(aligned, 2)
        for ext in aligned:
            self.assertIsNotNone(ext.confidence_score)
            self.assertGreater(ext.confidence_score, 0.0)
            self.assertLessEqual(ext.confidence_score, 1.0)

    def test_exact_match_gets_high_confidence(self):
        """An exact token match should get confidence close to 1.0."""
        source_text = "hello world"
        extractions = [
            data.Extraction(
                extraction_class="greeting",
                extraction_text="hello",
            ),
        ]
        res = resolver.Resolver()
        aligned = list(res.align(extractions, source_text, token_offset=0))
        self.assertLen(aligned, 1)
        self.assertIsNotNone(aligned[0].alignment_status)
        self.assertGreater(aligned[0].confidence_score, 0.8)

    def test_empty_extractions_returns_empty(self):
        """Empty input produces no output."""
        res = resolver.Resolver()
        aligned = list(res.align([], "some text", token_offset=0))
        self.assertEmpty(aligned)


class AnnotatedDocumentAverageConfidenceTest(absltest.TestCase):
    """Tests for ``AnnotatedDocument.average_confidence`` property."""

    def test_no_extractions_returns_none(self):
        """No extractions means average_confidence is None."""
        doc = data.AnnotatedDocument(extractions=None)
        self.assertIsNone(doc.average_confidence)

    def test_empty_extractions_returns_none(self):
        """Empty extractions list means average_confidence is None."""
        doc = data.AnnotatedDocument(extractions=[])
        self.assertIsNone(doc.average_confidence)

    def test_all_none_scores_returns_none(self):
        """If all extractions have None confidence, average is None."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(extraction_class="e", extraction_text="a"),
                data.Extraction(extraction_class="e", extraction_text="b"),
            ]
        )
        self.assertIsNone(doc.average_confidence)

    def test_average_of_known_scores(self):
        """Arithmetic mean of known confidence scores."""
        e1 = data.Extraction(
            extraction_class="e",
            extraction_text="a",
            confidence_score=1.0,
        )
        e2 = data.Extraction(
            extraction_class="e",
            extraction_text="b",
            confidence_score=0.5,
        )
        doc = data.AnnotatedDocument(extractions=[e1, e2])
        self.assertAlmostEqual(doc.average_confidence, 0.75, places=4)

    def test_mixed_none_and_scored(self):
        """None scores are excluded from the average."""
        e1 = data.Extraction(
            extraction_class="e",
            extraction_text="a",
            confidence_score=0.8,
        )
        e2 = data.Extraction(
            extraction_class="e",
            extraction_text="b",
        )
        e3 = data.Extraction(
            extraction_class="e",
            extraction_text="c",
            confidence_score=0.6,
        )
        doc = data.AnnotatedDocument(extractions=[e1, e2, e3])
        self.assertAlmostEqual(doc.average_confidence, 0.7, places=4)

    def test_single_extraction(self):
        """Single extraction's score is the average."""
        e1 = data.Extraction(
            extraction_class="e",
            extraction_text="x",
            confidence_score=0.9,
        )
        doc = data.AnnotatedDocument(extractions=[e1])
        self.assertAlmostEqual(doc.average_confidence, 0.9, places=4)


class MultiPassConfidenceAugmentationTest(absltest.TestCase):
    """Tests for multi-pass confidence combining alignment + cross-pass."""

    def test_single_pass_preserves_alignment_confidence(self):
        """With total_passes=1, alignment confidence is NOT overwritten."""
        ext = data.Extraction(
            extraction_class="e",
            extraction_text="hello",
            confidence_score=0.95,
            char_interval=data.CharInterval(start_pos=0, end_pos=5),
        )
        result = annotation._merge_non_overlapping_extractions([[ext]], total_passes=1)
        # total_passes=1 skips confidence scoring, preserving the
        # value set by Resolver.align.
        self.assertAlmostEqual(result[0].confidence_score, 0.95, places=4)

    def test_multi_pass_combines_scores(self):
        """Multi-pass merges cross-pass frequency with alignment conf."""
        ext_pass1 = data.Extraction(
            extraction_class="e",
            extraction_text="hello",
            confidence_score=0.9,
            char_interval=data.CharInterval(start_pos=0, end_pos=5),
        )
        ext_pass2 = data.Extraction(
            extraction_class="e",
            extraction_text="hello",
            confidence_score=0.85,
            char_interval=data.CharInterval(start_pos=0, end_pos=5),
        )
        result = annotation._merge_non_overlapping_extractions(
            [[ext_pass1], [ext_pass2]], total_passes=2
        )
        # First extraction wins. cross_pass = 2/2 = 1.0,
        # alignment_conf = 0.9 → final = 1.0 * 0.9 = 0.9
        self.assertLen(result, 1)
        self.assertAlmostEqual(result[0].confidence_score, 0.9, places=4)

    def test_multi_pass_partial_appearance(self):
        """Extraction appearing in 1 of 3 passes gets lower score."""
        ext = data.Extraction(
            extraction_class="e",
            extraction_text="world",
            confidence_score=1.0,
            char_interval=data.CharInterval(start_pos=6, end_pos=11),
        )
        result = annotation._merge_non_overlapping_extractions(
            [[ext], [], []], total_passes=3
        )
        # cross_pass = 1/3, alignment = 1.0 → final ≈ 0.3333
        self.assertLen(result, 1)
        self.assertAlmostEqual(result[0].confidence_score, round(1 / 3, 4), places=4)

    def test_multi_pass_no_alignment_confidence(self):
        """If alignment confidence is None, falls back to cross-pass."""
        ext = data.Extraction(
            extraction_class="e",
            extraction_text="test",
            confidence_score=None,
            char_interval=data.CharInterval(start_pos=0, end_pos=4),
        )
        result = annotation._merge_non_overlapping_extractions(
            [[ext], []], total_passes=2
        )
        # cross_pass = 1/2 = 0.5, alignment = None → final = 0.5
        self.assertAlmostEqual(result[0].confidence_score, 0.5, places=4)


if __name__ == "__main__":
    absltest.main()
