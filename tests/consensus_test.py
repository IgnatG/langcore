"""Tests for langcore._consensus module."""

from __future__ import annotations

import asyncio
from unittest import mock

from absl.testing import absltest

from langcore import _consensus
from langcore.core import data

# ── Helper factories ──────────────────────────────────────────────


def _make_extraction(
    cls: str,
    text: str,
    *,
    start: int | None = None,
    end: int | None = None,
    attributes: dict | None = None,
    confidence: float | None = None,
) -> data.Extraction:
    interval = None
    if start is not None and end is not None:
        interval = data.CharInterval(start_pos=start, end_pos=end)
    return data.Extraction(
        extraction_class=cls,
        extraction_text=text,
        char_interval=interval,
        attributes=attributes,
        confidence_score=confidence,
    )


def _make_doc(
    extractions: list[data.Extraction] | None = None,
    text: str = "sample text",
    usage: dict[str, int] | None = None,
) -> data.AnnotatedDocument:
    return data.AnnotatedDocument(extractions=extractions, text=text, usage=usage)


# ── _tag_extractions tests ───────────────────────────────────────


class TagExtractionsTest(absltest.TestCase):
    """Tests for _tag_extractions."""

    def test_tags_extractions_with_model_id(self):
        ext = _make_extraction("Person", "Alice")
        doc = _make_doc([ext])
        _consensus._tag_extractions(doc, "gpt-4o")
        self.assertEqual(ext.attributes[_consensus.MODEL_ID_KEY], "gpt-4o")

    def test_tags_with_existing_attributes(self):
        ext = _make_extraction("Person", "Alice", attributes={"age": "30"})
        doc = _make_doc([ext])
        _consensus._tag_extractions(doc, "gemini-2.5-flash")
        self.assertEqual(ext.attributes[_consensus.MODEL_ID_KEY], "gemini-2.5-flash")
        self.assertEqual(ext.attributes["age"], "30")

    def test_handles_none_extractions(self):
        doc = _make_doc(extractions=None)
        _consensus._tag_extractions(doc, "model-a")
        # Should not raise


# ── _merge_usage tests ───────────────────────────────────────────


class MergeUsageTest(absltest.TestCase):
    """Tests for _merge_usage."""

    def test_sums_usage(self):
        u1 = {"prompt_tokens": 100, "completion_tokens": 50}
        u2 = {"prompt_tokens": 200, "completion_tokens": 75}
        merged = _consensus._merge_usage([u1, u2])
        self.assertEqual(merged["prompt_tokens"], 300)
        self.assertEqual(merged["completion_tokens"], 125)

    def test_handles_none_entries(self):
        u1 = {"prompt_tokens": 100}
        result = _consensus._merge_usage([u1, None, None])
        self.assertEqual(result, {"prompt_tokens": 100})

    def test_all_none_returns_none(self):
        self.assertIsNone(_consensus._merge_usage([None, None]))


# ── merge_consensus_results tests ────────────────────────────────


class MergeConsensusResultsTest(absltest.TestCase):
    """Tests for merge_consensus_results."""

    def test_empty_results_returns_empty_doc(self):
        result = _consensus.merge_consensus_results([], text="hello")
        self.assertIsNone(result.extractions)
        self.assertEqual(result.text, "hello")

    def test_single_result_passthrough(self):
        ext = _make_extraction("Person", "Alice", start=0, end=5)
        doc = _make_doc([ext], text="Alice is here")
        result = _consensus.merge_consensus_results([doc], text="Alice is here")
        self.assertIs(result, doc)

    def test_merges_non_overlapping_extractions(self):
        """Extractions at different positions from different models are all kept."""
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        ext_b = _make_extraction("Person", "Bob", start=20, end=23, confidence=0.8)
        doc_a = _make_doc([ext_a], text="Alice ... Bob ...")
        doc_b = _make_doc([ext_b], text="Alice ... Bob ...")
        result = _consensus.merge_consensus_results(
            [doc_a, doc_b], text="Alice ... Bob ..."
        )
        self.assertLen(result.extractions, 2)

    def test_overlapping_extractions_merged_with_higher_confidence(self):
        """Extractions at the same position from 2 models get agreement boost."""
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        ext_b = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        doc_a = _make_doc([ext_a])
        doc_b = _make_doc([ext_b])
        result = _consensus.merge_consensus_results([doc_a, doc_b], text="sample")
        # Should produce 1 merged extraction with confidence = 2/2 * 1.0 = 1.0
        self.assertLen(result.extractions, 1)
        self.assertAlmostEqual(result.extractions[0].confidence_score, 1.0)

    def test_partial_agreement_confidence(self):
        """Extraction found by 1 of 3 models gets lower confidence."""
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        doc_a = _make_doc([ext])
        doc_b = _make_doc([], text="Alice text")
        doc_c = _make_doc([], text="Alice text")
        result = _consensus.merge_consensus_results(
            [doc_a, doc_b, doc_c], text="Alice text"
        )
        self.assertLen(result.extractions, 1)
        # confidence = 1/3 * 0.9 = 0.3
        self.assertAlmostEqual(result.extractions[0].confidence_score, 0.3)

    def test_merges_usage(self):
        doc_a = _make_doc([], usage={"prompt_tokens": 100})
        doc_b = _make_doc([], usage={"prompt_tokens": 200})
        result = _consensus.merge_consensus_results([doc_a, doc_b], text="text")
        self.assertEqual(result.usage["prompt_tokens"], 300)


# ── consensus_extract tests (sync) ──────────────────────────────


class ConsensusExtractTest(absltest.TestCase):
    """Tests for consensus_extract using mocked build/annotate."""

    def _make_mock_build_fn(
        self,
        results_by_model: dict[str, data.AnnotatedDocument],
    ):
        """Create a mock build_components_fn that returns model-specific annotators."""

        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            annotator.annotate_text.return_value = results_by_model[model_id]
            res = mock.MagicMock()
            return (kwargs.get("text_or_documents", ""), annotator, res, {})

        return build_fn

    def test_runs_each_model_and_merges(self):
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        ext_b = _make_extraction("Person", "Bob", start=20, end=23, confidence=0.8)
        results = {
            "model-a": _make_doc([ext_a], text="text"),
            "model-b": _make_doc([ext_b], text="text"),
        }
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": "text"},
            annotate_kwargs={},
        )
        self.assertLen(result.extractions, 2)

    def test_tags_model_ids(self):
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([], text="text"),
        }
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": "text"},
            annotate_kwargs={},
        )
        # The extraction from model-a should be tagged
        tagged_models = {
            e.attributes.get(_consensus.MODEL_ID_KEY)
            for e in result.extractions
            if e.attributes
        }
        self.assertIn("model-a", tagged_models)

    def test_consensus_confidence_scoring(self):
        """Overlapping extractions from both models get full agreement."""
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        ext_b = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        results = {
            "model-a": _make_doc([ext_a], text="text"),
            "model-b": _make_doc([ext_b], text="text"),
        }
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": "text"},
            annotate_kwargs={},
        )
        self.assertLen(result.extractions, 1)
        # 2/2 models agree -> agreement_ratio=1.0 x confidence=1.0
        self.assertAlmostEqual(result.extractions[0].confidence_score, 1.0)


# ── async_consensus_extract tests ────────────────────────────────


class AsyncConsensusExtractTest(absltest.TestCase):
    """Tests for async_consensus_extract using mocked build/annotate."""

    def _make_mock_build_fn(
        self,
        results_by_model: dict[str, data.AnnotatedDocument],
    ):
        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            annotator.async_annotate_text = mock.AsyncMock(
                return_value=results_by_model[model_id]
            )
            res = mock.MagicMock()
            return (kwargs.get("text_or_documents", ""), annotator, res, {})

        return build_fn

    def test_runs_each_model_concurrently(self):
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        ext_b = _make_extraction("Person", "Bob", start=20, end=23, confidence=0.8)
        results = {
            "model-a": _make_doc([ext_a], text="text"),
            "model-b": _make_doc([ext_b], text="text"),
        }
        result = asyncio.run(
            _consensus.async_consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mock_build_fn(results),
                build_kwargs={"text_or_documents": "text"},
                annotate_kwargs={},
            )
        )
        self.assertLen(result.extractions, 2)

    def test_async_consensus_confidence(self):
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        ext_b = _make_extraction("Person", "Alice", start=0, end=5, confidence=1.0)
        results = {
            "model-a": _make_doc([ext_a], text="text"),
            "model-b": _make_doc([ext_b], text="text"),
        }
        result = asyncio.run(
            _consensus.async_consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mock_build_fn(results),
                build_kwargs={"text_or_documents": "text"},
                annotate_kwargs={},
            )
        )
        self.assertLen(result.extractions, 1)
        self.assertAlmostEqual(result.extractions[0].confidence_score, 1.0)

    def test_async_tags_model_ids(self):
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([], text="text"),
        }
        result = asyncio.run(
            _consensus.async_consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mock_build_fn(results),
                build_kwargs={"text_or_documents": "text"},
                annotate_kwargs={},
            )
        )
        tagged = {
            e.attributes.get(_consensus.MODEL_ID_KEY)
            for e in result.extractions
            if e.attributes
        }
        self.assertIn("model-a", tagged)


if __name__ == "__main__":
    absltest.main()
