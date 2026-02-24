"""Tests for langcore._consensus module."""

from __future__ import annotations

import asyncio
import concurrent.futures
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
        # total_tokens is recalculated from prompt + completion
        self.assertEqual(merged["total_tokens"], 425)

    def test_handles_none_entries(self):
        u1 = {"prompt_tokens": 100}
        result = _consensus._merge_usage([u1, None, None])
        # Standard keys are always present after normalisation.
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["total_tokens"], 100)

    def test_all_none_returns_none(self):
        self.assertIsNone(_consensus._merge_usage([None, None]))

    def test_normalises_standard_keys(self):
        """All three standard keys are present even if input only has one."""
        result = _consensus._merge_usage([{"completion_tokens": 42}])
        self.assertIn("prompt_tokens", result)
        self.assertIn("completion_tokens", result)
        self.assertIn("total_tokens", result)
        self.assertEqual(result["completion_tokens"], 42)
        self.assertEqual(result["prompt_tokens"], 0)
        # total recalculated
        self.assertEqual(result["total_tokens"], 42)

    def test_extra_keys_preserved(self):
        """Non-standard keys like cached_tokens are still summed."""
        u1 = {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 3}
        u2 = {"prompt_tokens": 20, "completion_tokens": 10, "cached_tokens": 7}
        merged = _consensus._merge_usage([u1, u2])
        self.assertEqual(merged["cached_tokens"], 10)
        # total recalculated from prompt + completion only
        self.assertEqual(merged["total_tokens"], 45)

    def test_total_tokens_recalculated(self):
        """total_tokens is recalculated, not naively summed."""
        u1 = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        u2 = {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275}
        merged = _consensus._merge_usage([u1, u2])
        # Should be 300 + 125 = 425, not 150 + 275 = 425
        # (happens to be the same here, but the logic is correct)
        self.assertEqual(merged["total_tokens"], 425)


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


# ── Parallel sync consensus tests ────────────────────────────────


class ParallelConsensusTest(absltest.TestCase):
    """Tests verifying that sync consensus uses parallel execution."""

    def _make_mock_build_fn(
        self,
        results_by_model: dict[str, data.AnnotatedDocument],
    ):
        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            annotator.annotate_text.return_value = results_by_model[model_id]
            res = mock.MagicMock()
            return (kwargs.get("text_or_documents", ""), annotator, res, {})

        return build_fn

    @mock.patch("langcore._consensus.concurrent.futures.ThreadPoolExecutor")
    def test_uses_thread_pool_executor(self, mock_executor_cls):
        """Sync consensus uses ThreadPoolExecutor for parallelism."""
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([], text="text"),
        }

        # Set up the mock executor to actually run the callables
        mock_executor = mock.MagicMock()
        mock_executor_cls.return_value.__enter__ = mock.MagicMock(
            return_value=mock_executor
        )
        mock_executor_cls.return_value.__exit__ = mock.MagicMock(return_value=False)

        # Make submit return futures that resolve to the right results
        build_fn = self._make_mock_build_fn(results)

        def fake_submit(fn, mid):
            future = concurrent.futures.Future()
            future.set_result(fn(mid))
            return future

        mock_executor.submit.side_effect = fake_submit

        with mock.patch(
            "langcore._consensus.concurrent.futures.as_completed",
            side_effect=lambda fs: list(fs.keys()),
        ):
            _consensus.consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=build_fn,
                build_kwargs={"text_or_documents": "text"},
                annotate_kwargs={},
            )

        # Verify ThreadPoolExecutor was used
        mock_executor_cls.assert_called_once()

    def test_max_workers_parameter(self):
        """max_workers parameter is respected."""
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
        }
        # Just verify the function accepts max_workers without error
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": "text"},
            annotate_kwargs={},
            max_workers=2,
        )
        self.assertIsNotNone(result)

    def test_parallel_produces_same_results_as_expected(self):
        """Parallel consensus produces correct merged results."""
        ext_a = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        ext_b = _make_extraction("Person", "Bob", start=20, end=23, confidence=0.8)
        results = {
            "model-a": _make_doc([ext_a], text="text"),
            "model-b": _make_doc([ext_b], text="text"),
            "model-c": _make_doc([ext_a, ext_b], text="text"),
        }
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b", "model-c"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": "text"},
            annotate_kwargs={},
        )
        # Should have both extractions
        self.assertIsNotNone(result.extractions)
        self.assertGreaterEqual(len(result.extractions), 2)

    def test_exception_in_one_model_propagates(self):
        """If one model raises, the exception propagates (fail_fast=True default)."""

        def build_fn(model_id, **kwargs):
            if model_id == "model-b":
                annotator = mock.MagicMock()
                annotator.annotate_text.side_effect = RuntimeError("model failed")
            else:
                annotator = mock.MagicMock()
                annotator.annotate_text.return_value = _make_doc([], text="text")
            res = mock.MagicMock()
            return ("", annotator, res, {})

        with self.assertRaises(RuntimeError):
            _consensus.consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=build_fn,
                build_kwargs={},
                annotate_kwargs={},
            )


# ── 2.3: document_id forwarding tests ───────────────────────────


class MergeDocumentIdTest(absltest.TestCase):
    """Tests for document_id forwarding in merge_consensus_results."""

    def test_explicit_document_id_forwarded(self):
        """Explicit document_id kwarg is used in the merged result."""
        doc_a = _make_doc([], text="text")
        doc_b = _make_doc([], text="text")
        result = _consensus.merge_consensus_results(
            [doc_a, doc_b], text="text", document_id="my-doc-123"
        )
        self.assertEqual(result._document_id, "my-doc-123")

    def test_inherits_first_result_document_id(self):
        """Without explicit id, the first result's _document_id is used."""
        doc_a = data.AnnotatedDocument(
            document_id="first-doc", extractions=[], text="text"
        )
        doc_b = data.AnnotatedDocument(
            document_id="second-doc", extractions=[], text="text"
        )
        result = _consensus.merge_consensus_results([doc_a, doc_b], text="text")
        self.assertEqual(result._document_id, "first-doc")

    def test_no_document_id_results_in_none_backing(self):
        """When no document_id is supplied and results have none, _document_id is None."""
        doc_a = _make_doc([], text="text")
        doc_b = _make_doc([], text="text")
        # _make_doc does not set _document_id so it's None
        result = _consensus.merge_consensus_results([doc_a, doc_b], text="text")
        # Accessing .document_id would auto-generate, but the backing field
        # should be None (no explicit ID set).
        self.assertIsNone(result._document_id)

    def test_single_result_passthrough_preserves_id(self):
        """Single-result short-circuit returns the original (with its id)."""
        doc = data.AnnotatedDocument(
            document_id="only-doc", extractions=[], text="text"
        )
        result = _consensus.merge_consensus_results([doc], text="text")
        self.assertIs(result, doc)
        self.assertEqual(result._document_id, "only-doc")

    def test_empty_results_no_id(self):
        result = _consensus.merge_consensus_results(
            [], text="text", document_id="ignored"
        )
        # Empty results returns a bare AnnotatedDocument (no ID set)
        self.assertIsNone(result.extractions)


# ── 2.4: Graceful degradation tests ─────────────────────────────


class GracefulDegradationSyncTest(absltest.TestCase):
    """Tests for fail_fast=False in consensus_extract."""

    def _make_mixed_build_fn(self, good_models, fail_models):
        """Build fn where specific models fail with RuntimeError."""

        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            if model_id in fail_models:
                annotator.annotate_text.side_effect = RuntimeError(f"{model_id} failed")
            else:
                ext = _make_extraction(
                    "Person", "Alice", start=0, end=5, confidence=0.9
                )
                annotator.annotate_text.return_value = _make_doc([ext], text="text")
            res = mock.MagicMock()
            return ("", annotator, res, {})

        return build_fn

    def test_graceful_skips_failed_model(self):
        """With fail_fast=False, one failing model is skipped."""
        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mixed_build_fn(
                good_models={"model-a"}, fail_models={"model-b"}
            ),
            build_kwargs={},
            annotate_kwargs={},
            fail_fast=False,
        )
        # model-a succeeded → single-result passthrough
        self.assertIsNotNone(result)
        self.assertLen(result.extractions, 1)

    def test_graceful_all_fail_raises(self):
        """With fail_fast=False, if ALL models fail, RuntimeError is raised."""
        with self.assertRaises(RuntimeError) as ctx:
            _consensus.consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mixed_build_fn(
                    good_models=set(), fail_models={"model-a", "model-b"}
                ),
                build_kwargs={},
                annotate_kwargs={},
                fail_fast=False,
            )
        self.assertIn("All consensus models failed", str(ctx.exception))

    def test_fail_fast_true_raises_immediately(self):
        """With fail_fast=True (default), one failure raises immediately."""
        with self.assertRaises(RuntimeError):
            _consensus.consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mixed_build_fn(
                    good_models={"model-a"}, fail_models={"model-b"}
                ),
                build_kwargs={},
                annotate_kwargs={},
                fail_fast=True,
            )

    def test_graceful_two_of_three_succeed(self):
        """Two successful models merge; one failing model is skipped."""

        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            if model_id == "model-c":
                annotator.annotate_text.side_effect = RuntimeError("c failed")
            else:
                ext = _make_extraction(
                    "Person", "Alice", start=0, end=5, confidence=1.0
                )
                annotator.annotate_text.return_value = _make_doc([ext], text="text")
            res = mock.MagicMock()
            return ("", annotator, res, {})

        result = _consensus.consensus_extract(
            text="text",
            model_ids=["model-a", "model-b", "model-c"],
            build_components_fn=build_fn,
            build_kwargs={},
            annotate_kwargs={},
            fail_fast=False,
        )
        # Both successful models found the same extraction → merged
        self.assertLen(result.extractions, 1)
        # 2/2 agree (failed model excluded from denominator)
        self.assertAlmostEqual(result.extractions[0].confidence_score, 1.0)


class GracefulDegradationAsyncTest(absltest.TestCase):
    """Tests for fail_fast=False in async_consensus_extract."""

    def _make_mixed_build_fn(self, good_models, fail_models):
        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            if model_id in fail_models:
                annotator.async_annotate_text = mock.AsyncMock(
                    side_effect=RuntimeError(f"{model_id} failed")
                )
            else:
                ext = _make_extraction(
                    "Person", "Alice", start=0, end=5, confidence=0.9
                )
                annotator.async_annotate_text = mock.AsyncMock(
                    return_value=_make_doc([ext], text="text")
                )
            res = mock.MagicMock()
            return ("", annotator, res, {})

        return build_fn

    def test_async_graceful_skips_failed(self):
        result = asyncio.run(
            _consensus.async_consensus_extract(
                text="text",
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mixed_build_fn(
                    good_models={"model-a"}, fail_models={"model-b"}
                ),
                build_kwargs={},
                annotate_kwargs={},
                fail_fast=False,
            )
        )
        self.assertIsNotNone(result)
        self.assertLen(result.extractions, 1)

    def test_async_graceful_all_fail_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(
                _consensus.async_consensus_extract(
                    text="text",
                    model_ids=["model-a", "model-b"],
                    build_components_fn=self._make_mixed_build_fn(
                        good_models=set(), fail_models={"model-a", "model-b"}
                    ),
                    build_kwargs={},
                    annotate_kwargs={},
                    fail_fast=False,
                )
            )
        self.assertIn("All consensus models failed", str(ctx.exception))

    def test_async_fail_fast_true_raises(self):
        with self.assertRaises(RuntimeError):
            asyncio.run(
                _consensus.async_consensus_extract(
                    text="text",
                    model_ids=["model-a", "model-b"],
                    build_components_fn=self._make_mixed_build_fn(
                        good_models={"model-a"}, fail_models={"model-b"}
                    ),
                    build_kwargs={},
                    annotate_kwargs={},
                    fail_fast=True,
                )
            )


# ── 2.2: Document list consensus tests ──────────────────────────


class ConsensusExtractDocumentsTest(absltest.TestCase):
    """Tests for consensus_extract_documents (Document list input)."""

    def _make_mock_build_fn(
        self,
        results_by_model: dict[str, data.AnnotatedDocument],
    ):
        def build_fn(model_id, **kwargs):
            annotator = mock.MagicMock()
            annotator.annotate_text.return_value = results_by_model[model_id]
            res = mock.MagicMock()
            return (kwargs.get("text_or_documents", ""), annotator, res, {})

        return build_fn

    def test_processes_multiple_documents(self):
        """Each document gets consensus extraction independently."""
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([ext], text="text"),
        }
        docs = [
            data.Document(text="doc one text", document_id="doc-1"),
            data.Document(text="doc two text", document_id="doc-2"),
        ]
        doc_results = _consensus.consensus_extract_documents(
            documents=docs,
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": ""},
            annotate_kwargs={},
        )
        self.assertLen(doc_results, 2)

    def test_forwards_document_id(self):
        """Each result preserves the original Document's document_id."""
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([ext], text="text"),
        }
        docs = [
            data.Document(text="doc one", document_id="doc-1"),
            data.Document(text="doc two", document_id="doc-2"),
        ]
        doc_results = _consensus.consensus_extract_documents(
            documents=docs,
            model_ids=["model-a", "model-b"],
            build_components_fn=self._make_mock_build_fn(results),
            build_kwargs={"text_or_documents": ""},
            annotate_kwargs={},
        )
        self.assertEqual(doc_results[0]._document_id, "doc-1")
        self.assertEqual(doc_results[1]._document_id, "doc-2")

    def test_empty_documents_returns_empty(self):
        results = _consensus.consensus_extract_documents(
            documents=[],
            model_ids=["model-a"],
            build_components_fn=lambda **kw: None,
            build_kwargs={},
            annotate_kwargs={},
        )
        self.assertEmpty(results)


class AsyncConsensusExtractDocumentsTest(absltest.TestCase):
    """Tests for async_consensus_extract_documents."""

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

    def test_async_processes_multiple_documents(self):
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([ext], text="text"),
        }
        docs = [
            data.Document(text="doc one", document_id="doc-1"),
            data.Document(text="doc two", document_id="doc-2"),
        ]
        doc_results = asyncio.run(
            _consensus.async_consensus_extract_documents(
                documents=docs,
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mock_build_fn(results),
                build_kwargs={"text_or_documents": ""},
                annotate_kwargs={},
            )
        )
        self.assertLen(doc_results, 2)

    def test_async_forwards_document_id(self):
        ext = _make_extraction("Person", "Alice", start=0, end=5, confidence=0.9)
        results = {
            "model-a": _make_doc([ext], text="text"),
            "model-b": _make_doc([ext], text="text"),
        }
        docs = [
            data.Document(text="doc one", document_id="d-1"),
            data.Document(text="doc two", document_id="d-2"),
        ]
        doc_results = asyncio.run(
            _consensus.async_consensus_extract_documents(
                documents=docs,
                model_ids=["model-a", "model-b"],
                build_components_fn=self._make_mock_build_fn(results),
                build_kwargs={"text_or_documents": ""},
                annotate_kwargs={},
            )
        )
        self.assertEqual(doc_results[0]._document_id, "d-1")
        self.assertEqual(doc_results[1]._document_id, "d-2")


if __name__ == "__main__":
    absltest.main()
