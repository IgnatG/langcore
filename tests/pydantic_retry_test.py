"""Tests for langcore._pydantic_validation module and retry wiring."""

from __future__ import annotations

import asyncio
from unittest import mock

import pydantic
from absl.testing import absltest

from langcore import _pydantic_validation as pv
from langcore import extraction as extraction_mod
from langcore import hooks as hooks_lib
from langcore.core import data

# ── Test models ──────────────────────────────────────────────────


class _Person(pydantic.BaseModel):
    """Simple model with a text primary field and an int attribute."""

    name: str = pydantic.Field(description="Full name")
    age: int = pydantic.Field(description="Age in years")


class _Invoice(pydantic.BaseModel):
    """Model with string-only fields."""

    invoice_number: str = pydantic.Field(description="Invoice number")
    amount: float = pydantic.Field(description="Total amount")


# ── Helper factories ──────────────────────────────────────────────


def _make_extraction(
    cls: str,
    text: str,
    attributes: dict | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
) -> data.Extraction:
    interval = None
    if start is not None and end is not None:
        interval = data.CharInterval(start_pos=start, end_pos=end)
    return data.Extraction(
        extraction_class=cls,
        extraction_text=text,
        attributes=attributes,
        char_interval=interval,
    )


def _make_doc(
    extractions: list[data.Extraction] | None = None,
    text: str = "sample text",
) -> data.AnnotatedDocument:
    return data.AnnotatedDocument(extractions=extractions, text=text)


# ── _extraction_to_field_data tests ──────────────────────────────


class ExtractionToFieldDataTest(absltest.TestCase):
    """Tests for the _extraction_to_field_data helper."""

    def test_maps_primary_field(self):
        ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        result = pv._extraction_to_field_data(ext, _Person, "name")
        self.assertEqual(result["name"], "Alice")

    def test_maps_attributes(self):
        ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        result = pv._extraction_to_field_data(ext, _Person, "name")
        self.assertEqual(result["age"], "30")

    def test_skips_extra_attributes(self):
        ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30", "role": "admin"}
        )
        result = pv._extraction_to_field_data(ext, _Person, "name")
        self.assertNotIn("role", result)

    def test_no_attributes(self):
        ext = _make_extraction("_Person", "Alice")
        result = pv._extraction_to_field_data(ext, _Person, "name")
        self.assertEqual(result, {"name": "Alice"})


# ── validate_extractions tests ───────────────────────────────────


class ValidateExtractionsTest(absltest.TestCase):
    """Tests for validate_extractions."""

    def test_all_valid(self):
        exts = [
            _make_extraction("_Person", "Alice", attributes={"age": "30"}),
            _make_extraction("_Person", "Bob", attributes={"age": "25"}),
        ]
        doc = _make_doc(exts)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertLen(valid, 2)
        self.assertEmpty(invalid)

    def test_invalid_detected(self):
        exts = [
            _make_extraction("_Person", "Alice", attributes={"age": "not_a_number"}),
        ]
        doc = _make_doc(exts)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertEmpty(valid)
        self.assertLen(invalid, 1)
        self.assertIn("Alice", invalid[0][1])

    def test_mixed_valid_and_invalid(self):
        exts = [
            _make_extraction("_Person", "Alice", attributes={"age": "30"}),
            _make_extraction("_Person", "Bob", attributes={"age": "nope"}),
        ]
        doc = _make_doc(exts)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertLen(valid, 1)
        self.assertLen(invalid, 1)
        self.assertEqual(valid[0].extraction_text, "Alice")

    def test_empty_extractions(self):
        doc = _make_doc(extractions=[])
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertEmpty(valid)
        self.assertEmpty(invalid)

    def test_none_extractions(self):
        doc = _make_doc(extractions=None)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertEmpty(valid)
        self.assertEmpty(invalid)

    def test_different_class_passes_through(self):
        """Extractions that don't match the schema class are treated as valid."""
        exts = [
            _make_extraction("OtherClass", "something", attributes={}),
        ]
        doc = _make_doc(exts)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertLen(valid, 1)
        self.assertEmpty(invalid)

    def test_case_insensitive_class_match(self):
        """Class name matching is case-insensitive."""
        exts = [
            _make_extraction("_person", "Alice", attributes={"age": "30"}),
        ]
        doc = _make_doc(exts)
        valid, invalid = pv.validate_extractions(doc, _Person)
        self.assertLen(valid, 1)
        self.assertEmpty(invalid)


# ── build_correction_context tests ───────────────────────────────


class BuildCorrectionContextTest(absltest.TestCase):
    """Tests for build_correction_context."""

    def test_includes_error_messages(self):
        ext = _make_extraction("_Person", "Alice", attributes={"age": "bad"})
        invalid = [(ext, "Age must be an integer")]
        result = pv.build_correction_context(invalid)
        self.assertIn("Age must be an integer", result)
        self.assertIn("VALIDATION FEEDBACK", result)

    def test_numbers_multiple_errors(self):
        invalid = [
            (_make_extraction("_Person", "A"), "Error 1"),
            (_make_extraction("_Person", "B"), "Error 2"),
        ]
        result = pv.build_correction_context(invalid)
        self.assertIn("1. Error 1", result)
        self.assertIn("2. Error 2", result)

    def test_empty_list_returns_template(self):
        result = pv.build_correction_context([])
        # Should still include the template wrapper
        self.assertIn("VALIDATION FEEDBACK", result)


# ── pydantic_retry tests (sync) ─────────────────────────────────


class PydanticRetryTest(absltest.TestCase):
    """Tests for the sync pydantic_retry function."""

    def _make_retry_kwargs(
        self, annotator: mock.MagicMock, resolver: mock.MagicMock
    ) -> dict:
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 1000,
            "batch_length": 10,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_returns_unchanged_when_all_valid(self):
        """No retry when all extractions are valid."""
        exts = [
            _make_extraction("_Person", "Alice", attributes={"age": "30"}),
        ]
        doc = _make_doc(exts)
        annotator = mock.MagicMock()
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        self.assertEqual(result.extractions, exts)
        annotator.annotate_text.assert_not_called()

    def test_retries_on_invalid_extractions(self):
        """Retry is triggered when extractions fail validation."""
        bad_ext = _make_extraction("_Person", "Alice", attributes={"age": "not_int"})
        doc = _make_doc([bad_ext])

        # Retry returns a corrected extraction
        corrected_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        retry_doc = _make_doc([corrected_ext])

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        annotator.annotate_text.assert_called_once()
        self.assertLen(result.extractions, 1)
        self.assertEqual(result.extractions[0].extraction_text, "Alice")
        self.assertEqual(result.extractions[0].attributes["age"], "30")

    def test_preserves_valid_from_first_pass(self):
        """Valid extractions from the first pass are kept alongside retry results."""
        good_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        bad_ext = _make_extraction("_Person", "Bob", attributes={"age": "nope"})
        doc = _make_doc([good_ext, bad_ext])

        corrected_ext = _make_extraction("_Person", "Bob", attributes={"age": "25"})
        retry_doc = _make_doc([corrected_ext])

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        self.assertLen(result.extractions, 2)
        texts = {e.extraction_text for e in result.extractions}
        self.assertSetEqual(texts, {"Alice", "Bob"})

    def test_returns_early_for_none_text(self):
        """If result.text is None, return immediately."""
        doc = data.AnnotatedDocument(extractions=None, text=None)
        annotator = mock.MagicMock()
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        self.assertIs(result, doc)
        annotator.annotate_text.assert_not_called()

    def test_multiple_retries(self):
        """Multiple retry attempts are supported."""
        bad_ext = _make_extraction("_Person", "Alice", attributes={"age": "bad"})
        doc = _make_doc([bad_ext])

        # First retry still produces invalid
        still_bad = _make_extraction(
            "_Person", "Alice", attributes={"age": "still_bad"}
        )
        # Second retry produces valid
        corrected = _make_extraction("_Person", "Alice", attributes={"age": "30"})

        annotator = mock.MagicMock()
        annotator.annotate_text.side_effect = [
            _make_doc([still_bad]),
            _make_doc([corrected]),
        ]
        resolver = mock.MagicMock()

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["max_retries"] = 2
        result = pv.pydantic_retry(doc, _Person, **kwargs)
        self.assertEqual(annotator.annotate_text.call_count, 2)
        self.assertLen(result.extractions, 1)
        self.assertEqual(result.extractions[0].attributes["age"], "30")

    def test_correction_context_passed_to_annotator(self):
        """The retry call includes validation feedback in additional_context."""
        bad_ext = _make_extraction("_Person", "Alice", attributes={"age": "xyz"})
        doc = _make_doc([bad_ext])

        corrected_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = _make_doc([corrected_ext])
        resolver = mock.MagicMock()

        pv.pydantic_retry(doc, _Person, **self._make_retry_kwargs(annotator, resolver))

        call_kwargs = annotator.annotate_text.call_args[1]
        self.assertIn("VALIDATION FEEDBACK", call_kwargs["additional_context"])

    def test_appends_to_existing_additional_context(self):
        """Correction context is appended to original additional_context."""
        bad_ext = _make_extraction("_Person", "Alice", attributes={"age": "xyz"})
        doc = _make_doc([bad_ext])

        corrected_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = _make_doc([corrected_ext])
        resolver = mock.MagicMock()

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["additional_context"] = "Original context here"
        pv.pydantic_retry(doc, _Person, **kwargs)

        call_kwargs = annotator.annotate_text.call_args[1]
        ctx = call_kwargs["additional_context"]
        self.assertIn("Original context here", ctx)
        self.assertIn("VALIDATION FEEDBACK", ctx)


# ── async_pydantic_retry tests ───────────────────────────────────


class AsyncPydanticRetryTest(absltest.TestCase):
    """Tests for the async async_pydantic_retry function."""

    def _make_retry_kwargs(
        self, annotator: mock.MagicMock, resolver: mock.MagicMock
    ) -> dict:
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 1000,
            "batch_length": 10,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_returns_unchanged_when_all_valid(self):
        exts = [
            _make_extraction("_Person", "Alice", attributes={"age": "30"}),
        ]
        doc = _make_doc(exts)
        annotator = mock.MagicMock()
        resolver = mock.MagicMock()

        result = asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        self.assertEqual(result.extractions, exts)

    def test_retries_on_invalid_extractions(self):
        bad_ext = _make_extraction("_Person", "Alice", attributes={"age": "not_int"})
        doc = _make_doc([bad_ext])

        corrected_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        retry_doc = _make_doc([corrected_ext])

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        result = asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        annotator.async_annotate_text.assert_called_once()
        self.assertLen(result.extractions, 1)
        self.assertEqual(result.extractions[0].attributes["age"], "30")

    def test_preserves_valid_from_first_pass(self):
        good_ext = _make_extraction("_Person", "Alice", attributes={"age": "30"})
        bad_ext = _make_extraction("_Person", "Bob", attributes={"age": "nope"})
        doc = _make_doc([good_ext, bad_ext])

        corrected_ext = _make_extraction("_Person", "Bob", attributes={"age": "25"})
        retry_doc = _make_doc([corrected_ext])

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        result = asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        self.assertLen(result.extractions, 2)


# ── _resolve_retry_count tests ───────────────────────────────────
class ResolveRetryCountTest(absltest.TestCase):
    """Tests for the _resolve_retry_count helper."""

    def test_unset_with_schema_returns_one(self):
        """When not explicitly set and schema provided, default to 1."""
        result = extraction_mod._resolve_retry_count(extraction_mod._UNSET, _Person)
        self.assertEqual(result, 1)

    def test_unset_without_schema_returns_zero(self):
        """When not explicitly set and no schema, default to 0."""
        result = extraction_mod._resolve_retry_count(extraction_mod._UNSET, None)
        self.assertEqual(result, 0)

    def test_explicit_zero_with_schema_returns_zero(self):
        """Explicit 0 disables retries even with schema."""
        result = extraction_mod._resolve_retry_count(0, _Person)
        self.assertEqual(result, 0)

    def test_explicit_value_passed_through(self):
        """Explicit positive int is returned as-is."""
        result = extraction_mod._resolve_retry_count(3, _Person)
        self.assertEqual(result, 3)

    def test_explicit_zero_without_schema(self):
        result = extraction_mod._resolve_retry_count(0, None)
        self.assertEqual(result, 0)

    def test_explicit_value_without_schema(self):
        """Explicit value is honoured even without schema."""
        result = extraction_mod._resolve_retry_count(2, None)
        self.assertEqual(result, 2)


# ── _build_retry_regions tests ───────────────────────────────────


class BuildRetryRegionsTest(absltest.TestCase):
    """Tests for _build_retry_regions helper."""

    def test_single_extraction_produces_centred_region(self):
        ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=500, end=510
        )
        regions = pv._build_retry_regions(
            [(ext, "err")], full_text="x" * 2000, max_char_buffer=600
        )
        self.assertLen(regions, 1)
        start, end = regions[0]
        # Region centred on midpoint 505, half=max(300, 200)=300
        self.assertEqual(start, 205)
        self.assertEqual(end, 805)

    def test_overlapping_regions_are_merged(self):
        ext_a = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=100, end=110
        )
        ext_b = _make_extraction(
            "_Person", "Bob", attributes={"age": "bad"}, start=150, end=160
        )
        regions = pv._build_retry_regions(
            [(ext_a, "err"), (ext_b, "err")],
            full_text="x" * 1000,
            max_char_buffer=200,
        )
        # Regions around 105 and 155 with half=100 overlap, should merge
        self.assertLen(regions, 1)

    def test_distant_regions_stay_separate(self):
        ext_a = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        ext_b = _make_extraction(
            "_Person", "Bob", attributes={"age": "bad"}, start=900, end=910
        )
        regions = pv._build_retry_regions(
            [(ext_a, "err"), (ext_b, "err")],
            full_text="x" * 1000,
            max_char_buffer=200,
        )
        self.assertLen(regions, 2)

    def test_no_char_interval_falls_back_to_full_text(self):
        ext = _make_extraction("_Person", "Alice", attributes={"age": "bad"})
        regions = pv._build_retry_regions(
            [(ext, "err")], full_text="x" * 500, max_char_buffer=200
        )
        self.assertLen(regions, 1)
        self.assertEqual(regions[0], (0, 500))

    def test_region_clamped_to_text_boundaries(self):
        ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        regions = pv._build_retry_regions(
            [(ext, "err")], full_text="x" * 50, max_char_buffer=200
        )
        self.assertLen(regions, 1)
        self.assertEqual(regions[0][0], 0)
        self.assertLessEqual(regions[0][1], 50)

    def test_empty_text_returns_no_regions(self):
        ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=5
        )
        regions = pv._build_retry_regions(
            [(ext, "err")], full_text="", max_char_buffer=200
        )
        self.assertEmpty(regions)


# ── _merge_usage_pair tests ──────────────────────────────────────


class MergeUsagePairTest(absltest.TestCase):
    """Tests for _merge_usage_pair helper."""

    def test_sums_both_dicts(self):
        result = pv._merge_usage_pair(
            {"prompt_tokens": 100, "completion_tokens": 50},
            {"prompt_tokens": 200, "completion_tokens": 75},
        )
        self.assertEqual(result["prompt_tokens"], 300)
        self.assertEqual(result["completion_tokens"], 125)

    def test_none_plus_dict(self):
        result = pv._merge_usage_pair(None, {"prompt_tokens": 100})
        self.assertEqual(result, {"prompt_tokens": 100})

    def test_dict_plus_none(self):
        result = pv._merge_usage_pair({"prompt_tokens": 100}, None)
        self.assertEqual(result, {"prompt_tokens": 100})

    def test_both_none(self):
        self.assertIsNone(pv._merge_usage_pair(None, None))


# ── _offset_extractions tests ────────────────────────────────────


class OffsetExtractionsTest(absltest.TestCase):
    """Tests for _offset_extractions helper."""

    def test_shifts_char_interval(self):
        ext = _make_extraction("_Person", "Alice", start=10, end=20)
        pv._offset_extractions([ext], 500)
        self.assertEqual(ext.char_interval.start_pos, 510)
        self.assertEqual(ext.char_interval.end_pos, 520)

    def test_no_interval_is_safe(self):
        ext = _make_extraction("_Person", "Alice")
        # Should not raise
        pv._offset_extractions([ext], 500)
        self.assertIsNone(ext.char_interval)

    def test_zero_offset(self):
        ext = _make_extraction("_Person", "Alice", start=10, end=20)
        pv._offset_extractions([ext], 0)
        self.assertEqual(ext.char_interval.start_pos, 10)
        self.assertEqual(ext.char_interval.end_pos, 20)


# ── Chunk-level retry integration tests ──────────────────────────


class ChunkLevelRetryTest(absltest.TestCase):
    """Tests verifying that pydantic_retry uses chunk-level extraction."""

    def _make_retry_kwargs(
        self, annotator: mock.MagicMock, resolver: mock.MagicMock
    ) -> dict:
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 200,
            "batch_length": 10,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_retry_sends_only_failing_region(self):
        """Retry re-extracts only the region around the failing extraction."""
        full_text = "A" * 100 + "Alice is 30" + "B" * 889  # 1000 chars total
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=100, end=111
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text=full_text)

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=11
        )
        retry_doc = data.AnnotatedDocument(
            extractions=[corrected_ext], text="region text"
        )

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        pv.pydantic_retry(doc, _Person, **self._make_retry_kwargs(annotator, resolver))

        # The annotator was called with a text shorter than the full document
        call_kwargs = annotator.annotate_text.call_args
        region_text = call_kwargs[1]["text"]
        self.assertLess(len(region_text), len(full_text))

    def test_retry_preserves_document_id(self):
        """Original document_id is preserved across retries."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 100)
        doc.document_id = "original-id-123"

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(
            extractions=[corrected_ext], text="region text"
        )
        retry_doc.document_id = "retry-id-456"

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        self.assertEqual(result.document_id, "original-id-123")

    def test_retry_accumulates_usage(self):
        """Token usage from retries is accumulated, not discarded."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(
            extractions=[bad_ext],
            text="x" * 100,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(
            extractions=[corrected_ext],
            text="region text",
            usage={"prompt_tokens": 30, "completion_tokens": 10},
        )

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        self.assertEqual(result.usage["prompt_tokens"], 130)
        self.assertEqual(result.usage["completion_tokens"], 60)

    def test_retry_offsets_extractions_to_document_coords(self):
        """Retry extractions are shifted back to full-document coordinates."""
        full_text = "A" * 500 + "Alice is 30" + "B" * 489  # 1000 chars
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=500, end=511
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text=full_text)

        # Retry returns extraction at region-relative coords
        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=11
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )

        # The corrected extraction should have been offset to document coords
        found = [
            e
            for e in result.extractions
            if e.extraction_text == "Alice" and e.attributes.get("age") == "30"
        ]
        self.assertLen(found, 1)
        self.assertIsNotNone(found[0].char_interval)
        # Offset should be > 0 (shifted from region-relative to absolute)
        self.assertGreater(found[0].char_interval.start_pos, 0)

    def test_multiple_failing_regions_each_retried(self):
        """Multiple distant failing extractions each get their own retry call."""
        full_text = "x" * 2000
        ext_a = _make_extraction(
            "_Person", "A", attributes={"age": "bad"}, start=0, end=10
        )
        ext_b = _make_extraction(
            "_Person", "B", attributes={"age": "bad"}, start=1500, end=1510
        )
        doc = data.AnnotatedDocument(extractions=[ext_a, ext_b], text=full_text)

        corrected_a = _make_extraction(
            "_Person", "A", attributes={"age": "30"}, start=0, end=10
        )
        corrected_b = _make_extraction(
            "_Person", "B", attributes={"age": "25"}, start=0, end=10
        )
        retry_doc_a = data.AnnotatedDocument(extractions=[corrected_a], text="region_a")
        retry_doc_b = data.AnnotatedDocument(extractions=[corrected_b], text="region_b")

        annotator = mock.MagicMock()
        annotator.annotate_text.side_effect = [retry_doc_a, retry_doc_b]
        resolver = mock.MagicMock()

        result = pv.pydantic_retry(
            doc, _Person, **self._make_retry_kwargs(annotator, resolver)
        )
        # Two separate retry calls (one per region)
        self.assertEqual(annotator.annotate_text.call_count, 2)
        self.assertLen(result.extractions, 2)


class AsyncChunkLevelRetryTest(absltest.TestCase):
    """Tests verifying that async_pydantic_retry uses chunk-level extraction."""

    def _make_retry_kwargs(
        self, annotator: mock.MagicMock, resolver: mock.MagicMock
    ) -> dict:
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 200,
            "batch_length": 10,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_async_retry_sends_only_failing_region(self):
        full_text = "A" * 100 + "Alice is 30" + "B" * 889
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=100, end=111
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text=full_text)

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=11
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        call_kwargs = annotator.async_annotate_text.call_args
        region_text = call_kwargs[1]["text"]
        self.assertLess(len(region_text), len(full_text))

    def test_async_retry_accumulates_usage(self):
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(
            extractions=[bad_ext],
            text="x" * 100,
            usage={"prompt_tokens": 100},
        )

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(
            extractions=[corrected_ext],
            text="region",
            usage={"prompt_tokens": 20},
        )

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        result = asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        self.assertEqual(result.usage["prompt_tokens"], 120)

    def test_async_retry_preserves_document_id(self):
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 100)
        doc.document_id = "original-id"

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")
        retry_doc.document_id = "retry-id"

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        result = asyncio.run(
            pv.async_pydantic_retry(
                doc, _Person, **self._make_retry_kwargs(annotator, resolver)
            )
        )
        self.assertEqual(result.document_id, "original-id")


# ── 5.5: Hook emission tests ────────────────────────────────────


class RetryHooksTest(absltest.TestCase):
    """Tests for VALIDATION_RETRY_START/COMPLETE hook emissions."""

    def _make_retry_kwargs(self, annotator, resolver):
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 1000,
            "batch_length": 500,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_hooks_emitted_on_retry(self):
        """VALIDATION_RETRY_START and VALIDATION_RETRY_COMPLETE fire during retry."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 100)

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        hooks = hooks_lib.Hooks()
        events: list[tuple[str, dict]] = []
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_START,
            lambda payload: events.append(("start", payload)),
        )
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_COMPLETE,
            lambda payload: events.append(("complete", payload)),
        )

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["hooks"] = hooks

        pv.pydantic_retry(doc, _Person, **kwargs)

        self.assertLen(events, 2)
        self.assertEqual(events[0][0], "start")
        self.assertEqual(events[0][1]["attempt"], 1)
        self.assertEqual(events[0][1]["invalid_count"], 1)
        self.assertIn("regions", events[0][1])

        self.assertEqual(events[1][0], "complete")
        self.assertEqual(events[1][1]["attempt"], 1)
        self.assertIn("retry_valid_count", events[1][1])
        self.assertIn("total_extractions", events[1][1])

    def test_no_hooks_when_all_valid(self):
        """No retry hooks fire when all extractions are valid."""
        good_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[good_ext], text="x" * 100)

        annotator = mock.MagicMock()
        resolver = mock.MagicMock()

        hooks = hooks_lib.Hooks()
        events: list = []
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_START,
            lambda p: events.append(p),
        )

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["hooks"] = hooks

        pv.pydantic_retry(doc, _Person, **kwargs)

        self.assertEmpty(events)

    def test_async_hooks_emitted_on_retry(self):
        """Async retry also emits hooks."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 100)

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        hooks = hooks_lib.Hooks()
        events: list[tuple[str, dict]] = []
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_START,
            lambda payload: events.append(("start", payload)),
        )
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_COMPLETE,
            lambda payload: events.append(("complete", payload)),
        )

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["hooks"] = hooks

        asyncio.run(pv.async_pydantic_retry(doc, _Person, **kwargs))

        self.assertLen(events, 2)
        self.assertEqual(events[0][0], "start")
        self.assertEqual(events[1][0], "complete")

    def test_hook_payload_contains_regions(self):
        """The retry_start payload includes the text regions being retried."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=50, end=60
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 500)

        corrected_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected_ext], text="region")

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        hooks = hooks_lib.Hooks()
        payloads: list[dict] = []
        hooks.on(
            hooks_lib.HookName.VALIDATION_RETRY_START,
            lambda p: payloads.append(p),
        )

        kwargs = self._make_retry_kwargs(annotator, resolver)
        kwargs["hooks"] = hooks

        pv.pydantic_retry(doc, _Person, **kwargs)

        self.assertLen(payloads, 1)
        regions = payloads[0]["regions"]
        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)
        # Each region is a (start, end) tuple
        start, end = regions[0]
        self.assertGreaterEqual(start, 0)
        self.assertGreater(end, start)


# ── 5.4: Extraction count cap tests ─────────────────────────────


class ExtractionCapTest(absltest.TestCase):
    """Tests for _MAX_EXTRACTION_GROWTH_FACTOR safety cap."""

    def _make_retry_kwargs(self, annotator, resolver):
        return {
            "annotator": annotator,
            "res": resolver,
            "max_char_buffer": 1000,
            "batch_length": 500,
            "additional_context": None,
            "debug": False,
            "extraction_passes": 1,
            "context_window_chars": None,
            "show_progress": False,
            "max_workers": 1,
            "tokenizer": None,
            "alignment_kwargs": {},
            "hooks": hooks_lib.Hooks(),
            "max_retries": 1,
        }

    def test_cap_trims_excess_extractions(self):
        """When retry produces too many extractions, cap trims them."""
        # Start with 2 extractions, 1 invalid.
        good_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        bad_ext = _make_extraction(
            "_Person", "Bob", attributes={"age": "bad"}, start=50, end=60
        )
        doc = data.AnnotatedDocument(extractions=[good_ext, bad_ext], text="x" * 200)

        # LLM retry returns 5 extractions (way more than the 1 invalid).
        hallucinated = [
            _make_extraction(
                "_Person",
                f"Person{i}",
                attributes={"age": str(20 + i)},
                start=0,
                end=10,
            )
            for i in range(5)
        ]
        retry_doc = data.AnnotatedDocument(extractions=hallucinated, text="region")

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        kwargs = self._make_retry_kwargs(annotator, resolver)
        result = pv.pydantic_retry(doc, _Person, **kwargs)

        # Cap = max(2 * 2, 1) = 4.  Original had 2, retry produced 5 valid +
        # 1 original valid = 6 total.  Should be trimmed to 4.
        self.assertIsNotNone(result.extractions)
        self.assertLessEqual(len(result.extractions), 4)

    def test_no_cap_when_within_limit(self):
        """Normal retry that stays within the cap is not trimmed."""
        bad_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "bad"}, start=0, end=10
        )
        doc = data.AnnotatedDocument(extractions=[bad_ext], text="x" * 100)

        corrected = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        retry_doc = data.AnnotatedDocument(extractions=[corrected], text="region")

        annotator = mock.MagicMock()
        annotator.annotate_text.return_value = retry_doc
        resolver = mock.MagicMock()

        kwargs = self._make_retry_kwargs(annotator, resolver)
        result = pv.pydantic_retry(doc, _Person, **kwargs)

        # Cap = max(1 * 2, 1) = 2.  Merged = 1 (retry valid).  No trimming.
        self.assertIsNotNone(result.extractions)
        self.assertLessEqual(len(result.extractions), 2)

    def test_async_cap_trims_excess(self):
        """Async retry also respects the extraction cap."""
        good_ext = _make_extraction(
            "_Person", "Alice", attributes={"age": "30"}, start=0, end=10
        )
        bad_ext = _make_extraction(
            "_Person", "Bob", attributes={"age": "bad"}, start=50, end=60
        )
        doc = data.AnnotatedDocument(extractions=[good_ext, bad_ext], text="x" * 200)

        hallucinated = [
            _make_extraction(
                "_Person",
                f"Person{i}",
                attributes={"age": str(20 + i)},
                start=0,
                end=10,
            )
            for i in range(5)
        ]
        retry_doc = data.AnnotatedDocument(extractions=hallucinated, text="region")

        annotator = mock.MagicMock()
        annotator.async_annotate_text = mock.AsyncMock(return_value=retry_doc)
        resolver = mock.MagicMock()

        kwargs = self._make_retry_kwargs(annotator, resolver)
        result = asyncio.run(pv.async_pydantic_retry(doc, _Person, **kwargs))

        self.assertIsNotNone(result.extractions)
        self.assertLessEqual(len(result.extractions), 4)


if __name__ == "__main__":
    absltest.main()
