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
) -> data.Extraction:
    return data.Extraction(
        extraction_class=cls,
        extraction_text=text,
        attributes=attributes,
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


if __name__ == "__main__":
    absltest.main()
