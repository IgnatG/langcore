"""Tests for AnnotatedDocument.to_pydantic and schema extraction integration."""

from __future__ import annotations

import warnings

import pydantic
from absl.testing import absltest

from langcore.core import data


class _Invoice(pydantic.BaseModel):
    """Test model for invoice extraction."""

    invoice_number: str = pydantic.Field(description="Invoice ID")
    amount: float = pydantic.Field(description="Total amount")
    due_date: str = pydantic.Field(description="Due date")


class _Person(pydantic.BaseModel):
    """Test model with 'name' as primary field."""

    name: str
    age: int


class ToPydanticTest(absltest.TestCase):
    """Tests for AnnotatedDocument.to_pydantic."""

    def test_converts_matching_extractions(self):
        """Extractions with matching class name are converted."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_Invoice",
                    extraction_text="INV-001",
                    attributes={
                        "amount": "100.0",
                        "due_date": "2024-01-01",
                    },
                )
            ],
            text="Invoice INV-001 for $100 due 2024-01-01",
        )
        results = doc.to_pydantic(_Invoice)
        self.assertLen(results, 1)
        self.assertIsInstance(results[0], _Invoice)
        self.assertEqual(results[0].invoice_number, "INV-001")
        self.assertAlmostEqual(results[0].amount, 100.0)
        self.assertEqual(results[0].due_date, "2024-01-01")

    def test_skips_non_matching_class(self):
        """Extractions with different class name are skipped."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="SomeOtherClass",
                    extraction_text="test",
                )
            ],
            text="test",
        )
        results = doc.to_pydantic(_Invoice)
        self.assertEmpty(results)

    def test_case_insensitive_class_match(self):
        """Class name matching is case-insensitive."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_invoice",
                    extraction_text="INV-002",
                    attributes={
                        "amount": "200.0",
                        "due_date": "2024-06-01",
                    },
                )
            ],
            text="Invoice INV-002 for $200",
        )
        results = doc.to_pydantic(_Invoice)
        self.assertLen(results, 1)

    def test_empty_extractions(self):
        """Empty extractions list returns empty."""
        doc = data.AnnotatedDocument(extractions=[], text="no data")
        results = doc.to_pydantic(_Invoice)
        self.assertEmpty(results)

    def test_none_extractions(self):
        """None extractions returns empty."""
        doc = data.AnnotatedDocument(extractions=None, text="no data")
        results = doc.to_pydantic(_Invoice)
        self.assertEmpty(results)

    def test_multiple_extractions(self):
        """Multiple matching extractions are all converted."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_Person",
                    extraction_text="John",
                    attributes={"age": "30"},
                ),
                data.Extraction(
                    extraction_class="_Person",
                    extraction_text="Jane",
                    attributes={"age": "25"},
                ),
            ],
            text="John is 30 and Jane is 25.",
        )
        results = doc.to_pydantic(_Person)
        self.assertLen(results, 2)
        self.assertEqual(results[0].name, "John")
        self.assertEqual(results[1].name, "Jane")

    def test_invalid_extraction_skipped_with_warning(self):
        """Extractions that fail validation are skipped with a warning."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_Person",
                    extraction_text="John",
                    # Missing required 'age' field with no default
                    attributes={},
                ),
            ],
            text="John",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = doc.to_pydantic(_Person)
            # Should have either a warning or an empty result
            # depending on whether 'age' defaults
            # Since _Person.age has no default, validation will fail
            self.assertLen(w, 1)
            self.assertEmpty(results)

    def test_ignores_extra_attributes(self):
        """Attributes not in the schema model are ignored."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_Person",
                    extraction_text="John",
                    attributes={"age": "30", "unknown_field": "foo"},
                ),
            ],
            text="John is 30.",
        )
        results = doc.to_pydantic(_Person)
        self.assertLen(results, 1)
        self.assertEqual(results[0].name, "John")
        self.assertEqual(results[0].age, 30)

    def test_mixed_classes_filters(self):
        """Only extractions matching the target class are returned."""
        doc = data.AnnotatedDocument(
            extractions=[
                data.Extraction(
                    extraction_class="_Person",
                    extraction_text="John",
                    attributes={"age": "30"},
                ),
                data.Extraction(
                    extraction_class="_Invoice",
                    extraction_text="INV-001",
                    attributes={
                        "amount": "100.0",
                        "due_date": "2024-01-01",
                    },
                ),
            ],
            text="John received Invoice INV-001.",
        )
        persons = doc.to_pydantic(_Person)
        invoices = doc.to_pydantic(_Invoice)
        self.assertLen(persons, 1)
        self.assertLen(invoices, 1)


if __name__ == "__main__":
    absltest.main()
