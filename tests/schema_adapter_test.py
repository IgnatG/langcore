"""Tests for langcore.schema_adapter module."""

from __future__ import annotations

import pydantic
from absl.testing import absltest, parameterized

from langcore import schema_adapter
from langcore.core import data


class _Invoice(pydantic.BaseModel):
    """Test model: Invoice with descriptions."""

    invoice_number: str = pydantic.Field(description="Invoice ID like INV-001")
    amount: float = pydantic.Field(description="Total amount in dollars")
    due_date: str = pydantic.Field(description="Due date in YYYY-MM-DD format")


class _Person(pydantic.BaseModel):
    """Test model: simple person with name as primary field."""

    name: str = pydantic.Field(description="Full name")
    age: int = pydantic.Field(description="Age in years")


class _Medication(pydantic.BaseModel):
    """Test model: medication without explicit descriptions."""

    name: str
    dosage: str
    frequency: str


class _NoTextField(pydantic.BaseModel):
    """Test model: no text/name/value/title field."""

    code: int
    category: str


class PydanticSchemaAdapterInitTest(absltest.TestCase):
    """Tests for PydanticSchemaAdapter initialization."""

    def test_accepts_valid_basemodel_subclass(self):
        """Adapter accepts a valid Pydantic BaseModel subclass."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        self.assertEqual(adapter.model_class, _Invoice)

    def test_rejects_non_basemodel_class(self):
        """Adapter raises TypeError for non-BaseModel inputs."""
        with self.assertRaises(TypeError):
            schema_adapter.PydanticSchemaAdapter(dict)

    def test_rejects_instance(self):
        """Adapter raises TypeError when given a model instance."""
        instance = _Invoice(
            invoice_number="INV-001", amount=100.0, due_date="2024-01-01"
        )
        with self.assertRaises(TypeError):
            schema_adapter.PydanticSchemaAdapter(instance)


class GeneratePromptDescriptionTest(parameterized.TestCase):
    """Tests for prompt description generation."""

    def test_includes_model_name(self):
        """Generated prompt includes the model class name."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        desc = adapter.generate_prompt_description()
        self.assertIn("Invoice", desc)

    def test_includes_field_descriptions(self):
        """Generated prompt includes Field descriptions."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        desc = adapter.generate_prompt_description()
        self.assertIn("Invoice ID like INV-001", desc)
        self.assertIn("Total amount in dollars", desc)
        self.assertIn("YYYY-MM-DD", desc)

    def test_includes_field_names(self):
        """Generated prompt includes field names."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        desc = adapter.generate_prompt_description()
        self.assertIn("invoice_number", desc)
        self.assertIn("amount", desc)
        self.assertIn("due_date", desc)

    def test_includes_type_annotations(self):
        """Generated prompt includes type info."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        desc = adapter.generate_prompt_description()
        self.assertIn("str", desc)
        self.assertIn("float", desc)

    def test_fields_without_description(self):
        """Fields without descriptions still appear in the prompt."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Medication)
        desc = adapter.generate_prompt_description()
        self.assertIn("name", desc)
        self.assertIn("dosage", desc)
        self.assertIn("frequency", desc)


class ExamplesToExtractionDataTest(absltest.TestCase):
    """Tests for converting example dicts to ExampleData."""

    def test_basic_conversion(self):
        """Correctly converts example dicts to ExampleData."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        examples = [
            {
                "text": "Invoice INV-001 for $100 due 2024-01-01",
                "extractions": [
                    {
                        "invoice_number": "INV-001",
                        "amount": 100.0,
                        "due_date": "2024-01-01",
                    }
                ],
            }
        ]
        result = adapter.examples_to_extraction_data(examples)
        self.assertLen(result, 1)
        self.assertIsInstance(result[0], data.ExampleData)
        self.assertEqual(result[0].text, "Invoice INV-001 for $100 due 2024-01-01")
        self.assertLen(result[0].extractions, 1)

    def test_extraction_class_is_model_name(self):
        """Extraction class is set to the model class name."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        examples = [
            {
                "text": "Invoice INV-001 for $100 due 2024-01-01",
                "extractions": [
                    {
                        "invoice_number": "INV-001",
                        "amount": 100.0,
                        "due_date": "2024-01-01",
                    }
                ],
            }
        ]
        result = adapter.examples_to_extraction_data(examples)
        ext = result[0].extractions[0]
        self.assertEqual(ext.extraction_class, "_Invoice")

    def test_primary_text_uses_name_field(self):
        """Models with 'name' field use it as extraction_text."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Person)
        examples = [
            {
                "text": "John is 30 years old.",
                "extractions": [{"name": "John", "age": 30}],
            }
        ]
        result = adapter.examples_to_extraction_data(examples)
        ext = result[0].extractions[0]
        self.assertEqual(ext.extraction_text, "John")

    def test_remaining_fields_become_attributes(self):
        """Non-primary fields are stored as attributes."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Person)
        examples = [
            {
                "text": "John is 30 years old.",
                "extractions": [{"name": "John", "age": 30}],
            }
        ]
        result = adapter.examples_to_extraction_data(examples)
        ext = result[0].extractions[0]
        self.assertIn("age", ext.attributes)
        self.assertEqual(ext.attributes["age"], "30")

    def test_missing_text_key_raises(self):
        """ValueError raised when example dict lacks 'text' key."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        with self.assertRaises(ValueError):
            adapter.examples_to_extraction_data([{"extractions": []}])

    def test_missing_extractions_key_raises(self):
        """ValueError raised when example dict lacks 'extractions' key."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        with self.assertRaises(ValueError):
            adapter.examples_to_extraction_data([{"text": "some text"}])

    def test_multiple_extractions_per_example(self):
        """Multiple extractions in one example are all converted."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Person)
        examples = [
            {
                "text": "John and Jane are friends.",
                "extractions": [
                    {"name": "John", "age": 30},
                    {"name": "Jane", "age": 25},
                ],
            }
        ]
        result = adapter.examples_to_extraction_data(examples)
        self.assertLen(result[0].extractions, 2)

    def test_multiple_examples(self):
        """Multiple example dicts produce multiple ExampleData."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Person)
        examples = [
            {
                "text": "John is 30.",
                "extractions": [{"name": "John", "age": 30}],
            },
            {
                "text": "Jane is 25.",
                "extractions": [{"name": "Jane", "age": 25}],
            },
        ]
        result = adapter.examples_to_extraction_data(examples)
        self.assertLen(result, 2)


class GetJsonSchemaTest(absltest.TestCase):
    """Tests for JSON schema generation."""

    def test_returns_dict(self):
        """JSON schema is a dict."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        schema = adapter.get_json_schema()
        self.assertIsInstance(schema, dict)

    def test_has_properties(self):
        """JSON schema contains all model fields."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        schema = adapter.get_json_schema()
        self.assertIn("properties", schema)
        props = schema["properties"]
        self.assertIn("invoice_number", props)
        self.assertIn("amount", props)
        self.assertIn("due_date", props)


class AdaptTest(absltest.TestCase):
    """Tests for the full adapt() method."""

    def test_produces_schema_config(self):
        """adapt() returns a SchemaConfig."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        config = adapter.adapt()
        self.assertIsInstance(config, schema_adapter.SchemaConfig)

    def test_config_has_prompt_description(self):
        """SchemaConfig contains a non-empty prompt description."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        config = adapter.adapt()
        self.assertTrue(config.prompt_description)

    def test_config_with_examples(self):
        """SchemaConfig includes examples when provided."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        examples = [
            {
                "text": "Invoice INV-001 for $100 due 2024-01-01",
                "extractions": [
                    {
                        "invoice_number": "INV-001",
                        "amount": 100.0,
                        "due_date": "2024-01-01",
                    }
                ],
            }
        ]
        config = adapter.adapt(examples=examples)
        self.assertLen(config.examples, 1)

    def test_prompt_override(self):
        """Custom prompt_description overrides auto-generated one."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        config = adapter.adapt(prompt_description="Custom prompt")
        self.assertEqual(config.prompt_description, "Custom prompt")

    def test_config_stores_model_class(self):
        """SchemaConfig references the original model class."""
        adapter = schema_adapter.PydanticSchemaAdapter(_Invoice)
        config = adapter.adapt()
        self.assertEqual(config.model_class, _Invoice)


class SchemaFromPydanticTest(absltest.TestCase):
    """Tests for the schema_from_pydantic convenience function."""

    def test_convenience_function(self):
        """schema_from_pydantic produces a valid SchemaConfig."""
        config = schema_adapter.schema_from_pydantic(_Invoice)
        self.assertIsInstance(config, schema_adapter.SchemaConfig)
        self.assertIn("Invoice", config.prompt_description)

    def test_with_examples(self):
        """schema_from_pydantic passes examples through."""
        config = schema_adapter.schema_from_pydantic(
            _Invoice,
            examples=[
                {
                    "text": "INV-001 for $50 due 2024-01-01",
                    "extractions": [
                        {
                            "invoice_number": "INV-001",
                            "amount": 50.0,
                            "due_date": "2024-01-01",
                        }
                    ],
                }
            ],
        )
        self.assertLen(config.examples, 1)


class FindPrimaryTextFieldTest(parameterized.TestCase):
    """Tests for _find_primary_text_field."""

    @parameterized.parameters(
        (_Person, "name"),
        (_NoTextField, "category"),
    )
    def test_primary_field_selection(self, model_cls, expected):
        """Correct primary text field is selected."""
        result = schema_adapter._find_primary_text_field(model_cls)
        self.assertEqual(result, expected)


class AnnotationToStrTest(parameterized.TestCase):
    """Tests for _annotation_to_str."""

    @parameterized.parameters(
        (str, "str"),
        (int, "int"),
        (float, "float"),
        (None, "Any"),
    )
    def test_basic_types(self, annotation, expected):
        """Basic type annotations produce expected strings."""
        result = schema_adapter._annotation_to_str(annotation)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    absltest.main()
