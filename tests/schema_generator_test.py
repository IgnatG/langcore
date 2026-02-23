"""Tests for langcore.schema_generator module."""

from __future__ import annotations

import pydantic
from absl.testing import absltest, parameterized

from langcore import schema_generator


class SchemaFromExampleTest(absltest.TestCase):
    """Tests for schema_from_example."""

    def test_basic_string_fields(self):
        """Dict with string values produces str-typed fields."""
        model = schema_generator.schema_from_example({"name": "John", "city": "NYC"})
        self.assertTrue(issubclass(model, pydantic.BaseModel))
        fields = model.model_fields
        self.assertIn("name", fields)
        self.assertIn("city", fields)

    def test_numeric_fields(self):
        """Dict with int/float values produces correct types."""
        model = schema_generator.schema_from_example({"age": 30, "score": 9.5})
        instance = model(age=25, score=8.0)
        self.assertEqual(instance.age, 25)
        self.assertAlmostEqual(instance.score, 8.0)

    def test_bool_fields(self):
        """Dict with bool values produces bool-typed fields."""
        model = schema_generator.schema_from_example({"active": True, "name": "test"})
        instance = model(active=False, name="x")
        self.assertFalse(instance.active)

    def test_list_fields(self):
        """Dict with list values produces list-typed fields."""
        model = schema_generator.schema_from_example(
            {"tags": ["a", "b"], "name": "test"}
        )
        instance = model(tags=["c"], name="x")
        self.assertEqual(instance.tags, ["c"])

    def test_custom_name(self):
        """Generated model uses the provided name."""
        model = schema_generator.schema_from_example({"x": 1}, name="MyModel")
        self.assertEqual(model.__name__, "MyModel")

    def test_empty_dict_raises(self):
        """Empty dict raises ValueError."""
        with self.assertRaises(ValueError):
            schema_generator.schema_from_example({})

    def test_none_value_creates_optional_field(self):
        """None values create optional fields."""
        model = schema_generator.schema_from_example(
            {"name": "test", "optional_field": None}
        )
        # Should accept None for optional_field
        instance = model(name="test", optional_field=None)
        self.assertIsNone(instance.optional_field)


class SchemaFromExamplesTest(parameterized.TestCase):
    """Tests for schema_from_examples."""

    def test_single_example_delegates(self):
        """Single-element list delegates to schema_from_example."""
        model = schema_generator.schema_from_examples([{"name": "John", "age": 30}])
        self.assertTrue(issubclass(model, pydantic.BaseModel))

    def test_merges_keys_across_examples(self):
        """Fields from all examples are present in the merged model."""
        model = schema_generator.schema_from_examples(
            [
                {"name": "John", "age": 30},
                {"name": "Jane", "email": "jane@example.com"},
            ]
        )
        fields = model.model_fields
        self.assertIn("name", fields)
        self.assertIn("age", fields)
        self.assertIn("email", fields)

    def test_missing_field_is_optional(self):
        """Fields absent from some examples become optional."""
        model = schema_generator.schema_from_examples(
            [
                {"name": "John", "age": 30},
                {"name": "Jane"},
            ]
        )
        # 'age' is only in the first example, so should be optional
        instance = model(name="Bob", age=None)
        self.assertIsNone(instance.age)

    def test_type_merging_int_float(self):
        """Mixed int and float for the same field becomes float."""
        model = schema_generator.schema_from_examples(
            [
                {"score": 10},
                {"score": 9.5},
            ]
        )
        instance = model(score=8.5)
        self.assertAlmostEqual(instance.score, 8.5)

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with self.assertRaises(ValueError):
            schema_generator.schema_from_examples([])

    def test_custom_name(self):
        """Generated model uses the provided name."""
        model = schema_generator.schema_from_examples([{"x": 1}], name="Custom")
        self.assertEqual(model.__name__, "Custom")


class InferTypeTest(parameterized.TestCase):
    """Tests for _infer_type."""

    @parameterized.parameters(
        ("hello", str),
        (42, int),
        (3.14, float),
        (True, bool),
        (None, str),
    )
    def test_scalar_types(self, value, expected):
        """Scalar values produce correct type inference."""
        result = schema_generator._infer_type(value)
        self.assertEqual(result, expected)

    def test_empty_list(self):
        """Empty list defaults to list[str]."""
        result = schema_generator._infer_type([])
        self.assertEqual(result, list[str])

    def test_dict_value(self):
        """Dict values produce dict[str, Any]."""
        from typing import Any

        result = schema_generator._infer_type({"a": 1})
        self.assertEqual(result, dict[str, Any])


class MergeTypesTest(parameterized.TestCase):
    """Tests for _merge_types."""

    def test_uniform_types(self):
        """All same type returns that type."""
        result = schema_generator._merge_types([str, str, str])
        self.assertEqual(result, str)

    def test_int_float_merge(self):
        """int + float merges to float."""
        result = schema_generator._merge_types([int, float])
        self.assertEqual(result, float)

    def test_mixed_types_fallback(self):
        """Incompatible types fall back to str."""
        result = schema_generator._merge_types([str, int])
        self.assertEqual(result, str)


if __name__ == "__main__":
    absltest.main()
