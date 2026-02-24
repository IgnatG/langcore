"""Auto-generate Pydantic models from example dictionaries.

This module provides utilities for inferring Pydantic model schemas
from one or more example dictionaries. It inspects the values to
determine appropriate types and merges multiple examples to produce
robust type annotations.
"""

from __future__ import annotations

import functools
import operator
from typing import Any

import pydantic

__all__ = [
    "schema_from_example",
    "schema_from_examples",
]


def schema_from_example(
    example_dict: dict[str, Any],
    name: str = "GeneratedSchema",
) -> type[pydantic.BaseModel]:
    """Generate a Pydantic model from a single example dict.

    Inspects each value to infer the appropriate Python type and
    creates a Pydantic model class with those fields.

    Parameters:
        example_dict: A dict whose keys become field names and whose
            values are used for type inference.
        name: The name for the generated model class.

    Returns:
        A dynamically created Pydantic BaseModel subclass.

    Raises:
        ValueError: If ``example_dict`` is empty.
    """
    if not example_dict:
        raise ValueError("Cannot generate schema from an empty dict.")

    field_definitions: dict[str, Any] = {}
    for key, value in example_dict.items():
        inferred_type = _infer_type(value)
        if _is_optional(value):
            field_definitions[key] = (
                inferred_type | None,
                pydantic.Field(default=None),
            )
        else:
            field_definitions[key] = (inferred_type, ...)

    return pydantic.create_model(name, **field_definitions)


def schema_from_examples(
    examples: list[dict[str, Any]],
    name: str = "GeneratedSchema",
) -> type[pydantic.BaseModel]:
    """Generate a Pydantic model by merging multiple example dicts.

    Collects all unique keys across examples and infers the broadest
    compatible type for each field. Fields that are missing in some
    examples are marked as ``Optional``.

    Parameters:
        examples: A list of dicts to merge for type inference.
        name: The name for the generated model class.

    Returns:
        A dynamically created Pydantic BaseModel subclass.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    if not examples:
        raise ValueError("Cannot generate schema from an empty list.")

    if len(examples) == 1:
        return schema_from_example(examples[0], name=name)

    # Collect all field names and their observed types
    all_keys: dict[str, list[Any]] = {}
    key_presence: dict[str, int] = {}
    total = len(examples)

    for example in examples:
        for key, value in example.items():
            all_keys.setdefault(key, []).append(value)
            key_presence[key] = key_presence.get(key, 0) + 1

    field_definitions: dict[str, Any] = {}
    for key, values in all_keys.items():
        merged_type = _merge_types([_infer_type(v) for v in values])
        is_optional = key_presence[key] < total or any(v is None for v in values)
        if is_optional:
            field_definitions[key] = (
                merged_type | None,
                pydantic.Field(default=None),
            )
        else:
            field_definitions[key] = (merged_type, ...)

    return pydantic.create_model(name, **field_definitions)


def _infer_type(value: Any) -> type:
    """Infer a Python type from a sample value.

    Parameters:
        value: A sample value to inspect.

    Returns:
        The inferred Python type annotation.
    """
    if value is None:
        return str  # Default to str for None values
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, str):
        return str
    if isinstance(value, list):
        if not value:
            return list[str]
        inner_types = {_infer_type(v) for v in value}
        if len(inner_types) == 1:
            return list[inner_types.pop()]
        return list[str]  # Mixed types default to str
    if isinstance(value, dict):
        return dict[str, Any]
    return str


def _is_optional(value: Any) -> bool:
    """Check whether a value suggests an optional field.

    Parameters:
        value: The value to check.

    Returns:
        True if the value is None.
    """
    return value is None


def _merge_types(types_list: list[type]) -> type:
    """Merge multiple inferred types into a single compatible type.

    When all observed types are the same, returns that type. When
    ``int`` and ``float`` are mixed, returns ``float``.  For other
    mixed-type combinations (e.g. ``int`` + ``str``), a ``Union``
    type is returned so that Pydantic can accept any of the
    observed types.

    Parameters:
        types_list: A list of inferred types to merge.

    Returns:
        The merged type annotation.
    """
    unique = set(types_list)

    if len(unique) == 1:
        return unique.pop()

    # int + float → float
    if unique == {int, float}:
        return float

    # All numeric → float
    if unique <= {int, float, bool}:
        return float

    # Mixed types → Union of observed types (sorted for determinism)
    sorted_types = sorted(unique, key=lambda t: t.__name__)
    return functools.reduce(operator.or_, sorted_types)  # type: ignore[return-value]
