"""Shared Pydantic schema utilities for LangCore.

Contains helper functions used across multiple modules to avoid
code duplication.
"""

from __future__ import annotations

import pydantic

__all__ = ["find_primary_text_field"]


def find_primary_text_field(
    model_class: type[pydantic.BaseModel],
) -> str:
    """Identify the primary text field from a Pydantic model.

    Looks for fields named 'text', 'name', 'value', or 'title' first.
    If none found, uses the first field with a ``str`` annotation.
    Falls back to the very first field as last resort.

    Parameters:
        model_class: The Pydantic model class.

    Returns:
        The field name to use as ``extraction_text``.

    Raises:
        ValueError: If no suitable text field is found.
    """
    # Priority names
    for candidate in ("text", "name", "value", "title"):
        if candidate in model_class.model_fields:
            return candidate

    # Fall back to first str field
    for field_name, field_info in model_class.model_fields.items():
        if field_info.annotation is str:
            return field_name

    # Last resort: first field
    fields = list(model_class.model_fields.keys())
    if fields:
        return fields[0]

    raise ValueError(
        f"Cannot determine primary text field for {model_class.__name__}."
        " Add a 'text', 'name', or 'value' field, or ensure at least"
        " one str-typed field exists."
    )
