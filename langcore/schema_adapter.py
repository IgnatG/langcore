"""Pydantic schema adapter for LangCore.

Converts Pydantic BaseModel classes into LangCore-compatible
prompt descriptions, ExampleData objects, and JSON schemas. This
enables an ergonomic schema-first API where users define extraction
targets as Pydantic models instead of manually constructing
ExampleData with Extraction entries.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pydantic

from langcore.core import data
from langcore.core.schema_utils import find_primary_text_field

__all__ = [
    "PydanticSchemaAdapter",
    "SchemaConfig",
    "schema_from_pydantic",
]


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaConfig:
    """Configuration produced by adapting a Pydantic model for extraction.

    Attributes:
        prompt_description: Auto-generated extraction instructions
            derived from the model's field names and descriptions.
        examples: ExampleData objects suitable for few-shot prompting,
            built from the provided example dicts.
        json_schema: The JSON Schema representation of the Pydantic
            model, usable for structured output / constrained decoding.
        model_class: Reference to the original Pydantic model class.
    """

    prompt_description: str
    examples: list[data.ExampleData]
    json_schema: dict[str, Any]
    model_class: type[pydantic.BaseModel]


class PydanticSchemaAdapter:
    """Converts a Pydantic BaseModel into LangCore extraction config.

    The adapter inspects model field metadata (names, types,
    descriptions) to auto-generate a prompt description and converts
    user-supplied example dicts into ``ExampleData`` objects.

    Parameters:
        model_class: The Pydantic model class to adapt.
    """

    def __init__(self, model_class: type[pydantic.BaseModel]) -> None:
        """Initialize the adapter with a Pydantic model class.

        Parameters:
            model_class: A Pydantic BaseModel subclass defining
                the extraction schema.

        Raises:
            TypeError: If ``model_class`` is not a Pydantic BaseModel
                subclass.
        """
        if not (
            isinstance(model_class, type)
            and issubclass(model_class, pydantic.BaseModel)
        ):
            raise TypeError(
                f"Expected a Pydantic BaseModel subclass, got {model_class!r}"
            )
        self._model_class = model_class

    @property
    def model_class(self) -> type[pydantic.BaseModel]:
        """The Pydantic model class being adapted."""
        return self._model_class

    def generate_prompt_description(self) -> str:
        """Generate an extraction prompt from model field metadata.

        Builds a natural-language instruction string describing what
        fields to extract, their types, and any descriptions provided
        via ``Field(description=...)``.

        Returns:
            A prompt description string suitable for LangCore's
            ``prompt_description`` parameter.
        """
        model_name = self._model_class.__name__
        lines: list[str] = [
            f"Extract {model_name} entities from the text.",
            "For each entity found, extract the following fields:",
        ]

        for field_name, field_info in self._model_class.model_fields.items():
            annotation = field_info.annotation
            type_str = _annotation_to_str(annotation)
            desc = field_info.description or ""
            if desc:
                lines.append(f"- {field_name} ({type_str}): {desc}")
            else:
                lines.append(f"- {field_name} ({type_str})")

        lines.append(
            "\nUse exact text from the source when possible."
            " List extractions in order of appearance."
        )
        return "\n".join(lines)

    def examples_to_extraction_data(
        self,
        examples: Sequence[dict[str, Any]],
    ) -> list[data.ExampleData]:
        """Convert example dicts into ExampleData objects.

        Each example dict must contain a ``"text"`` key with the source
        text and an ``"extractions"`` key with a list of dicts whose
        keys match the Pydantic model's field names.

        Parameters:
            examples: Sequence of example dicts with ``text`` and
                ``extractions`` keys.

        Returns:
            A list of ``ExampleData`` objects ready for few-shot
            prompting.

        Raises:
            ValueError: If an example dict is missing required keys.
        """
        result: list[data.ExampleData] = []
        for example in examples:
            if "text" not in example or "extractions" not in example:
                raise ValueError(
                    "Each example must have 'text' and 'extractions'"
                    f" keys. Got keys: {list(example.keys())}"
                )

            extractions: list[data.Extraction] = []
            for ext_dict in example["extractions"]:
                extraction = _dict_to_extraction(ext_dict, self._model_class)
                extractions.append(extraction)

            result.append(
                data.ExampleData(
                    text=example["text"],
                    extractions=extractions,
                )
            )
        return result

    def get_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the Pydantic model.

        Returns:
            A dict representing the JSON Schema, suitable for
            structured output constraints.
        """
        return self._model_class.model_json_schema()

    def adapt(
        self,
        examples: Sequence[dict[str, Any]] | None = None,
        prompt_description: str | None = None,
    ) -> SchemaConfig:
        """Produce a complete SchemaConfig from the model.

        Parameters:
            examples: Optional example dicts for few-shot prompting.
                If ``None``, no examples are included.
            prompt_description: Optional override for the auto-generated
                prompt description. When ``None``, the description is
                generated from field metadata.

        Returns:
            A ``SchemaConfig`` with prompt, examples, and JSON schema.
        """
        desc = (
            prompt_description
            if prompt_description is not None
            else self.generate_prompt_description()
        )
        example_data = self.examples_to_extraction_data(examples) if examples else []
        return SchemaConfig(
            prompt_description=desc,
            examples=example_data,
            json_schema=self.get_json_schema(),
            model_class=self._model_class,
        )


def schema_from_pydantic(
    model_class: type[pydantic.BaseModel],
    examples: Sequence[dict[str, Any]] | None = None,
    prompt_description: str | None = None,
) -> SchemaConfig:
    """Convenience function to adapt a Pydantic model for extraction.

    Parameters:
        model_class: A Pydantic BaseModel subclass defining the schema.
        examples: Optional example dicts for few-shot prompting.
        prompt_description: Optional prompt description override.

    Returns:
        A ``SchemaConfig`` ready for use with ``lx.extract()``.
    """
    adapter = PydanticSchemaAdapter(model_class)
    return adapter.adapt(
        examples=examples,
        prompt_description=prompt_description,
    )


def _annotation_to_str(annotation: Any) -> str:
    """Convert a type annotation to a human-readable string.

    Parameters:
        annotation: A Python type annotation.

    Returns:
        A concise string representation of the type.
    """
    if annotation is None:
        return "Any"

    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        args_str = ", ".join(_annotation_to_str(a) for a in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]" if args_str else origin_name

    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation)


def _dict_to_extraction(
    ext_dict: dict[str, Any],
    model_class: type[pydantic.BaseModel],
) -> data.Extraction:
    """Convert an extraction dict into an Extraction object.

    Maps Pydantic field names to the LangCore Extraction structure:
    - The first ``str`` field is used as ``extraction_text``.
    - The model class name is used as ``extraction_class``.
    - Remaining fields become ``attributes``.

    Parameters:
        ext_dict: Dict with keys matching model field names.
        model_class: The Pydantic model for field metadata.

    Returns:
        An ``Extraction`` object.
    """
    model_name = model_class.__name__

    # Validate the dict against the Pydantic model to catch errors
    # early. We use model_validate to ensure type coercion works.
    validated = model_class.model_validate(ext_dict)
    validated_dict = validated.model_dump()

    # Determine the primary text field: look for a field named
    # 'text', 'name', or 'value'. Fall back to the first str field.
    text_field = find_primary_text_field(model_class)
    extraction_text = str(validated_dict.get(text_field, ""))

    # Remaining fields become attributes
    attributes: dict[str, str | list[str]] = {}
    for key, value in validated_dict.items():
        if key == text_field:
            continue
        if isinstance(value, list):
            attributes[key] = [str(v) for v in value]
        else:
            attributes[key] = str(value)

    return data.Extraction(
        extraction_class=model_name,
        extraction_text=extraction_text,
        attributes=attributes if attributes else None,
    )


# Backward-compatible alias for external code that may have
# imported the private helper directly.
_find_primary_text_field = find_primary_text_field
