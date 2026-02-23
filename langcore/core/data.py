"""Classes used to represent core data types of annotation pipeline."""

from __future__ import annotations

import dataclasses
import enum
import uuid
from typing import Any

import pydantic

from langcore.core import tokenizer, types
from langcore.core.schema_utils import find_primary_text_field

FormatType = types.FormatType

EXTRACTIONS_KEY = "extractions"
ATTRIBUTE_SUFFIX = "_attributes"

__all__ = [
    "ATTRIBUTE_SUFFIX",
    "EXTRACTIONS_KEY",
    "AlignmentStatus",
    "AnnotatedDocument",
    "CharInterval",
    "Document",
    "ExampleData",
    "Extraction",
    "FormatType",
]


class AlignmentStatus(enum.Enum):
    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclasses.dataclass
class CharInterval:
    """Class for representing a character interval.

    Attributes:
      start_pos: The starting position of the interval (inclusive).
      end_pos: The ending position of the interval (exclusive).
    """

    start_pos: int | None = None
    end_pos: int | None = None


@dataclasses.dataclass(init=False)
class Extraction:
    """Represents an extraction extracted from text.

    This class encapsulates an extraction's characteristics and its position
    within the source text. It can represent a diverse range of information for
    NLP information extraction tasks.

    Attributes:
      extraction_class: The class of the extraction.
      extraction_text: The text of the extraction.
      char_interval: The character interval of the extraction in the original
        text.
      alignment_status: The alignment status of the extraction.
      extraction_index: The index of the extraction in the list of extractions.
      group_index: The index of the group the extraction belongs to.
      description: A description of the extraction.
      attributes: A list of attributes of the extraction.
      confidence_score: Confidence score in ``[0.0, 1.0]``.  For
        single-pass extraction this combines alignment quality (70%)
        and token overlap ratio (30%).  For multi-pass extraction the
        cross-pass appearance frequency is further combined with the
        per-extraction alignment confidence.  ``None`` when confidence
        has not been computed.
      token_interval: The token interval of the extraction.
    """

    extraction_class: str
    extraction_text: str
    char_interval: CharInterval | None = None
    alignment_status: AlignmentStatus | None = None
    extraction_index: int | None = None
    group_index: int | None = None
    description: str | None = None
    attributes: dict[str, str | list[str]] | None = None
    confidence_score: float | None = None
    _token_interval: tokenizer.TokenInterval | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    def __init__(
        self,
        extraction_class: str,
        extraction_text: str,
        *,
        token_interval: tokenizer.TokenInterval | None = None,
        char_interval: CharInterval | None = None,
        alignment_status: AlignmentStatus | None = None,
        extraction_index: int | None = None,
        group_index: int | None = None,
        description: str | None = None,
        attributes: dict[str, str | list[str]] | None = None,
        confidence_score: float | None = None,
    ):
        self.extraction_class = extraction_class
        self.extraction_text = extraction_text
        self.char_interval = char_interval
        self._token_interval = token_interval
        self.alignment_status = alignment_status
        self.extraction_index = extraction_index
        self.group_index = group_index
        self.description = description
        self.attributes = attributes
        self.confidence_score = confidence_score

    @property
    def token_interval(self) -> tokenizer.TokenInterval | None:
        return self._token_interval

    @token_interval.setter
    def token_interval(self, value: tokenizer.TokenInterval | None) -> None:
        self._token_interval = value


@dataclasses.dataclass
class Document:
    """Document class for annotating documents.

    Attributes:
      text: Raw text representation for the document.
      document_id: Unique identifier for each document and is auto-generated if
        not set.
      additional_context: Additional context to supplement prompt instructions.
      tokenized_text: Tokenized text for the document, computed from `text`.
    """

    text: str
    additional_context: str | None = None
    _document_id: str | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    _tokenized_text: tokenizer.TokenizedText | None = dataclasses.field(
        init=False, default=None, repr=False, compare=False
    )

    def __init__(
        self,
        text: str,
        *,
        document_id: str | None = None,
        additional_context: str | None = None,
    ):
        self.text = text
        self.additional_context = additional_context
        self._document_id = document_id

    @property
    def document_id(self) -> str:
        """Returns the document ID, generating a unique one if not set."""
        if self._document_id is None:
            self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
        return self._document_id

    @document_id.setter
    def document_id(self, value: str | None) -> None:
        """Sets the document ID."""
        self._document_id = value

    @property
    def tokenized_text(self) -> tokenizer.TokenizedText:
        if self._tokenized_text is None:
            self._tokenized_text = tokenizer.tokenize(self.text)
        return self._tokenized_text

    @tokenized_text.setter
    def tokenized_text(self, value: tokenizer.TokenizedText) -> None:
        self._tokenized_text = value


@dataclasses.dataclass
class AnnotatedDocument:
    """Class for representing annotated documents.

    Attributes:
      document_id: Unique identifier for each document - autogenerated if not
        set.
      extractions: List of extractions in the document.
      text: Raw text representation of the document.
      tokenized_text: Tokenized text of the document, computed from `text`.
      usage: Optional token usage statistics from the language model.
    """

    extractions: list[Extraction] | None = None
    text: str | None = None
    usage: dict[str, int] | None = None
    _document_id: str | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    _tokenized_text: tokenizer.TokenizedText | None = dataclasses.field(
        init=False, default=None, repr=False, compare=False
    )

    def __init__(
        self,
        *,
        document_id: str | None = None,
        extractions: list[Extraction] | None = None,
        text: str | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.extractions = extractions
        self.text = text
        self.usage = usage
        self._document_id = document_id

    @property
    def document_id(self) -> str:
        """Returns the document ID, generating a unique one if not set."""
        if self._document_id is None:
            self._document_id = f"doc_{uuid.uuid4().hex[:8]}"
        return self._document_id

    @document_id.setter
    def document_id(self, value: str | None) -> None:
        """Sets the document ID."""
        self._document_id = value

    @property
    def tokenized_text(self) -> tokenizer.TokenizedText | None:
        if self._tokenized_text is None and self.text is not None:
            self._tokenized_text = tokenizer.tokenize(self.text)
        return self._tokenized_text

    @tokenized_text.setter
    def tokenized_text(self, value: tokenizer.TokenizedText) -> None:
        self._tokenized_text = value

    @property
    def average_confidence(self) -> float | None:
        """Compute the mean confidence score across all extractions.

        Returns ``None`` when no extractions exist or none of them have
        a ``confidence_score`` set.  Otherwise returns the arithmetic
        mean of the non-null scores.
        """
        if not self.extractions:
            return None
        scores = [
            e.confidence_score
            for e in self.extractions
            if e.confidence_score is not None
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    def to_pydantic(
        self,
        schema: type[pydantic.BaseModel],
    ) -> list[pydantic.BaseModel]:
        """Convert extractions to Pydantic model instances.

        Maps each ``Extraction`` back to an instance of the given Pydantic
        model. The ``extraction_text`` is mapped to the primary text field
        (determined by the same heuristic as ``schema_adapter``), and
        ``attributes`` are mapped to the remaining fields.

        Parameters:
            schema: A Pydantic BaseModel subclass to instantiate.

        Returns:
            A list of validated Pydantic model instances. Extractions
            that cannot be mapped (e.g., wrong ``extraction_class``)
            are skipped with a warning.
        """
        if not self.extractions:
            return []

        results: list[pydantic.BaseModel] = []
        model_name = schema.__name__
        primary_field = find_primary_text_field(schema)

        for extraction in self.extractions:
            # Only convert extractions that match the schema class
            if (
                extraction.extraction_class != model_name
                and extraction.extraction_class.lower() != model_name.lower()
            ):
                continue

            field_data: dict[str, Any] = {
                primary_field: extraction.extraction_text,
            }

            if extraction.attributes:
                for key, value in extraction.attributes.items():
                    if key in schema.model_fields:
                        field_data[key] = value

            try:
                instance = schema.model_validate(field_data)
                results.append(instance)
            except pydantic.ValidationError:
                # Skip extractions that don't validate against schema
                import warnings

                warnings.warn(
                    f"Extraction '{extraction.extraction_text}' could"
                    f" not be validated against {model_name}. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )

        return results


@dataclasses.dataclass
class ExampleData:
    """A single training/example data instance for a structured prompting.

    Attributes:
      text: The raw input text (sentence, paragraph, etc.).
      extractions: A list of Extraction objects extracted from the text.
    """

    text: str
    extractions: list[Extraction] = dataclasses.field(default_factory=list)


# Backward-compatible alias for external code that may have
# imported the private helper directly.
_find_primary_text_field = find_primary_text_field
