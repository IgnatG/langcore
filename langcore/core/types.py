"""Core data types for LangCore."""

from __future__ import annotations

import dataclasses
import enum
import textwrap

__all__ = [
    "Constraint",
    "ConstraintType",
    "FormatType",
    "ScoredOutput",
]


class FormatType(enum.Enum):
    """Enumeration of prompt output formats."""

    YAML = "yaml"
    JSON = "json"


class ConstraintType(enum.Enum):
    """Enumeration of constraint types."""

    NONE = "none"


@dataclasses.dataclass
class Constraint:
    """Represents a constraint for model output decoding.

    Attributes:
      constraint_type: The type of constraint applied.
    """

    constraint_type: ConstraintType = ConstraintType.NONE


@dataclasses.dataclass(frozen=True)
class ScoredOutput:
    """Scored output from language model inference."""

    score: float | None = None
    output: str | None = None
    usage: dict[str, int] | None = None

    def __str__(self) -> str:
        score_str = "-" if self.score is None else f"{self.score:.2f}"
        if self.output is None:
            return f"Score: {score_str}\nOutput: None"
        formatted_lines = textwrap.indent(self.output, prefix="  ")
        return f"Score: {score_str}\nOutput:\n{formatted_lines}"
