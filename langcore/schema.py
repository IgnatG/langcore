"""Re-export core schema types for convenience.

New code should import from ``langcore.core.schema`` directly.
"""

from __future__ import annotations

from langcore.core.data import ATTRIBUTE_SUFFIX, EXTRACTIONS_KEY
from langcore.core.schema import (
    BaseSchema,
    Constraint,
    ConstraintType,
    FormatModeSchema,
)
from langcore.providers.schemas.gemini import GeminiSchema

__all__ = [
    "ATTRIBUTE_SUFFIX",
    "EXTRACTIONS_KEY",
    "BaseSchema",
    "Constraint",
    "ConstraintType",
    "FormatModeSchema",
    "GeminiSchema",
]
