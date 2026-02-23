"""Core error types for LangCore.

This module defines all base exceptions for LangCore. These are the
foundational error types that are used throughout the codebase.
"""

from __future__ import annotations

__all__ = [
    "FormatError",
    "FormatParseError",
    "InferenceConfigError",
    "InferenceError",
    "InferenceOutputError",
    "InferenceRuntimeError",
    "InternalError",
    "InvalidDocumentError",
    "LangCoreError",
    "ProviderError",
    "SchemaError",
]


class LangCoreError(Exception):
    """Base exception for all LangCore errors.

    All exceptions raised by LangCore should inherit from this class.
    This allows users to catch all LangCore-specific errors with a single
    except clause.
    """


class InferenceError(LangCoreError):
    """Base exception for inference-related errors."""


class InferenceConfigError(InferenceError):
    """Exception raised for configuration errors.

    This includes missing API keys, invalid model IDs, or other
    configuration-related issues that prevent model instantiation.
    """


class InferenceRuntimeError(InferenceError):
    """Exception raised for runtime inference errors.

    This includes API call failures, network errors, or other issues
    that occur during inference execution.
    """

    def __init__(
        self,
        message: str,
        *,
        original: BaseException | None = None,
        provider: str | None = None,
    ) -> None:
        """Initialize the runtime error.

        Args:
          message: Error message.
          original: Original exception from the provider SDK.
          provider: Name of the provider that raised the error.
        """
        super().__init__(message)
        self.original = original
        self.provider = provider


class InferenceOutputError(LangCoreError):
    """Exception raised when no scored outputs are available from the language model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InvalidDocumentError(LangCoreError):
    """Exception raised when document input is invalid.

    This includes cases like duplicate document IDs or malformed documents.
    """


class InternalError(LangCoreError):
    """Exception raised for internal invariant violations.

    This indicates a bug in LangCore itself rather than user error.
    """


class ProviderError(LangCoreError):
    """Provider/backend specific error."""


class SchemaError(LangCoreError):
    """Schema validation/serialization error."""


class FormatError(LangCoreError):
    """Base exception for format handling errors."""


class FormatParseError(FormatError):
    """Raised when format parsing fails.

    This consolidates all parsing errors including:
    - Missing fence markers when required
    - Multiple fenced blocks
    - JSON/YAML decode errors
    - Missing wrapper keys
    - Invalid structure
    """
