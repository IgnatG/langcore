"""Public exceptions API for LangCore.

This module re-exports exceptions from core.exceptions for backward compatibility.
All new code should import directly from langcore.core.exceptions.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

from langcore.core import exceptions as core_exceptions

# Backward compat re-exports
InferenceConfigError = core_exceptions.InferenceConfigError
InferenceError = core_exceptions.InferenceError
InferenceOutputError = core_exceptions.InferenceOutputError
InferenceRuntimeError = core_exceptions.InferenceRuntimeError
LangCoreError = core_exceptions.LangCoreError
ProviderError = core_exceptions.ProviderError
SchemaError = core_exceptions.SchemaError

__all__ = [
    "InferenceConfigError",
    "InferenceError",
    "InferenceOutputError",
    "InferenceRuntimeError",
    "LangCoreError",
    "ProviderError",
    "SchemaError",
]
