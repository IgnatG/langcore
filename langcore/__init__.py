"""LangCore: Extract structured information from text with LLMs.

This package provides the main extract and visualize functions,
with lazy loading for other submodules accessed via attribute access.
"""

from __future__ import annotations

__version__ = "1.0.1"

import importlib
import sys
from typing import Any

from langcore import visualization
from langcore.extraction import async_extract as async_extract_func
from langcore.extraction import extract as extract_func
from langcore.schema_adapter import schema_from_pydantic
from langcore.schema_generator import schema_from_example, schema_from_examples

__all__ = [
    # Submodules exposed lazily on attribute access for ergonomics:
    "annotation",
    "async_extract",
    "core",
    "data",
    "evaluate",
    "evaluation",
    "exceptions",
    # Public convenience functions (thin wrappers)
    "extract",
    "factory",
    "hooks",
    "inference",
    "io",
    "plugins",
    "prompting",
    "providers",
    "resolver",
    "schema",
    "schema_adapter",
    "schema_from_example",
    "schema_from_examples",
    # Schema utilities
    "schema_from_pydantic",
    "schema_generator",
    "visualization",
    "visualize",
]

_CACHE: dict[str, Any] = {}


def extract(*args: Any, **kwargs: Any):
    """Top-level API: lx.extract(...)."""
    return extract_func(*args, **kwargs)


async def async_extract(*args: Any, **kwargs: Any):
    """Top-level async API: await lx.async_extract(...)."""
    return await async_extract_func(*args, **kwargs)


def evaluate(*args: Any, **kwargs: Any):
    """Top-level API: lx.evaluate(predictions, ground_truth, ...).\n\n    Convenience wrapper around ``ExtractionMetrics.evaluate()``.\n    Accepts the same arguments as ``ExtractionMetrics.evaluate()``\n    plus an optional ``schema`` keyword for per-field breakdown.\n"""
    from langcore.evaluation import ExtractionMetrics

    schema = kwargs.pop("schema", None)
    strict_attributes = kwargs.pop("strict_attributes", False)
    metrics = ExtractionMetrics(schema=schema, strict_attributes=strict_attributes)
    return metrics.evaluate(*args, **kwargs)


def visualize(*args: Any, **kwargs: Any):
    """Top-level API: lx.visualize(...)."""
    return visualization.visualize(*args, **kwargs)


# PEP 562 lazy loading
_LAZY_MODULES = {
    "annotation": "langcore.annotation",
    "chunking": "langcore.chunking",
    "data": "langcore.data",
    "data_lib": "langcore.data_lib",
    "debug_utils": "langcore.core.debug_utils",
    "evaluation": "langcore.evaluation",
    "exceptions": "langcore.exceptions",
    "factory": "langcore.factory",
    "hooks": "langcore.hooks",
    "inference": "langcore.inference",
    "io": "langcore.io",
    "progress": "langcore.progress",
    "prompting": "langcore.prompting",
    "providers": "langcore.providers",
    "resolver": "langcore.resolver",
    "schema": "langcore.schema",
    "schema_adapter": "langcore.schema_adapter",
    "schema_generator": "langcore.schema_generator",
    "tokenizer": "langcore.tokenizer",
    "visualization": "langcore.visualization",
    "core": "langcore.core",
    "plugins": "langcore.plugins",
}


def __getattr__(name: str) -> Any:
    if name in _CACHE:
        return _CACHE[name]
    modpath = _LAZY_MODULES.get(name)
    if modpath is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(modpath)
    # ensure future 'import langcore.<name>' returns the same module
    sys.modules[f"{__name__}.{name}"] = module
    setattr(sys.modules[__name__], name, module)
    _CACHE[name] = module
    return module


def __dir__():
    return sorted(__all__)
