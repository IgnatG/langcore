"""LangCore: Extract structured information from text with LLMs.

This package provides the main extract and visualize functions,
with lazy loading for other submodules accessed via attribute access.
"""

from __future__ import annotations

__version__ = "1.0.2"

import importlib
import sys
from typing import Any

from langcore import _config as _config_mod
from langcore import visualization
from langcore.extraction import async_extract as async_extract_func
from langcore.extraction import extract as extract_func
from langcore.hooks import Hooks
from langcore.schema_adapter import schema_from_pydantic
from langcore.schema_generator import schema_from_example, schema_from_examples

# PEP 562 lazy loading — must be defined before __all__ so we can
# include the lazy module names without the static analyser flagging
# them as undefined.
_LAZY_MODULES = {
    "annotation": "langcore.annotation",
    "chunking": "langcore.chunking",
    "data": "langcore.core.data",
    "data_lib": "langcore.data_lib",
    "debug_utils": "langcore.core.debug_utils",
    "evaluation": "langcore.evaluation",
    "exceptions": "langcore.core.exceptions",
    "factory": "langcore.factory",
    "hooks": "langcore.hooks",
    "inference": "langcore.inference",
    "io": "langcore.io",
    "progress": "langcore.progress",
    "prompting": "langcore.prompting",
    "providers": "langcore.providers",
    "resolver": "langcore.resolver",
    "schema": "langcore.core.schema",
    "schema_adapter": "langcore.schema_adapter",
    "schema_generator": "langcore.schema_generator",
    "tokenizer": "langcore.core.tokenizer",
    "visualization": "langcore.visualization",
    "core": "langcore.core",
    "plugins": "langcore.plugins",
}

__all__ = [
    # Public convenience functions
    "async_extract",
    "configure",
    "evaluate",
    "extract",
    "get_config",
    "reset",
    "schema_from_example",
    "schema_from_examples",
    "schema_from_pydantic",
    "visualize",
    # Lazily-loaded submodules (resolved via __getattr__)
    *_LAZY_MODULES,
]

_CACHE: dict[str, Any] = {}


def extract(*args: Any, **kwargs: Any):
    """Top-level API: lx.extract(...)."""
    return extract_func(*args, **kwargs)


async def async_extract(*args: Any, **kwargs: Any):
    """Top-level async API: await lx.async_extract(...)."""
    return await async_extract_func(*args, **kwargs)


def evaluate(*args: Any, **kwargs: Any):
    """Top-level API: ``lx.evaluate(predictions, ground_truth, ...)``.

    Convenience wrapper around ``ExtractionMetrics.evaluate()``.
    Accepts the same arguments as ``ExtractionMetrics.evaluate()``
    plus an optional ``schema`` keyword for per-field breakdown.
    """
    from langcore.evaluation import ExtractionMetrics

    schema = kwargs.pop("schema", None)
    strict_attributes = kwargs.pop("strict_attributes", False)
    metrics = ExtractionMetrics(schema=schema, strict_attributes=strict_attributes)
    return metrics.evaluate(*args, **kwargs)


def visualize(*args: Any, **kwargs: Any):
    """Top-level API: lx.visualize(...)."""
    return visualization.visualize(*args, **kwargs)


def configure(*, hooks: Hooks | None = None) -> None:
    """Set global configuration applied to all ``extract`` / ``async_extract`` calls.

    Global hooks fire **before** any per-call hooks passed directly to
    ``extract(hooks=...)``.  Calling ``configure`` again replaces the
    previous global hooks entirely.

    Example::

        import langcore as lx
        from langcore.hooks import Hooks

        hooks = Hooks()
        hooks.on("extraction:start", lambda cfg: print("Starting:", cfg))
        lx.configure(hooks=hooks)

        # Every extract() call now emits to the global hooks.
        result = lx.extract(text, examples=examples)

    Args:
        hooks: Optional ``Hooks`` instance to apply globally.
    """
    _config_mod.set_hooks(hooks)


def get_config() -> dict[str, Any]:
    """Return a snapshot of the current global configuration.

    Returns:
        A dictionary with the current global settings.  Currently
        contains ``hooks`` (``Hooks | None``).
    """
    return {"hooks": _config_mod.get_hooks()}


def reset() -> None:
    """Reset all global configuration to defaults.

    Clears any hooks set via ``configure(hooks=...)``.
    """
    _config_mod.reset()


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
