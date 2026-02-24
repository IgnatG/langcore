"""Global configuration state for LangCore.

Stores settings applied by ``lx.configure()`` so they take effect
across all subsequent ``extract`` / ``async_extract`` calls without
the user having to pass them on every invocation.

This module is intentionally internal (``_config``).  The public API
lives in ``langcore.__init__`` (``lx.configure``, ``lx.get_config``,
``lx.reset``).
"""

from __future__ import annotations

import threading

from langcore.hooks import Hooks

__all__: list[str] = []

_lock = threading.Lock()

# ── Internal state ────────────────────────────────────────────────
_global_hooks: Hooks | None = None


def set_hooks(hooks: Hooks | None) -> None:
    """Store the global hooks instance (thread-safe).

    Parameters:
        hooks: A ``Hooks`` instance to apply globally, or ``None``
            to clear global hooks.
    """
    global _global_hooks
    with _lock:
        _global_hooks = hooks


def get_hooks() -> Hooks | None:
    """Return the currently configured global hooks, if any."""
    return _global_hooks


def resolve_hooks(per_call: Hooks | None) -> Hooks:
    """Merge global and per-call hooks into a single ``Hooks`` instance.

    Merging order: global hooks fire first, then per-call hooks.

    Parameters:
        per_call: Hooks passed directly to ``extract()`` /
            ``async_extract()``.  ``None`` means the caller did not
            provide any.

    Returns:
        A ``Hooks`` instance ready for use in the extraction pipeline.
    """
    g = _global_hooks
    if g is not None and per_call is not None:
        return g + per_call
    if g is not None:
        return g
    if per_call is not None:
        return per_call
    return Hooks()


def reset() -> None:
    """Clear all global configuration back to defaults."""
    global _global_hooks
    with _lock:
        _global_hooks = None
