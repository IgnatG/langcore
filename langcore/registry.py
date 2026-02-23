"""Compatibility shim for langcore.registry imports.

This module redirects to langcore.plugins for backward compatibility.
Will be removed in v2.0.0.
"""

from __future__ import annotations

import warnings

from langcore import plugins


def __getattr__(name: str):
    """Redirect to plugins module with deprecation warning."""
    warnings.warn(
        "`langcore.registry` is deprecated and will be removed in v2.0.0; "
        "use `langcore.plugins` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return getattr(plugins, name)
