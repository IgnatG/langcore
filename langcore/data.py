"""Compatibility shim for langcore.data imports.

This module provides backward compatibility for code that imports from
langcore.data. All functionality has moved to langcore.core.data.
"""

from __future__ import annotations

# Re-export everything from core.data for backward compatibility
# pylint: disable=unused-wildcard-import
from langcore.core.data import *
