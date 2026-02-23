"""Compatibility shim for langcore.tokenizer imports.

This module provides backward compatibility for code that imports from
langcore.tokenizer. All functionality has moved to langcore.core.tokenizer.
"""

from __future__ import annotations

# Re-export everything from core.tokenizer for backward compatibility
# pylint: disable=unused-wildcard-import
from langcore.core.tokenizer import *
