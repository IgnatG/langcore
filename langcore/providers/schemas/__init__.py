"""Provider-specific schema implementations."""

from __future__ import annotations

from langcore.providers.schemas import gemini

GeminiSchema = gemini.GeminiSchema  # Backward compat

__all__ = ["GeminiSchema"]
