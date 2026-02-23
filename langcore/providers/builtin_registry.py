"""Built-in provider registration configuration.

This module defines the registration details for all built-in providers,
using patterns from the centralized patterns module.
"""

from typing import TypedDict

from langcore.providers import patterns


class ProviderConfig(TypedDict):
    """Configuration for a provider registration."""

    patterns: tuple[str, ...]
    target: str
    priority: int


# Built-in provider configurations using centralized patterns
BUILTIN_PROVIDERS: list[ProviderConfig] = [
    {
        "patterns": patterns.GEMINI_PATTERNS,
        "target": "langcore.providers.gemini:GeminiLanguageModel",
        "priority": patterns.GEMINI_PRIORITY,
    },
    {
        "patterns": patterns.OLLAMA_PATTERNS,
        "target": "langcore.providers.ollama:OllamaLanguageModel",
        "priority": patterns.OLLAMA_PRIORITY,
    },
    {
        "patterns": patterns.OPENAI_PATTERNS,
        "target": "langcore.providers.openai:OpenAILanguageModel",
        "priority": patterns.OPENAI_PRIORITY,
    },
]
