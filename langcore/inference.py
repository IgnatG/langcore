"""Re-export core base_model types for convenience.

New code should import from ``langcore.core.base_model`` directly.
"""

from __future__ import annotations

from langcore.core.base_model import BaseLanguageModel

__all__ = [
    "BaseLanguageModel",
]
