"""Benchmark configuration settings and constants.

Centralized configuration for tokenization tests, model parameters,
display formatting, and test text sources.
"""

import enum
from dataclasses import dataclass
from pathlib import Path

# Result dictionary keys
RESULTS_KEY = "results"
EXTRACTION_KEY = "extraction"
EXTRACTION_RESULT_KEY = "extraction_result"
TOKENIZATION_KEY = "tokenization"


@dataclass(frozen=True)
class TokenizationConfig:
    """Settings for tokenization performance tests."""

    default_text_sizes: tuple[int, ...] = (100, 1000, 10000)  # Word counts
    benchmark_iterations: int = 10  # Iterations per size for averaging


@dataclass(frozen=True)
class ModelConfig:
    """Model and API configuration."""

    default_model: str = "gemini-2.5-flash"  # Cloud model default
    local_model: str = "gemma2:9b"  # Ollama model default
    default_temperature: float = 0.0  # Deterministic output
    default_max_workers: int = 10  # Parallel processing threads
    default_extraction_passes: int = 1  # Single pass extraction
    gemini_rate_limit_delay: float = 8.0  # Seconds between batches


class TextTypes(enum.StrEnum):
    """Supported languages for extraction testing."""

    ENGLISH = "english"
    JAPANESE = "japanese"
    FRENCH = "french"
    SPANISH = "spanish"


# Test texts from Project Gutenberg (similar genres for fair comparison)
GUTENBERG_TEXTS = {
    TextTypes.ENGLISH: (
        "https://www.gutenberg.org/files/11/11-0.txt"
    ),  # Alice's Adventures
    TextTypes.JAPANESE: ("https://www.gutenberg.org/files/1982/1982-0.txt"),  # Rashomon
    TextTypes.FRENCH: (
        "https://www.gutenberg.org/files/55456/55456-0.txt"
    ),  # Alice (French)
    TextTypes.SPANISH: (
        "https://www.gutenberg.org/files/67248/67248-0.txt"
    ),  # El clavo
}


@dataclass(frozen=True)
class DisplayConfig:
    """Display configuration."""

    separator_width: int = 50
    subseparator_width: int = 40
    figure_size_single: tuple[int, int] = (12, 5)
    figure_size_multi: tuple[int, int] = (14, 10)
    plot_style: str = "seaborn-v0_8-darkgrid"


@dataclass(frozen=True)
class PathConfig:
    """Path configuration."""

    results_dir: Path = Path("benchmark_results")

    def get_result_path(self, timestamp: str, suffix: str = "") -> Path:
        """Get result file path."""
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
        filename = f"benchmark{suffix}_{timestamp}"
        return self.results_dir / filename


# Global config instances
TOKENIZATION = TokenizationConfig()
MODELS = ModelConfig()
DISPLAY = DisplayConfig()
PATHS = PathConfig()
