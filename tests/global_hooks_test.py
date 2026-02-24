"""Tests for lx.configure(hooks=...) global hooks API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

import pytest

import langcore as lx
from langcore import _config, extraction
from langcore.core import base_model, data, types
from langcore.hooks import HookName, Hooks

# ── Helpers ──────────────────────────────────────────────────────


class _DummyModel(base_model.BaseLanguageModel):
    """Minimal model that returns empty extractions."""

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[types.ScoredOutput]]:
        for _ in batch_prompts:
            yield [types.ScoredOutput(score=1.0, output='{"extractions": []}')]


_EXAMPLES = [
    data.ExampleData(
        text="hello world",
        extractions=[data.Extraction("Greeting", "hello")],
    )
]


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Ensure global config is clean before and after every test."""
    lx.reset()
    yield
    lx.reset()


# =============================================================================
#  lx.configure / lx.get_config / lx.reset
# =============================================================================


class TestConfigureAPI:
    """Tests for the public configure / get_config / reset surface."""

    def test_configure_stores_hooks(self) -> None:
        hooks = Hooks()
        lx.configure(hooks=hooks)
        assert lx.get_config()["hooks"] is hooks

    def test_configure_replaces_previous_hooks(self) -> None:
        hooks1 = Hooks()
        hooks2 = Hooks()
        lx.configure(hooks=hooks1)
        lx.configure(hooks=hooks2)
        assert lx.get_config()["hooks"] is hooks2

    def test_configure_none_clears_hooks(self) -> None:
        lx.configure(hooks=Hooks())
        lx.configure(hooks=None)
        assert lx.get_config()["hooks"] is None

    def test_reset_clears_hooks(self) -> None:
        lx.configure(hooks=Hooks())
        lx.reset()
        assert lx.get_config()["hooks"] is None

    def test_get_config_returns_dict(self) -> None:
        cfg = lx.get_config()
        assert isinstance(cfg, dict)
        assert "hooks" in cfg

    def test_initial_config_has_no_hooks(self) -> None:
        assert lx.get_config()["hooks"] is None


# =============================================================================
#  _config.resolve_hooks
# =============================================================================


class TestResolveHooks:
    """Tests for the internal resolve_hooks merging logic."""

    def test_no_global_no_per_call(self) -> None:
        result = _config.resolve_hooks(None)
        assert isinstance(result, Hooks)
        assert result.handler_count() == 0

    def test_global_only(self) -> None:
        g = Hooks()
        cb = mock.Mock()
        g.on("extraction:start", cb)
        _config.set_hooks(g)

        result = _config.resolve_hooks(None)
        assert result.handler_count("extraction:start") == 1

    def test_per_call_only(self) -> None:
        pc = Hooks()
        cb = mock.Mock()
        pc.on("extraction:start", cb)

        result = _config.resolve_hooks(pc)
        assert result.handler_count("extraction:start") == 1

    def test_global_plus_per_call_merged(self) -> None:
        g = Hooks()
        g.on("extraction:start", mock.Mock())
        _config.set_hooks(g)

        pc = Hooks()
        pc.on("extraction:start", mock.Mock())

        result = _config.resolve_hooks(pc)
        # Both callbacks present (global first, then per-call)
        assert result.handler_count("extraction:start") == 2

    def test_global_fires_before_per_call(self) -> None:
        order: list[str] = []

        g = Hooks()
        g.on("extraction:start", lambda _: order.append("global"))
        _config.set_hooks(g)

        pc = Hooks()
        pc.on("extraction:start", lambda _: order.append("per_call"))

        merged = _config.resolve_hooks(pc)
        merged.emit("extraction:start", {})

        assert order == ["global", "per_call"]

    def test_different_events_both_preserved(self) -> None:
        g = Hooks()
        g.on("extraction:start", mock.Mock())
        _config.set_hooks(g)

        pc = Hooks()
        pc.on("extraction:error", mock.Mock())

        result = _config.resolve_hooks(pc)
        assert result.handler_count("extraction:start") == 1
        assert result.handler_count("extraction:error") == 1
        assert result.handler_count() == 2


# =============================================================================
#  Integration: global hooks in extract()
# =============================================================================


class TestGlobalHooksExtract:
    """Test that global hooks fire in extract()."""

    def test_global_hooks_fire_on_extract(self) -> None:
        events: list[str] = []
        hooks = Hooks()
        hooks.on(HookName.EXTRACTION_START, lambda _: events.append("start"))
        hooks.on(HookName.EXTRACTION_COMPLETE, lambda _: events.append("complete"))
        lx.configure(hooks=hooks)

        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract greetings",
            examples=_EXAMPLES,
            model=_DummyModel(),
            show_progress=False,
            prompt_validation_level="off",
        )

        assert "start" in events
        assert "complete" in events

    def test_global_and_per_call_both_fire(self) -> None:
        order: list[str] = []

        global_hooks = Hooks()
        global_hooks.on(HookName.EXTRACTION_START, lambda _: order.append("global"))
        lx.configure(hooks=global_hooks)

        per_call = Hooks()
        per_call.on(HookName.EXTRACTION_START, lambda _: order.append("per_call"))

        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract greetings",
            examples=_EXAMPLES,
            model=_DummyModel(),
            show_progress=False,
            prompt_validation_level="off",
            hooks=per_call,
        )

        assert order == ["global", "per_call"]

    def test_per_call_hooks_only_when_no_global(self) -> None:
        events: list[str] = []
        per_call = Hooks()
        per_call.on(HookName.EXTRACTION_START, lambda _: events.append("start"))

        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract greetings",
            examples=_EXAMPLES,
            model=_DummyModel(),
            show_progress=False,
            prompt_validation_level="off",
            hooks=per_call,
        )

        assert events == ["start"]

    def test_global_error_hook_fires(self) -> None:
        errors: list[Exception] = []
        hooks = Hooks()
        hooks.on(HookName.EXTRACTION_ERROR, lambda e: errors.append(e))
        lx.configure(hooks=hooks)

        with pytest.raises(ValueError):
            extraction.extract(
                text_or_documents="hello",
                prompt_description="extract",
                examples=None,
            )

        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)


# =============================================================================
#  Integration: global hooks in async_extract()
# =============================================================================


class TestGlobalHooksAsyncExtract:
    """Test that global hooks fire in async_extract()."""

    def test_global_hooks_fire_on_async_extract(self) -> None:
        events: list[str] = []
        hooks = Hooks()
        hooks.on(HookName.EXTRACTION_START, lambda _: events.append("start"))
        hooks.on(HookName.EXTRACTION_COMPLETE, lambda _: events.append("complete"))
        lx.configure(hooks=hooks)

        asyncio.run(
            extraction.async_extract(
                text_or_documents="hello world",
                prompt_description="extract greetings",
                examples=_EXAMPLES,
                model=_DummyModel(),
                show_progress=False,
                prompt_validation_level="off",
            )
        )

        assert "start" in events
        assert "complete" in events

    def test_global_and_per_call_order_async(self) -> None:
        order: list[str] = []

        global_hooks = Hooks()
        global_hooks.on(HookName.EXTRACTION_START, lambda _: order.append("global"))
        lx.configure(hooks=global_hooks)

        per_call = Hooks()
        per_call.on(HookName.EXTRACTION_START, lambda _: order.append("per_call"))

        asyncio.run(
            extraction.async_extract(
                text_or_documents="hello world",
                prompt_description="extract greetings",
                examples=_EXAMPLES,
                model=_DummyModel(),
                show_progress=False,
                prompt_validation_level="off",
                hooks=per_call,
            )
        )

        assert order == ["global", "per_call"]

    def test_global_error_hook_fires_async(self) -> None:
        errors: list[Exception] = []
        hooks = Hooks()
        hooks.on(HookName.EXTRACTION_ERROR, lambda e: errors.append(e))
        lx.configure(hooks=hooks)

        with pytest.raises(ValueError):
            asyncio.run(
                extraction.async_extract(
                    text_or_documents="hello",
                    prompt_description="extract",
                    examples=None,
                )
            )

        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
