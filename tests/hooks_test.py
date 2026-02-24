"""Tests for the langcore.hooks module."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from typing import Any
from unittest import mock

import pytest

from langcore.core import base_model, data, types
from langcore.hooks import HookName, Hooks

# =============================================================================
#  HookName enum tests
# =============================================================================


class TestHookName:
    """Tests for the HookName enum."""

    def test_enum_values(self) -> None:
        """Verify all expected hook names exist with correct values."""
        assert HookName.EXTRACTION_START.value == "extraction:start"
        assert HookName.EXTRACTION_CHUNK.value == "extraction:chunk"
        assert HookName.EXTRACTION_LLM_CALL.value == "extraction:llm_call"
        assert HookName.EXTRACTION_ALIGNMENT.value == "extraction:alignment"
        assert HookName.EXTRACTION_COMPLETE.value == "extraction:complete"
        assert HookName.EXTRACTION_ERROR.value == "extraction:error"

    def test_enum_count(self) -> None:
        """Ensure no events were accidentally removed."""
        assert len(HookName) == 8


# =============================================================================
#  Hooks.on() / registration tests
# =============================================================================


class TestHooksOn:
    """Tests for registering callbacks with Hooks.on()."""

    def test_register_single_callback(self) -> None:
        """A single callback can be registered."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on("extraction:start", cb)
        assert hooks.handler_count("extraction:start") == 1

    def test_register_multiple_callbacks_same_event(self) -> None:
        """Multiple callbacks on the same event are preserved."""
        hooks = Hooks()
        cb1, cb2 = mock.Mock(), mock.Mock()
        hooks.on("extraction:start", cb1)
        hooks.on("extraction:start", cb2)
        assert hooks.handler_count("extraction:start") == 2

    def test_register_with_enum_key(self) -> None:
        """HookName enum members work as event keys."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on(HookName.EXTRACTION_ERROR, cb)
        assert hooks.handler_count(HookName.EXTRACTION_ERROR) == 1
        assert hooks.handler_count("extraction:error") == 1

    def test_fluent_chaining(self) -> None:
        """on() returns self for fluent API."""
        hooks = Hooks()
        result = hooks.on("extraction:start", lambda _: None)
        assert result is hooks

    def test_register_on_different_events(self) -> None:
        """Callbacks on different events are tracked independently."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:error", mock.Mock())
        assert hooks.handler_count("extraction:start") == 1
        assert hooks.handler_count("extraction:error") == 1
        assert hooks.handler_count() == 2


# =============================================================================
#  Hooks.off() / deregistration tests
# =============================================================================


class TestHooksOff:
    """Tests for removing callbacks with Hooks.off()."""

    def test_remove_specific_callback(self) -> None:
        """A specific callback can be removed by reference."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on("extraction:start", cb)
        hooks.off("extraction:start", cb)
        assert hooks.handler_count("extraction:start") == 0

    def test_remove_all_callbacks_for_event(self) -> None:
        """All callbacks for an event are removed when no callback specified."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:start", mock.Mock())
        hooks.off("extraction:start")
        assert hooks.handler_count("extraction:start") == 0

    def test_remove_nonexistent_callback_is_noop(self) -> None:
        """Removing a callback that was never registered does not raise."""
        hooks = Hooks()
        hooks.off("extraction:start", mock.Mock())
        assert hooks.handler_count("extraction:start") == 0

    def test_remove_from_empty_event_is_noop(self) -> None:
        """Removing from an event with no handlers is a no-op."""
        hooks = Hooks()
        hooks.off("extraction:start")

    def test_fluent_chaining(self) -> None:
        """off() returns self for fluent API."""
        hooks = Hooks()
        result = hooks.off("extraction:start")
        assert result is hooks


# =============================================================================
#  Hooks.clear() tests
# =============================================================================


class TestHooksClear:
    """Tests for clearing all hooks."""

    def test_clear_removes_all_handlers(self) -> None:
        """clear() removes every registered callback."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:error", mock.Mock())
        hooks.clear()
        assert hooks.handler_count() == 0

    def test_clear_empty_hooks_is_noop(self) -> None:
        """clear() on empty hooks does not raise."""
        hooks = Hooks()
        hooks.clear()

    def test_fluent_chaining(self) -> None:
        """clear() returns self for fluent API."""
        hooks = Hooks()
        result = hooks.clear()
        assert result is hooks


# =============================================================================
#  Hooks.emit() tests
# =============================================================================


class TestHooksEmit:
    """Tests for emitting events."""

    def test_emit_calls_all_handlers(self) -> None:
        """All registered handlers for the event are invoked."""
        hooks = Hooks()
        cb1, cb2 = mock.Mock(), mock.Mock()
        hooks.on("extraction:start", cb1)
        hooks.on("extraction:start", cb2)
        hooks.emit("extraction:start", {"key": "value"})
        cb1.assert_called_once_with({"key": "value"})
        cb2.assert_called_once_with({"key": "value"})

    def test_emit_with_enum_key(self) -> None:
        """Emit works with HookName enum members."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on(HookName.EXTRACTION_COMPLETE, cb)
        hooks.emit(HookName.EXTRACTION_COMPLETE, "done")
        cb.assert_called_once_with("done")

    def test_emit_no_handlers_is_noop(self) -> None:
        """Emitting an event with no handlers does not raise."""
        hooks = Hooks()
        hooks.emit("extraction:start", {})

    def test_emit_with_none_payload(self) -> None:
        """Emit with no payload passes None to handlers."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on("extraction:start", cb)
        hooks.emit("extraction:start")
        cb.assert_called_once_with(None)

    def test_handler_order_preserved(self) -> None:
        """Handlers fire in the order they were registered."""
        order: list[int] = []
        hooks = Hooks()
        hooks.on("extraction:start", lambda _: order.append(1))
        hooks.on("extraction:start", lambda _: order.append(2))
        hooks.on("extraction:start", lambda _: order.append(3))
        hooks.emit("extraction:start", None)
        assert order == [1, 2, 3]

    def test_faulty_handler_does_not_break_pipeline(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An exception in a handler is logged and swallowed."""
        hooks = Hooks()
        cb_before = mock.Mock()
        cb_after = mock.Mock()

        def bad_handler(payload: Any) -> None:
            raise RuntimeError("hook exploded")

        hooks.on("extraction:start", cb_before)
        hooks.on("extraction:start", bad_handler)
        hooks.on("extraction:start", cb_after)

        with caplog.at_level(logging.ERROR, logger="langcore.hooks"):
            hooks.emit("extraction:start", {})

        cb_before.assert_called_once()
        cb_after.assert_called_once()
        assert "hook exploded" in caplog.text

    def test_emit_different_events_independent(self) -> None:
        """Emitting one event does not trigger handlers of another."""
        hooks = Hooks()
        cb_start = mock.Mock()
        cb_error = mock.Mock()
        hooks.on("extraction:start", cb_start)
        hooks.on("extraction:error", cb_error)
        hooks.emit("extraction:start", {})
        cb_start.assert_called_once()
        cb_error.assert_not_called()


# =============================================================================
#  Hooks introspection tests
# =============================================================================


class TestHooksIntrospection:
    """Tests for has_handlers() and handler_count()."""

    def test_has_handlers_empty(self) -> None:
        """Empty hooks report no handlers."""
        hooks = Hooks()
        assert not hooks.has_handlers()
        assert not hooks.has_handlers("extraction:start")

    def test_has_handlers_with_event(self) -> None:
        """has_handlers returns True for events with callbacks."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        assert hooks.has_handlers("extraction:start")
        assert hooks.has_handlers()
        assert not hooks.has_handlers("extraction:error")

    def test_handler_count_global(self) -> None:
        """handler_count(None) sums across all events."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:error", mock.Mock())
        hooks.on("extraction:error", mock.Mock())
        assert hooks.handler_count() == 3

    def test_handler_count_per_event(self) -> None:
        """handler_count(event) returns count for that event only."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:start", mock.Mock())
        assert hooks.handler_count("extraction:start") == 2
        assert hooks.handler_count("extraction:error") == 0


# =============================================================================
#  Hooks.__add__() composition tests
# =============================================================================


class TestHooksAdd:
    """Tests for combining hooks with the + operator."""

    def test_merge_two_hooks(self) -> None:
        """Two Hooks instances merge correctly."""
        h1 = Hooks()
        h2 = Hooks()
        h1.on("extraction:start", mock.Mock())
        h2.on("extraction:error", mock.Mock())
        merged = h1 + h2
        assert merged.handler_count("extraction:start") == 1
        assert merged.handler_count("extraction:error") == 1

    def test_merge_same_event(self) -> None:
        """Handlers for the same event are combined in order."""
        order: list[str] = []
        h1 = Hooks()
        h2 = Hooks()
        h1.on("extraction:start", lambda _: order.append("h1"))
        h2.on("extraction:start", lambda _: order.append("h2"))
        merged = h1 + h2
        merged.emit("extraction:start", None)
        assert order == ["h1", "h2"]

    def test_merge_does_not_mutate_originals(self) -> None:
        """Merging creates a new Hooks; originals are unchanged."""
        h1 = Hooks()
        h2 = Hooks()
        h1.on("extraction:start", mock.Mock())
        merged = h1 + h2
        merged.on("extraction:error", mock.Mock())
        assert h1.handler_count() == 1
        assert h2.handler_count() == 0

    def test_merge_with_non_hooks_returns_not_implemented(self) -> None:
        """Adding a non-Hooks object returns NotImplemented."""
        h = Hooks()
        assert h.__add__("not a hooks") is NotImplemented


# =============================================================================
#  Hooks.__repr__() tests
# =============================================================================


class TestHooksRepr:
    """Tests for the string representation."""

    def test_repr_empty(self) -> None:
        """Empty hooks show empty dict."""
        assert repr(Hooks()) == "Hooks({})"

    def test_repr_with_handlers(self) -> None:
        """Repr shows event counts."""
        hooks = Hooks()
        hooks.on("extraction:start", mock.Mock())
        hooks.on("extraction:start", mock.Mock())
        assert repr(hooks) == "Hooks({'extraction:start': 2})"


# =============================================================================
#  Custom string event tests
# =============================================================================


class TestCustomEvents:
    """Tests for user-defined (non-enum) event strings."""

    def test_custom_event_string(self) -> None:
        """Users can register and emit custom event names."""
        hooks = Hooks()
        cb = mock.Mock()
        hooks.on("my:custom:event", cb)
        hooks.emit("my:custom:event", {"data": 42})
        cb.assert_called_once_with({"data": 42})


# =============================================================================
#  Integration: hooks wired through Annotator
# =============================================================================


class _DummyModel(base_model.BaseLanguageModel):
    """Minimal mock model for integration tests."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        super().__init__()
        self._outputs = outputs or ['{"extractions": []}']

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[types.ScoredOutput]]:
        """Return canned outputs for each prompt."""
        for prompt in batch_prompts:
            idx = min(batch_prompts.index(prompt), len(self._outputs) - 1)
            yield [types.ScoredOutput(score=1.0, output=self._outputs[idx])]


class TestAnnotatorHooks:
    """Test that Annotator emits hook events during annotation."""

    def test_llm_call_hook_fires(self) -> None:
        """extraction:llm_call hook fires with prompts and outputs."""
        from langcore import annotation, prompting, resolver
        from langcore.core import format_handler as fh

        hooks = Hooks()
        received: list[dict] = []
        hooks.on(HookName.EXTRACTION_LLM_CALL, lambda p: received.append(p))

        model = _DummyModel()
        tmpl = prompting.PromptTemplateStructured(description="test")
        tmpl.examples.append(
            data.ExampleData(
                text="hello world",
                extractions=[
                    data.Extraction("Greeting", "hello"),
                ],
            )
        )
        handler = fh.FormatHandler()
        annotator = annotation.Annotator(
            language_model=model,
            prompt_template=tmpl,
            format_handler=handler,
            hooks=hooks,
        )
        res = resolver.Resolver(format_handler=handler)
        annotator.annotate_text(
            text="hello world",
            resolver=res,
            max_char_buffer=1000,
            show_progress=False,
        )
        assert len(received) >= 1
        assert "prompts" in received[0]
        assert "outputs" in received[0]

    def test_chunk_hook_fires(self) -> None:
        """extraction:chunk hook fires with chunk info."""
        from langcore import annotation, prompting, resolver
        from langcore.core import format_handler as fh

        hooks = Hooks()
        chunks: list[dict] = []
        hooks.on(HookName.EXTRACTION_CHUNK, lambda p: chunks.append(p))

        model = _DummyModel()
        tmpl = prompting.PromptTemplateStructured(description="test")
        tmpl.examples.append(
            data.ExampleData(
                text="hello world",
                extractions=[
                    data.Extraction("Greeting", "hello"),
                ],
            )
        )
        handler = fh.FormatHandler()
        annotator = annotation.Annotator(
            language_model=model,
            prompt_template=tmpl,
            format_handler=handler,
            hooks=hooks,
        )
        res = resolver.Resolver(format_handler=handler)
        annotator.annotate_text(
            text="hello world",
            resolver=res,
            max_char_buffer=1000,
            show_progress=False,
        )
        assert len(chunks) >= 1
        assert "chunk_text" in chunks[0]
        assert "document_id" in chunks[0]
        assert "extractions" in chunks[0]

    def test_alignment_hook_fires(self) -> None:
        """extraction:alignment hook fires with alignment info."""
        from langcore import annotation, prompting, resolver
        from langcore.core import format_handler as fh

        hooks = Hooks()
        alignments: list[dict] = []
        hooks.on(
            HookName.EXTRACTION_ALIGNMENT,
            lambda p: alignments.append(p),
        )

        model = _DummyModel()
        tmpl = prompting.PromptTemplateStructured(description="test")
        tmpl.examples.append(
            data.ExampleData(
                text="hello world",
                extractions=[
                    data.Extraction("Greeting", "hello"),
                ],
            )
        )
        handler = fh.FormatHandler()
        annotator = annotation.Annotator(
            language_model=model,
            prompt_template=tmpl,
            format_handler=handler,
            hooks=hooks,
        )
        res = resolver.Resolver(format_handler=handler)
        annotator.annotate_text(
            text="hello world",
            resolver=res,
            max_char_buffer=1000,
            show_progress=False,
        )
        assert len(alignments) >= 1
        assert "extractions" in alignments[0]
        assert "chunk_text" in alignments[0]

    def test_no_hooks_does_not_raise(self) -> None:
        """Annotator with hooks=None still works."""
        from langcore import annotation, prompting, resolver
        from langcore.core import format_handler as fh

        model = _DummyModel()
        tmpl = prompting.PromptTemplateStructured(description="test")
        tmpl.examples.append(
            data.ExampleData(
                text="hello world",
                extractions=[
                    data.Extraction("Greeting", "hello"),
                ],
            )
        )
        handler = fh.FormatHandler()
        annotator = annotation.Annotator(
            language_model=model,
            prompt_template=tmpl,
            format_handler=handler,
            hooks=None,
        )
        res = resolver.Resolver(format_handler=handler)
        # Should not raise
        annotator.annotate_text(
            text="hello world",
            resolver=res,
            max_char_buffer=1000,
            show_progress=False,
        )


# =============================================================================
#  Integration: hooks wired through extract() API
# =============================================================================


class TestExtractHooks:
    """Test that extract() emits start/complete/error hooks."""

    def test_start_and_complete_hooks_fire(self) -> None:
        """extraction:start and extraction:complete fire during extract()."""
        from langcore import extraction

        hooks = Hooks()
        events: list[str] = []
        hooks.on(HookName.EXTRACTION_START, lambda _: events.append("start"))
        hooks.on(
            HookName.EXTRACTION_COMPLETE,
            lambda _: events.append("complete"),
        )

        model = _DummyModel()
        tmpl_examples = [
            data.ExampleData(
                text="hello world",
                extractions=[data.Extraction("Greeting", "hello")],
            )
        ]

        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract greetings",
            examples=tmpl_examples,
            model=model,
            show_progress=False,
            prompt_validation_level="off",
            hooks=hooks,
        )
        assert "start" in events
        assert "complete" in events

    def test_error_hook_fires_on_failure(self) -> None:
        """extraction:error fires when extraction raises."""
        from langcore import extraction

        hooks = Hooks()
        errors: list[Exception] = []
        hooks.on(HookName.EXTRACTION_ERROR, lambda err: errors.append(err))

        # No examples and no schema → ValueError
        with pytest.raises(ValueError):
            extraction.extract(
                text_or_documents="hello",
                prompt_description="extract",
                examples=None,
                hooks=hooks,
            )
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

    def test_start_hook_payload(self) -> None:
        """extraction:start payload has expected config fields."""
        from langcore import extraction

        hooks = Hooks()
        payloads: list[dict] = []
        hooks.on(HookName.EXTRACTION_START, lambda p: payloads.append(p))

        model = _DummyModel()
        examples = [
            data.ExampleData(
                text="hello world",
                extractions=[data.Extraction("Greeting", "hello")],
            )
        ]

        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract",
            examples=examples,
            model=model,
            show_progress=False,
            prompt_validation_level="off",
            hooks=hooks,
        )
        assert len(payloads) == 1
        p = payloads[0]
        assert "model_id" in p
        assert "batch_length" in p
        assert "extraction_passes" in p
        assert "text_length" in p
        assert p["text_length"] == 11  # len("hello world")

    def test_hooks_none_is_safe(self) -> None:
        """extract() with hooks=None works without errors."""
        from langcore import extraction

        model = _DummyModel()
        examples = [
            data.ExampleData(
                text="hello world",
                extractions=[data.Extraction("Greeting", "hello")],
            )
        ]

        # Should not raise
        extraction.extract(
            text_or_documents="hello world",
            prompt_description="extract",
            examples=examples,
            model=model,
            show_progress=False,
            prompt_validation_level="off",
            hooks=None,
        )


# =============================================================================
#  Async hooks integration tests
# =============================================================================


class TestAsyncExtractHooks:
    """Test that async_extract() emits hooks."""

    def test_start_and_complete_async(self) -> None:
        """extraction:start and extraction:complete fire in async_extract()."""
        from langcore import extraction

        hooks = Hooks()
        events: list[str] = []
        hooks.on(HookName.EXTRACTION_START, lambda _: events.append("start"))
        hooks.on(
            HookName.EXTRACTION_COMPLETE,
            lambda _: events.append("complete"),
        )

        model = _DummyModel()
        examples = [
            data.ExampleData(
                text="hello world",
                extractions=[data.Extraction("Greeting", "hello")],
            )
        ]

        asyncio.run(
            extraction.async_extract(
                text_or_documents="hello world",
                prompt_description="extract",
                examples=examples,
                model=model,
                show_progress=False,
                prompt_validation_level="off",
                hooks=hooks,
            )
        )
        assert "start" in events
        assert "complete" in events

    def test_error_async(self) -> None:
        """extraction:error fires in async_extract() on failure."""
        from langcore import extraction

        hooks = Hooks()
        errors: list[Exception] = []
        hooks.on(HookName.EXTRACTION_ERROR, lambda e: errors.append(e))

        with pytest.raises(ValueError):
            asyncio.run(
                extraction.async_extract(
                    text_or_documents="hello",
                    prompt_description="extract",
                    examples=None,
                    hooks=hooks,
                )
            )
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
