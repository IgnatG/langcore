"""Extraction hooks and event system for LangCore.

Provides a lightweight event system that lets users inject custom logic
(logging, monitoring, error handling, etc.) at every stage of the
extraction pipeline.

Usage example::

    from langcore.hooks import Hooks

    hooks = Hooks()
    hooks.on("extraction:start", lambda config: print("Starting extraction"))
    hooks.on("extraction:error", lambda err: alert_team(err))

    result = lx.extract(text, examples=examples, hooks=hooks)
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import logging
from collections.abc import Callable
from typing import Any

__all__ = [
    "HookName",
    "Hooks",
]

logger = logging.getLogger(__name__)


class HookName(enum.Enum):
    """Enumeration of pipeline events that hooks can listen to.

    Each value corresponds to a specific stage in the extraction
    pipeline where user callbacks can be injected.

    Attributes:
        EXTRACTION_START: Fired before extraction begins. Receives a
            dict with extraction configuration (model_id, text length,
            batch_length, extraction_passes, etc.).
        EXTRACTION_CHUNK: Fired after each chunk is processed. Receives
            a dict with ``chunk_text``, ``document_id``, and the list
            of ``extractions`` produced for that chunk.
        EXTRACTION_LLM_CALL: Fired around each LLM inference call.
            Receives a dict with ``prompts`` (list of str) before the
            call, and ``outputs`` (list of ScoredOutput sequences)
            after.
        EXTRACTION_ALIGNMENT: Fired after alignment of extractions to
            source text. Receives a dict with the ``extractions`` list
            and their ``alignment_status`` values.
        EXTRACTION_COMPLETE: Fired after the full extraction finishes.
            Receives the ``AnnotatedDocument`` (or list thereof).
        EXTRACTION_ERROR: Fired on any error during extraction.
            Receives the ``Exception`` instance.
    """

    EXTRACTION_START = "extraction:start"
    EXTRACTION_CHUNK = "extraction:chunk"
    EXTRACTION_LLM_CALL = "extraction:llm_call"
    EXTRACTION_ALIGNMENT = "extraction:alignment"
    EXTRACTION_COMPLETE = "extraction:complete"
    EXTRACTION_ERROR = "extraction:error"


# Allow both enum values and raw strings as hook names.
_HookKey = HookName | str


def _normalise_key(key: _HookKey) -> str:
    """Normalise a hook key to its string value.

    Parameters:
        key: A ``HookName`` enum member or a raw string.

    Returns:
        The string value of the hook name.
    """
    if isinstance(key, HookName):
        return key.value
    return key


class Hooks:
    """Container for extraction lifecycle callbacks.

    Hooks are organised by event name.  Multiple callbacks can be
    registered for the same event and they execute in registration
    order.  Callbacks receive a single payload argument whose type
    depends on the event (see ``HookName`` for details).

    Hooks instances can be combined with the ``+`` operator to create
    a merged set containing all callbacks from both operands.

    Example::

        hooks = Hooks()
        hooks.on("extraction:start", lambda cfg: print(cfg))
        hooks.on("extraction:error", lambda err: log_error(err))
    """

    def __init__(self) -> None:
        """Initialise an empty hooks container."""
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    # ── Registration ──────────────────────────────────────────────

    def on(
        self,
        event: _HookKey,
        callback: Callable[..., Any],
    ) -> Hooks:
        """Register a callback for an event.

        Parameters:
            event: The hook event name (``HookName`` enum or string).
            callback: A callable invoked when the event fires.
                Receives a single payload argument.

        Returns:
            ``self`` for fluent chaining.
        """
        key = _normalise_key(event)
        self._handlers.setdefault(key, []).append(callback)
        return self

    def off(
        self,
        event: _HookKey,
        callback: Callable[..., Any] | None = None,
    ) -> Hooks:
        """Remove a callback (or all callbacks) for an event.

        Parameters:
            event: The hook event name.
            callback: The specific callback to remove.  If ``None``,
                removes **all** callbacks for this event.

        Returns:
            ``self`` for fluent chaining.
        """
        key = _normalise_key(event)
        if callback is None:
            self._handlers.pop(key, None)
        elif key in self._handlers:
            try:
                self._handlers[key].remove(callback)
            except ValueError:
                pass  # callback was not registered
            if not self._handlers[key]:
                del self._handlers[key]
        return self

    def clear(self) -> Hooks:
        """Remove all registered callbacks across every event.

        Returns:
            ``self`` for fluent chaining.
        """
        self._handlers.clear()
        return self

    # ── Emission ──────────────────────────────────────────────────

    def emit(self, event: _HookKey, payload: Any = None) -> None:
        """Fire an event, calling every registered callback.

        Callbacks are invoked synchronously in registration order.
        Exceptions from individual callbacks are logged and
        **swallowed** so that a faulty hook never breaks the pipeline.

        .. note::

            If an ``async def`` callback is registered, its return
            value (a coroutine) will be silently discarded.  Use
            :meth:`async_emit` when running inside an async context.

        Parameters:
            event: The hook event name.
            payload: Data passed to each callback.
        """
        key = _normalise_key(event)
        handlers = self._handlers.get(key)
        if not handlers:
            return
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                logger.exception(
                    "Hook callback %r for event %r raised an exception",
                    handler,
                    key,
                )

    async def async_emit(self, event: _HookKey, payload: Any = None) -> None:
        """Fire an event, awaiting async callbacks when appropriate.

        Callbacks are invoked in registration order.  If a callback
        is an ``async def`` (coroutine function), it is awaited.
        Sync callbacks are called normally.  Exceptions from
        individual callbacks are logged and **swallowed**.

        Parameters:
            event: The hook event name.
            payload: Data passed to each callback.
        """
        key = _normalise_key(event)
        handlers = self._handlers.get(key)
        if not handlers:
            return
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    result = handler(payload)
                    # Handle the case where a regular function
                    # returns a coroutine (wrapped async).
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                logger.exception(
                    "Hook callback %r for event %r raised an exception",
                    handler,
                    key,
                )

    # ── Introspection ─────────────────────────────────────────────

    def has_handlers(self, event: _HookKey | None = None) -> bool:
        """Check whether any callbacks are registered.

        Parameters:
            event: If given, check only for this event.  If ``None``,
                check across all events.

        Returns:
            ``True`` if at least one callback is registered.
        """
        if event is None:
            return any(self._handlers.values())
        key = _normalise_key(event)
        return bool(self._handlers.get(key))

    def handler_count(self, event: _HookKey | None = None) -> int:
        """Return the number of registered callbacks.

        Parameters:
            event: If given, count only for this event.  If ``None``,
                count across all events.

        Returns:
            Number of registered callbacks.
        """
        if event is None:
            return sum(len(h) for h in self._handlers.values())
        key = _normalise_key(event)
        return len(self._handlers.get(key, []))

    # ── Composition ───────────────────────────────────────────────

    def __add__(self, other: Hooks) -> Hooks:
        """Combine two ``Hooks`` instances into a new one.

        The returned ``Hooks`` contains all callbacks from ``self``
        followed by all callbacks from ``other``, preserving
        registration order within each operand.

        Parameters:
            other: Another ``Hooks`` instance to merge with.

        Returns:
            A new ``Hooks`` instance with the union of all callbacks.

        Raises:
            TypeError: If ``other`` is not a ``Hooks`` instance.
        """
        if not isinstance(other, Hooks):
            return NotImplemented
        merged = Hooks()
        for key, handlers in self._handlers.items():
            merged._handlers.setdefault(key, []).extend(handlers)
        for key, handlers in other._handlers.items():
            merged._handlers.setdefault(key, []).extend(handlers)
        return merged

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        events = {k: len(v) for k, v in self._handlers.items() if v}
        return f"Hooks({events})"
