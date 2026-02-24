"""Main extraction API for LangCore."""

from __future__ import annotations

import typing
import warnings
from collections.abc import Iterable
from typing import cast

import pydantic

from langcore import (
    _config,
    annotation,
    factory,
    io,
    prompting,
    resolver,
    schema_adapter,
)
from langcore import hooks as hooks_lib
from langcore import prompt_validation as pv
from langcore.core import base_model, data
from langcore.core import format_handler as fh
from langcore.core import tokenizer as tokenizer_lib


def _build_extraction_components(
    text_or_documents: typing.Any,
    prompt_description: str | None,
    examples: typing.Sequence[typing.Any] | None,
    model_id: str,
    api_key: str | None,
    format_type: typing.Any,
    max_char_buffer: int,
    temperature: float | None,
    fence_output: bool | None,
    use_schema_constraints: bool,
    batch_length: int,
    max_workers: int,
    additional_context: str | None,
    resolver_params: dict | None,
    language_model_params: dict | None,
    debug: bool,
    model_url: str | None,
    extraction_passes: int,
    context_window_chars: int | None,
    config: typing.Any,
    model: typing.Any,
    fetch_urls: bool,
    prompt_validation_level: pv.PromptValidationLevel,
    prompt_validation_strict: bool,
    show_progress: bool,
    tokenizer: tokenizer_lib.Tokenizer | None,
    schema: type[pydantic.BaseModel] | None = None,
    hooks: hooks_lib.Hooks | None = None,
) -> tuple:
    """Shared setup for ``extract`` and ``async_extract``.

    Validates inputs, resolves the language model, and builds the annotator,
    resolver, and alignment kwargs.  Returns a tuple of:

      ``(text_or_documents, annotator, res, alignment_kwargs, format_handler)``

    so that callers only need to dispatch sync vs. async annotation.
    """
    # When a Pydantic schema is provided, derive prompt_description
    # and examples from the model class if they were not explicitly
    # supplied by the caller.
    if schema is not None:
        adapter = schema_adapter.PydanticSchemaAdapter(schema)
        if prompt_description is None:
            prompt_description = adapter.generate_prompt_description()
        if not examples:
            # With schema-only usage (no examples), skip the examples
            # requirement — the prompt description carries the schema.
            # We still need at least a placeholder ExampleData so the
            # prompt template can build, but we can skip validation.
            prompt_validation_level = pv.PromptValidationLevel.OFF

    if not examples and schema is None:
        raise ValueError(
            "Examples are required for reliable extraction. Please provide at"
            " least one ExampleData object with sample extractions, or pass a"
            " Pydantic BaseModel class via the 'schema' parameter."
        )

    if prompt_validation_level is not pv.PromptValidationLevel.OFF:
        report = pv.validate_prompt_alignment(
            examples=examples,
            aligner=resolver.WordAligner(),
            policy=pv.AlignmentPolicy(),
            tokenizer=tokenizer,
        )
        pv.handle_alignment_report(
            report,
            level=prompt_validation_level,
            strict_non_exact=prompt_validation_strict,
        )

    if debug:
        from langcore.core import debug_utils

        debug_utils.configure_debug_logging()

    if format_type is None:
        format_type = data.FormatType.JSON

    if max_workers is not None and batch_length < max_workers:
        warnings.warn(
            f"batch_length ({batch_length}) < max_workers ({max_workers}). "
            f"Only {batch_length} workers will be used. "
            "Set batch_length >= max_workers for optimal parallelization.",
            UserWarning,
            stacklevel=2,
        )

    if (
        fetch_urls
        and isinstance(text_or_documents, str)
        and io.is_url(text_or_documents)
    ):
        text_or_documents = io.download_text_from_url(text_or_documents)

    prompt_template = prompting.PromptTemplateStructured(description=prompt_description)
    prompt_template.examples.extend(examples)

    language_model: base_model.BaseLanguageModel | None = None

    if model:
        language_model = model
        if fence_output is not None:
            language_model.set_fence_output(fence_output)
        if use_schema_constraints:
            warnings.warn(
                "'use_schema_constraints' is ignored when 'model' is provided. "
                "The model should already be configured with schema constraints.",
                UserWarning,
                stacklevel=3,
            )
    elif config:
        if use_schema_constraints:
            warnings.warn(
                "With 'config', schema constraints are still applied via examples. "
                "Or pass explicit schema in config.provider_kwargs.",
                UserWarning,
                stacklevel=3,
            )
        language_model = factory.create_model(
            config=config,
            examples=prompt_template.examples if use_schema_constraints else None,
            use_schema_constraints=use_schema_constraints,
            fence_output=fence_output,
        )
    else:
        base_lm_kwargs: dict[str, typing.Any] = {
            "api_key": api_key,
            "format_type": format_type,
            "temperature": temperature,
            "model_url": model_url,
            "base_url": model_url,
            "max_workers": max_workers,
        }

        base_lm_kwargs.update(language_model_params or {})

        filtered_kwargs = {k: v for k, v in base_lm_kwargs.items() if v is not None}

        config = factory.ModelConfig(model_id=model_id, provider_kwargs=filtered_kwargs)
        language_model = factory.create_model(
            config=config,
            examples=prompt_template.examples if use_schema_constraints else None,
            use_schema_constraints=use_schema_constraints,
            fence_output=fence_output,
        )

    # Build FormatHandler from resolver_params or defaults.
    rp = dict(resolver_params or {})
    if rp.get("format_handler") is not None:
        format_handler = rp.pop("format_handler")
    else:
        format_handler = fh.FormatHandler(
            format_type=format_type,
            use_fences=language_model.requires_fence_output,
            attribute_suffix=data.ATTRIBUTE_SUFFIX,
            use_wrapper=True,
            wrapper_key=data.EXTRACTIONS_KEY,
        )
    remaining_params = rp

    if language_model.schema is not None:
        language_model.schema.validate_format(format_handler)

    alignment_kwargs: dict[str, typing.Any] = {}
    for key in resolver.ALIGNMENT_PARAM_KEYS:
        val = remaining_params.pop(key, None)
        if val is not None:
            alignment_kwargs[key] = val

    effective_params = {"format_handler": format_handler, **remaining_params}

    try:
        res = resolver.Resolver(**effective_params)
    except TypeError as e:
        msg = str(e)
        if (
            "unexpected keyword argument" in msg
            or "got an unexpected keyword argument" in msg
        ):
            raise TypeError(
                f"Unknown key in resolver_params; check spelling: {e}"
            ) from e
        raise

    annotator_inst = annotation.Annotator(
        language_model=language_model,
        prompt_template=prompt_template,
        format_handler=format_handler,
        hooks=hooks,
    )

    return (text_or_documents, annotator_inst, res, alignment_kwargs)


def extract(
    text_or_documents: typing.Any,
    prompt_description: str | None = None,
    examples: typing.Sequence[typing.Any] | None = None,
    model_id: str = "gemini-2.5-flash",
    api_key: str | None = None,
    format_type: typing.Any = None,
    max_char_buffer: int = 1000,
    temperature: float | None = None,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    batch_length: int = 10,
    max_workers: int = 10,
    additional_context: str | None = None,
    resolver_params: dict | None = None,
    language_model_params: dict | None = None,
    debug: bool = False,
    model_url: str | None = None,
    extraction_passes: int = 1,
    context_window_chars: int | None = None,
    config: typing.Any = None,
    model: typing.Any = None,
    *,
    schema: type[pydantic.BaseModel] | None = None,
    fetch_urls: bool = True,
    prompt_validation_level: pv.PromptValidationLevel = pv.PromptValidationLevel.WARNING,
    prompt_validation_strict: bool = False,
    show_progress: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
    hooks: hooks_lib.Hooks | None = None,
    optimized_config: typing.Any = None,
) -> list[data.AnnotatedDocument] | data.AnnotatedDocument:
    """Extracts structured information from text.

    Retrieves structured information from the provided text or documents using a
    language model based on the instructions in prompt_description and guided by
    examples. Supports sequential extraction passes to improve recall at the cost
    of additional API calls.

    Args:
        text_or_documents: The source text to extract information from, a URL to
          download text from (starting with http:// or https:// when fetch_urls
          is True), or an iterable of Document objects.
        prompt_description: Instructions for what to extract from the text.
        examples: List of ExampleData objects to guide the extraction.
        tokenizer: Optional Tokenizer instance to use for chunking and alignment.
          If None, defaults to RegexTokenizer.
        api_key: API key for Gemini or other LLM services (can also use
          environment variable LANGCORE_API_KEY). Cost considerations: Most
          APIs charge by token volume. Smaller max_char_buffer values increase the
          number of API calls, while extraction_passes > 1 reprocesses tokens
          multiple times. Note that max_workers improves processing speed without
          additional token costs. Refer to your API provider's pricing details and
          monitor usage with small test runs to estimate costs.
        model_id: The model ID to use for extraction (e.g., 'gemini-2.5-flash').
          If your model ID is not recognized or you need to use a custom provider,
          use the 'config' parameter with factory.ModelConfig to specify the
          provider explicitly.
        format_type: The format type for the output (JSON or YAML).
        max_char_buffer: Max number of characters for inference.
        temperature: The sampling temperature for generation. When None (default),
          uses the model's default temperature. Set to 0.0 for deterministic output
          or higher values for more variation.
        fence_output: Whether to expect/generate fenced output (```json or
          ```yaml). When True, the model is prompted to generate fenced output and
          the resolver expects it. When False, raw JSON/YAML is expected. When None,
          automatically determined based on provider schema capabilities: if a schema
          is applied and requires_raw_output is True, defaults to False; otherwise
          True. If your model utilizes schema constraints, this can generally be set
          to False unless the constraint also accounts for code fence delimiters.
        use_schema_constraints: Whether to generate schema constraints for models.
          For supported models, this enables structured outputs. Defaults to True.
        batch_length: Number of text chunks processed per batch. Higher values
          enable greater parallelization when batch_length >= max_workers.
          Defaults to 10.
        max_workers: Maximum parallel workers for concurrent processing. Effective
          parallelization is limited by min(batch_length, max_workers). Supported
          by Gemini models. Defaults to 10.
        additional_context: Additional context to be added to the prompt during
          inference.
        resolver_params: Parameters for the `resolver.Resolver`, which parses the
          raw language model output string (e.g., extracting JSON from ```json ...
          ``` blocks) into structured `data.Extraction` objects. This dictionary
          overrides default settings. Keys include: - 'extraction_index_suffix'
          (str | None): Suffix for keys indicating extraction order. Default is
          None (order by appearance). Additional alignment parameters can be
          included: 'enable_fuzzy_alignment' (bool): Whether to use fuzzy matching
          if exact matching fails. Disabling this can improve performance but may
          reduce recall. Default is True. 'fuzzy_alignment_threshold' (float):
          Minimum token overlap ratio for fuzzy match (0.0-1.0). Default is 0.75.
          'accept_match_lesser' (bool): Whether to accept partial exact matches.
          Default is True. 'suppress_parse_errors' (bool): Whether to suppress
          parsing errors and continue pipeline. Default is False.
        language_model_params: Additional parameters for the language model.
        debug: Whether to enable debug logging. When True, enables detailed logging
          of function calls, arguments, return values, and timing for the langcore
          namespace. Note: Debug logging remains enabled for the process once activated.
        model_url: Endpoint URL for self-hosted or on-prem models.
        extraction_passes: Number of sequential extraction attempts to improve
          recall and find additional entities. Defaults to 1 (standard single
          extraction). When > 1, the system performs multiple independent
          extractions and merges non-overlapping results (first extraction wins
          for overlaps). WARNING: Each additional pass reprocesses tokens,
          potentially increasing API costs. For example, extraction_passes=3
          reprocesses tokens 3x.
        context_window_chars: Number of characters from the previous chunk to
          include as context for the current chunk. This helps with coreference
          resolution across chunk boundaries (e.g., resolving "She" to a person
          mentioned in the previous chunk). Defaults to None (disabled).
        config: Model configuration to use for extraction. Takes precedence over
          model_id and api_key parameters. When both model and config are
          provided, model takes precedence.
        model: Pre-configured language model to use for extraction. Takes
          precedence over all other parameters including config.
        fetch_urls: Whether to automatically download content when the input is a
          URL string. When True (default), strings starting with http:// or
          https:// are fetched. When False, all strings are treated as literal
          text to analyze. This is a keyword-only parameter.
        prompt_validation_level: Controls pre-flight alignment checks on few-shot
          examples. OFF skips validation, WARNING logs issues but continues, ERROR
          raises on failures. Defaults to WARNING.
        prompt_validation_strict: When True and prompt_validation_level is ERROR,
          raises on non-exact matches (MATCH_FUZZY, MATCH_LESSER). Defaults to False.
        show_progress: Whether to show progress bar during extraction. Defaults to True.
        schema: Optional Pydantic BaseModel subclass defining the extraction
          schema. When provided, ``prompt_description`` and ``examples`` are
          auto-generated from the model's field metadata if not explicitly
          passed. This enables a schema-first API where you define what to
          extract using a Pydantic model.
        hooks: Optional ``Hooks`` instance for lifecycle event callbacks.
          Receives events such as ``extraction:start``, ``extraction:chunk``,
          ``extraction:llm_call``, ``extraction:alignment``,
          ``extraction:complete``, and ``extraction:error``.
        optimized_config: Optional ``OptimizedConfig`` from
          ``langcore-dspy``.  When provided, its
          ``prompt_description`` and ``examples`` override the
          corresponding parameters.

    Returns:
        An AnnotatedDocument with the extracted information when input is a
        string or URL, or an iterable of AnnotatedDocuments when input is an
        iterable of Documents.

    Raises:
        ValueError: If neither examples nor schema is provided.
        ValueError: If no API key is provided or found in environment variables.
        requests.RequestException: If URL download fails.
        pv.PromptAlignmentError: If validation fails in ERROR mode.
    """
    # Apply optimized_config overrides if provided.
    if optimized_config is not None:
        prompt_description = optimized_config.prompt_description
        examples = optimized_config.examples

    _hooks = _config.resolve_hooks(hooks)

    try:
        text_or_documents, annotator, res, alignment_kwargs = (
            _build_extraction_components(
                text_or_documents=text_or_documents,
                prompt_description=prompt_description,
                examples=examples,
                model_id=model_id,
                api_key=api_key,
                format_type=format_type,
                max_char_buffer=max_char_buffer,
                temperature=temperature,
                fence_output=fence_output,
                use_schema_constraints=use_schema_constraints,
                batch_length=batch_length,
                max_workers=max_workers,
                additional_context=additional_context,
                resolver_params=resolver_params,
                language_model_params=language_model_params,
                debug=debug,
                model_url=model_url,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                config=config,
                model=model,
                fetch_urls=fetch_urls,
                prompt_validation_level=prompt_validation_level,
                prompt_validation_strict=prompt_validation_strict,
                show_progress=show_progress,
                tokenizer=tokenizer,
                schema=schema,
                hooks=hooks,
            )
        )

        # Emit extraction:start hook.
        _hooks.emit(
            hooks_lib.HookName.EXTRACTION_START,
            {
                "model_id": model_id,
                "batch_length": batch_length,
                "extraction_passes": extraction_passes,
                "max_char_buffer": max_char_buffer,
                "text_length": (
                    len(text_or_documents)
                    if isinstance(text_or_documents, str)
                    else None
                ),
            },
        )

        if isinstance(text_or_documents, str):
            result = annotator.annotate_text(
                text=text_or_documents,
                resolver=res,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                additional_context=additional_context,
                debug=debug,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                show_progress=show_progress,
                max_workers=max_workers,
                tokenizer=tokenizer,
                **alignment_kwargs,
            )
            _hooks.emit(hooks_lib.HookName.EXTRACTION_COMPLETE, result)
            return result
        else:
            documents = cast("Iterable[data.Document]", text_or_documents)
            result = annotator.annotate_documents(
                documents=documents,
                resolver=res,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                debug=debug,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                show_progress=show_progress,
                max_workers=max_workers,
                tokenizer=tokenizer,
                **alignment_kwargs,
            )
            result_list = list(result)
            _hooks.emit(hooks_lib.HookName.EXTRACTION_COMPLETE, result_list)
            return result_list
    except Exception as exc:
        _hooks.emit(hooks_lib.HookName.EXTRACTION_ERROR, exc)
        raise


async def async_extract(
    text_or_documents: typing.Any,
    prompt_description: str | None = None,
    examples: typing.Sequence[typing.Any] | None = None,
    model_id: str = "gemini-2.5-flash",
    api_key: str | None = None,
    format_type: typing.Any = None,
    max_char_buffer: int = 1000,
    temperature: float | None = None,
    fence_output: bool | None = None,
    use_schema_constraints: bool = True,
    batch_length: int = 10,
    max_workers: int = 10,
    additional_context: str | None = None,
    resolver_params: dict | None = None,
    language_model_params: dict | None = None,
    debug: bool = False,
    model_url: str | None = None,
    extraction_passes: int = 1,
    context_window_chars: int | None = None,
    config: typing.Any = None,
    model: typing.Any = None,
    *,
    schema: type[pydantic.BaseModel] | None = None,
    fetch_urls: bool = True,
    prompt_validation_level: pv.PromptValidationLevel = pv.PromptValidationLevel.WARNING,
    prompt_validation_strict: bool = False,
    show_progress: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
    hooks: hooks_lib.Hooks | None = None,
    optimized_config: typing.Any = None,
) -> list[data.AnnotatedDocument] | data.AnnotatedDocument:
    """Async version of ``extract`` for non-blocking LLM inference.

    Uses ``BaseLanguageModel.async_infer`` under the hood and pipelines
    I/O-bound inference with CPU-bound alignment for improved throughput.
    The function signature mirrors ``extract`` so it can be used as a
    drop-in replacement in async contexts.

    Providers that implement native ``async_infer`` (e.g. LiteLLM via
    ``litellm.acompletion``) get true async I/O.  All other providers
    fall back to ``asyncio.to_thread`` automatically.

    Args:
      text_or_documents: Text, URL, or iterable of Document objects.
      prompt_description: Instructions for what to extract.
      examples: List of ExampleData objects.
      model_id: The model ID to use.
      api_key: API key for LLM services.
      format_type: Output format (JSON or YAML).
      max_char_buffer: Max characters per inference chunk.
      temperature: Sampling temperature.
      fence_output: Whether to expect fenced output.
      use_schema_constraints: Whether to generate schema constraints.
      batch_length: Chunks per batch.
      max_workers: Maximum parallel workers.
      additional_context: Additional context for the prompt.
      resolver_params: Parameters for the Resolver.
      language_model_params: Additional model parameters.
      debug: Enable debug logging.
      model_url: Endpoint URL for self-hosted models.
      extraction_passes: Number of extraction passes.
      context_window_chars: Context overlap between chunks.
      config: Model configuration.
      model: Pre-configured language model.
      schema: Optional Pydantic BaseModel subclass defining the extraction
        schema. When provided, prompt_description and examples are
        auto-generated from the model's field metadata if not explicitly
        passed.
      fetch_urls: Whether to fetch URLs.
      prompt_validation_level: Prompt validation level.
      prompt_validation_strict: Strict prompt validation.
      show_progress: Show progress bar.
      tokenizer: Optional tokenizer instance.
      hooks: Optional Hooks instance for lifecycle event callbacks.
      optimized_config: Optional ``OptimizedConfig`` from
        ``langcore-dspy``.  When provided, its
        ``prompt_description`` and ``examples`` override the
        corresponding parameters.

    Returns:
      AnnotatedDocument (string input) or list of AnnotatedDocuments.
    """
    # Apply optimized_config overrides if provided.
    if optimized_config is not None:
        prompt_description = optimized_config.prompt_description
        examples = optimized_config.examples

    _hooks = _config.resolve_hooks(hooks)

    try:
        text_or_documents, annotator, res, alignment_kwargs = (
            _build_extraction_components(
                text_or_documents=text_or_documents,
                prompt_description=prompt_description,
                examples=examples,
                model_id=model_id,
                api_key=api_key,
                format_type=format_type,
                max_char_buffer=max_char_buffer,
                temperature=temperature,
                fence_output=fence_output,
                use_schema_constraints=use_schema_constraints,
                batch_length=batch_length,
                max_workers=max_workers,
                additional_context=additional_context,
                resolver_params=resolver_params,
                language_model_params=language_model_params,
                debug=debug,
                model_url=model_url,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                config=config,
                model=model,
                fetch_urls=fetch_urls,
                prompt_validation_level=prompt_validation_level,
                prompt_validation_strict=prompt_validation_strict,
                show_progress=show_progress,
                tokenizer=tokenizer,
                schema=schema,
                hooks=hooks,
            )
        )

        # Emit extraction:start hook.
        await _hooks.async_emit(
            hooks_lib.HookName.EXTRACTION_START,
            {
                "model_id": model_id,
                "batch_length": batch_length,
                "extraction_passes": extraction_passes,
                "max_char_buffer": max_char_buffer,
                "text_length": (
                    len(text_or_documents)
                    if isinstance(text_or_documents, str)
                    else None
                ),
            },
        )

        if isinstance(text_or_documents, str):
            result = await annotator.async_annotate_text(
                text=text_or_documents,
                resolver=res,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                additional_context=additional_context,
                debug=debug,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                show_progress=show_progress,
                max_workers=max_workers,
                tokenizer=tokenizer,
                **alignment_kwargs,
            )
            await _hooks.async_emit(hooks_lib.HookName.EXTRACTION_COMPLETE, result)
            return result
        else:
            documents = cast("Iterable[data.Document]", text_or_documents)
            result = await annotator.async_annotate_documents(
                documents=documents,
                resolver=res,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                debug=debug,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                show_progress=show_progress,
                max_workers=max_workers,
                tokenizer=tokenizer,
                **alignment_kwargs,
            )
            await _hooks.async_emit(hooks_lib.HookName.EXTRACTION_COMPLETE, result)
            return result
    except Exception as exc:
        await _hooks.async_emit(hooks_lib.HookName.EXTRACTION_ERROR, exc)
        raise
