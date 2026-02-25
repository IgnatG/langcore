# CHANGELOG

<!-- version list -->

## v1.1.7 (2026-02-25)

### Bug Fixes

- Enhance extraction value handling to skip nested lists of dicts
  ([`3c89871`](https://github.com/IgnatG/langcore/commit/3c8987139cadff0701bee6a956e3a07975b34753))

- Streamline extraction value handling to skip nested lists of dicts
  ([`3dbb439`](https://github.com/IgnatG/langcore/commit/3dbb43939d248ea9c0d0f85b0755282174a8d27c))


## v1.1.6 (2026-02-25)

### Bug Fixes

- Enhance error handling for extraction values in Resolver
  ([`7ad69af`](https://github.com/IgnatG/langcore/commit/7ad69afa2bf51a5004554fab3658fb3bbb65a2fa))


## v1.1.5 (2026-02-25)

### Bug Fixes

- Improve error handling and logging in Annotator and FormatHandler
  ([`e49e478`](https://github.com/IgnatG/langcore/commit/e49e4783f0fec1c1116f6907e7d64c9ff7d443c0))


## v1.1.4 (2026-02-25)

### Bug Fixes

- Add `registry` as a public alias for `providers.router` so that plugins using
  `lx.providers.registry.register(...)` resolve correctly

## v1.1.3 (2026-02-24)

### Bug Fixes

- Update resolver initialization to use format_handler and improve caching in various components
  ([`fa9cac2`](https://github.com/IgnatG/langcore/commit/fa9cac242e607b36a9626c40433133ab1d02541d))

## v1.1.2 (2026-02-24)

### Bug Fixes

- Simplify class definition and improve terminal state handling in gemini provider
  ([`c7f3253`](https://github.com/IgnatG/langcore/commit/c7f32534567dfa27adee75757a0ccb83af923d30))

## v1.1.1 (2026-02-24)

### Bug Fixes

- Make gemini depdendencies optional
  ([`2fdf577`](https://github.com/IgnatG/langcore/commit/2fdf57710baf7590efe6c60071a280679e226bbe))

### Chores

- Fix failing tests
  ([`267b101`](https://github.com/IgnatG/langcore/commit/267b101df8d9c44f726e51ae71d89cdf7b641a76))

## v1.0.2 (2026-02-23)

### Bug Fixes

- Update repository links and improve documentation for provider plugin creation
  ([`4ca737c`](https://github.com/IgnatG/langcore/commit/4ca737c9bb3c4ef0225cb4b502427b111525bf75))

## v1.0.1 (2026-02-23)

### Bug Fixes

- Fixed failing tests
  ([`e007322`](https://github.com/IgnatG/langcore/commit/e00732266dfbc7b9f9856c0a85bd447ebb5bd4c5))

## v1.0.0 (2026-02-23)

- Initial Release

## v1.2.1 (2026-02-23)

### Bug Fixes

- Implement extraction hooks and event system, add configurable confidence weights, and centralize
  schema utilities
  ([`76038c6`](https://github.com/IgnatG/langcore/commit/76038c67fe94004dd2c054e2b4b01b8dbbd82fcd))

## v1.2.0 (2026-02-23)

### Features

- Add quality metrics and evaluation module with extraction metrics and convenience functions
  ([`87cf0b1`](https://github.com/IgnatG/langcore/commit/87cf0b15fcdca5c00f74e6de13640f2e5b6b6d20))

- Add RAG Query Parsing section to README with feature comparisons for langcore-rag plugin
  ([`e5f3170`](https://github.com/IgnatG/langcore/commit/e5f3170a5ab851d45b31b95b617d0f4634f46db4))

- **Extraction Hooks & Event System** (`langcore.hooks`): New `HookName` enum with 6 lifecycle
  events (`extraction:start`, `extraction:chunk`, `extraction:llm_call`, `extraction:alignment`,
  `extraction:complete`, `extraction:error`). `Hooks` class with `on()`, `off()`, `clear()`,
  `emit()`, `async_emit()`, and `__add__()` for composing hook sets. Both sync and async hook
  emission supported. Per-call hooks via `hooks=` parameter on `extract()` and `async_extract()`.

- **Configurable confidence weights**: `compute_alignment_confidence()` now accepts optional
  `w_alignment` and `w_overlap` keyword arguments to override the default 0.7/0.3 weighting.

- **Shared schema utilities**: Centralised `find_primary_text_field()` in
  `langcore.core.schema_utils` — eliminates duplication across `schema_adapter` and `core.data`.

### Bug Fixes

- Fix `schema_from_example()`: fields inferred from `None` values now correctly type as
  `str | None` instead of bare `str`, which caused Pydantic v2 validation errors.

- Updated `Extraction.confidence_score` docstring to reflect its dual meaning (single-pass
  alignment confidence and multi-pass cross-pass frequency).

## v1.1.0 (2026-02-22)

### Features

- **Pydantic Schema Support:** New `schema` parameter on `lx.extract()` and `lx.async_extract()` accepts a Pydantic `BaseModel` subclass to define extraction targets. When provided, prompt descriptions and JSON schema constraints are auto-generated from model field metadata.
- **Schema Adapter (`langcore.schema_adapter`):** New `PydanticSchemaAdapter` class and `schema_from_pydantic()` convenience function to convert Pydantic models into LangCore-compatible `SchemaConfig` objects (prompt description, examples, JSON schema).
- **Schema Generator (`langcore.schema_generator`):** New `schema_from_example()` and `schema_from_examples()` functions to auto-generate Pydantic models from plain dictionaries with type inference.
- **`AnnotatedDocument.to_pydantic(schema)`:** New convenience method to convert raw `Extraction` objects back into typed Pydantic model instances.
- **Confidence Scoring:** Every extraction now receives a `confidence_score` (0.0–1.0) after alignment. The score combines alignment quality (70% weight) with token overlap ratio (30% weight). Alignment quality maps: `MATCH_EXACT` → 1.0, `MATCH_LESSER` → 0.8, `MATCH_GREATER` → 0.7, `MATCH_FUZZY` → 0.5, unaligned → 0.2.
- **`AnnotatedDocument.average_confidence`:** New computed property returning the mean confidence score across all extractions in a document.
- **Multi-pass confidence augmentation:** Multi-pass extraction now combines cross-pass appearance frequency with per-extraction alignment confidence for more accurate scoring.

### Backward Compatibility

- Existing `prompt_description + examples` API remains fully functional and unchanged.
- The `schema` parameter is keyword-only and defaults to `None`.

### Removed

- Removed deprecated `language_model_type` parameter from `extract()` and `async_extract()`.
- Removed deprecated `gemini_schema` parameter handling.
- Removed `_compat` backward compatibility module.
- Removed `registry` lazy module from package init.

## v1.0.0 (2026-02-21)

- Initial Release
