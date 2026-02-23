# CHANGELOG

<!-- version list -->

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
