# LangCore

## Overview

**LangCore** is a Python library for LLM-powered structured information extraction from unstructured text. It is built on top of Google's open-source [LangCore](https://github.com/ignatg/langcore) library (Apache 2.0), extending it with additional capabilities for production  document processing workflows.

> **Attribution:** The core extraction engine is derived from [LangCore by Google LLC](https://github.com/ignatg/langcore). See the [NOTICE](NOTICE) file for full attribution details.

## Table of Contents

- [Overview](#overview)
- [Feature Overview](#feature-overview)
- [Core Capabilities](#core-capabilities)
- [Quick Start](#quick-start)
- [Schema-First Extraction with Pydantic](#schema-first-extraction-with-pydantic)
- [Multi-Model Consensus Extraction](#multi-model-consensus-extraction)
- [Confidence Scoring](#confidence-scoring)
- [Extraction Reliability Score](#extraction-reliability-score)
- [Extraction Hooks & Events](#extraction-hooks--events)
- [Quality Metrics & Evaluation](#quality-metrics--evaluation)
- [Ecosystem Plugins](#ecosystem-plugins)
- [Installation](#installation)
- [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
- [Adding Custom Model Providers](#adding-custom-model-providers)
- [Using OpenAI Models](#using-openai-models)
- [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
- [More Examples](#more-examples)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)

## Feature Overview

LangCore is a batteries-included extraction framework. Core features ship with the `langcore` package; additional capabilities are available through first-party plugin packages that you install as needed.

### Core Features (`langcore`)

| Feature | Description |
|---|---|
| **Few-Shot Extraction** | Define extraction tasks with a prompt and examples — no fine-tuning required |
| **Pydantic Schema Extraction** | Define targets as Pydantic models; prompts, JSON schema constraints, and seed examples are auto-generated |
| **Schema Validation Retry** | Instructor-style validate → re-ask loop: validates extractions against the Pydantic schema and retries invalid ones with error feedback |
| **Multi-Model Consensus** | Run extraction across multiple LLM providers and merge results — extractions confirmed by multiple models receive higher confidence |
| **Confidence Scoring** | Per-extraction confidence (0.0–1.0) combining alignment quality + token overlap, with configurable weights |
| **Extraction Reliability Score** | Composite quality metric (0.0–1.0) combining confidence, schema validity, field completeness, and source grounding |
| **Source Grounding & Alignment** | Maps every extraction to its exact character position in the source text for traceability and verification |
| **Long Document Chunking** | Optimized chunking + parallel processing overcomes the "needle-in-a-haystack" problem in large documents |
| **Multi-Pass Extraction** | Configurable `extraction_passes` for higher recall with cross-pass confidence boosting |
| **Schema Utilities** | `to_pydantic()`, `schema_from_pydantic()`, `schema_from_example()`, `schema_from_examples()` — convert between Pydantic models, dicts, and LangCore's internal format |
| **Extraction Hooks & Events** | 6 lifecycle events (`start`, `chunk`, `llm_call`, `alignment`, `complete`, `error`) with fault-tolerant callbacks |
| **Global Configuration** | `lx.configure(hooks=...)` sets app-wide hooks; per-call hooks compose via `+` operator |
| **Prompt Alignment Validation** | Automatic warnings when example `extraction_text` doesn't appear verbatim in the example `text` |
| **Quality Metrics & Evaluation** | Built-in P/R/F1/accuracy with per-field and per-document breakdown, fuzzy matching, and averaging modes |
| **Response Caching** | Built-in LLM response cache with automatic cache-busting for multi-pass extraction |
| **Interactive Visualization** | Generates self-contained HTML to review extractions in their original context |
| **Flexible Model Support** | Built-in providers for Gemini, OpenAI, and Ollama; extensible via plugins |
| **Custom Provider System** | `BaseLanguageModel` ABC + entry-point plugin registry with priority-based resolution |
| **Async & Parallel** | `async_extract()` with `max_workers` for concurrent chunk processing |
| **URL/File Input** | Accepts URLs, file paths, and raw text directly |
| **Batch API** | Vertex AI Batch API support for large-scale jobs |
| **Controlled Generation** | JSON schema constraints via supported models (Gemini) |

### Plugin Features

| Feature | Package | Description |
|---|---|---|
| **100+ LLM Providers** | `langcore-litellm` | OpenAI, Anthropic, Azure, Mistral, Groq, Cohere, HuggingFace, Ollama, vLLM, and more via LiteLLM |
| **Output Validation & Retry** | `langcore-guardrails` | 7 built-in validators with corrective retry loop and 4 on-fail actions |
| **Grounding Validator** | `langcore-guardrails` | Rejects hallucinated extractions using alignment quality and character coverage checks |
| **Confidence Threshold** | `langcore-guardrails` | Filters extractions below a confidence score cutoff |
| **Schema / JSON Validation** | `langcore-guardrails` | Pydantic and JSON Schema validators with strict/lenient modes |
| **Consistency Rules** | `langcore-guardrails` | Cross-checks extracted values using user-supplied business rules |
| **Regex Validation** | `langcore-guardrails` | Match extracted output against regex patterns |
| **Field Completeness** | `langcore-guardrails` | Ensure required schema fields are present and non-empty |
| **Validator Chaining** | `langcore-guardrails` | Compose multiple validators with per-validator failure actions |
| **Validator Registry** | `langcore-guardrails` | `@register_validator` decorator for plugging in custom validators |
| **Audit Logging** | `langcore-audit` | Structured audit records for every LLM call — pluggable sinks (logging, JSONL, OpenTelemetry) |
| **Hybrid Rules + LLM** | `langcore-hybrid` | Deterministic regex/function rules with LLM fallback — save 50–80% on LLM costs |
| **Prompt Optimization** | `langcore-dspy` | Automatic prompt and few-shot example optimization using DSPy (MIPROv2, GEPA) |
| **Evaluation (P/R/F1)** | `langcore-dspy` | Built-in evaluation with per-document precision, recall, and F1 for optimized configs |
| **RAG Query Parsing** | `langcore-rag` | Decompose natural-language queries into semantic terms + structured metadata filters |
| **Query Caching** | `langcore-rag` | LRU cache for parsed queries with Pydantic schema introspection |
| **PDF Support** | `langcore-docling` | Native PDF extraction via Docling — drop-in replacement for `lx.extract()` |
| **HTTP API** | `langcore-api` | Production-ready REST service with task queuing, caching, webhooks, and Prometheus metrics |

## Core Capabilities

1. **Precise Source Grounding:** Maps every extraction to its exact location in the source text, enabling visual highlighting for easy traceability and verification.
2. **Reliable Structured Outputs:** Enforces a consistent output schema based on your few-shot examples, leveraging controlled generation in supported models like Gemini to guarantee robust, structured results.
3. **Optimized for Long Documents:** Overcomes the "needle-in-a-haystack" challenge of large document extraction by using an optimized strategy of text chunking, parallel processing, and multiple passes for higher recall.
4. **Interactive Visualization:** Instantly generates a self-contained, interactive HTML file to visualize and review thousands of extracted entities in their original context.
5. **Flexible LLM Support:** Supports your preferred models, from cloud-based LLMs like the Google Gemini family to local open-source models via the built-in Ollama interface.
6. **Adaptable to Any Domain:** Define extraction tasks for any domain using just a few examples — no model fine-tuning required.
7. **Leverages LLM World Knowledge:** Utilize precise prompt wording and few-shot examples to influence how the extraction task may utilize LLM knowledge.

## Quick Start

> **Note:** Using cloud-hosted models like Gemini requires an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions on how to get and configure your key.

Extract structured information with just a few lines of code.

### 1. Define Your Extraction Task

First, create a prompt that clearly describes what you want to extract. Then, provide a high-quality example to guide the model.

```python
import langcore as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]
```

> **Note:** Examples drive model behavior. Each `extraction_text` should ideally be verbatim from the example's `text` (no paraphrasing), listed in order of appearance. LangCore raises `Prompt alignment` warnings by default if examples don't follow this pattern—resolve these for best results.

### 2. Run the Extraction

Provide your input text and the prompt materials to the `lx.extract` function.

```python
# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering an excellent balance of speed, cost, and quality. For highly complex tasks requiring deeper reasoning, `gemini-2.5-pro` may provide superior results. For large-scale or production use, a Tier 2 Gemini quota is suggested to increase throughput and avoid rate limits. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Note that Gemini models have a lifecycle with defined retirement dates. Users should consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest stable and legacy versions.

### 3. Visualize the Results

The extractions can be saved to a `.jsonl` file, a popular format for working with language model data. LangCore can then generate an interactive HTML visualization from this file to review the entities in context.

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization](docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text evidence - extracting "longing" for Lady Juliet's emotional state and identifying "yearning" from "gazed longingly at the stars." The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

For larger texts, you can process entire documents directly from URLs with parallel processing and enhanced sensitivity:

```python
# Process Romeo & Juliet directly from Project Gutenberg
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=20,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)
```

> **Multi-pass & caching:** When `extraction_passes > 1`, the first pass uses
> normal caching behaviour while subsequent passes include a `pass_num` keyword
> argument that providers can use to bypass response caches. The
> [langcore-litellm](https://github.com/JustStas/langcore-litellm)
> provider does this automatically — passes ≥ 2 always hit the live LLM API.

This approach can extract hundreds of entities from full novels while maintaining high accuracy. The interactive visualization seamlessly handles large result sets, making it easy to explore hundreds of entities from the output JSONL file. **[See the full *Romeo and Juliet* extraction example →](docs/examples/longer_text_example.md)** for detailed results and performance insights.

### Vertex AI Batch Processing

Save costs on large-scale tasks by enabling Vertex AI Batch API: `language_model_params={"vertexai": True, "batch": {"enabled": True}}`.

See an example of the Vertex AI Batch API usage in [this example](docs/examples/batch_api_example.md).

### Schema-First Extraction with Pydantic

Instead of manually constructing `ExampleData` objects, you can define your extraction schema as a Pydantic model. LangCore will auto-generate the prompt and schema constraints for you.

```python
from pydantic import BaseModel, Field
import langcore as lx

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID like INV-001")
    amount: float = Field(description="Total amount in dollars")
    due_date: str = Field(description="Due date in YYYY-MM-DD format")

result = lx.extract(
    text="Invoice INV-2024-789 for $3,450 is due April 20th, 2024",
    schema=Invoice,
    model_id="gemini-2.5-flash",
)

# Convert extractions back to typed Pydantic instances
invoices = result.to_pydantic(Invoice)
for inv in invoices:
    print(f"{inv.invoice_number}: ${inv.amount} due {inv.due_date}")
```

You can also combine `schema` with explicit `examples` for the best of both worlds — the Pydantic model defines the structure, and examples provide few-shot guidance:

```python
result = lx.extract(
    text="...",
    schema=Invoice,
    examples=[
        lx.data.ExampleData(
            text="Invoice INV-001 for $100 due Jan 1, 2024",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Invoice",
                    extraction_text="INV-001",
                    attributes={"amount": "100.0", "due_date": "2024-01-01"},
                )
            ],
        )
    ],
    model_id="gemini-2.5-flash",
)
```

> **Tip:** Use `lx.schema_from_pydantic(Invoice)` to inspect the auto-generated prompt and JSON schema before running extraction. Use `lx.schema_from_example({"name": "John", "age": 30})` to auto-generate a Pydantic model from a plain dict, or `lx.schema_from_examples([{"name": "John"}, {"name": "Jane", "age": 30}])` to merge multiple examples (fields default to optional when absent from some examples). When a field has mixed types across examples (e.g. `int` in one and `str` in another), the generated model uses a `Union` type (`int | str`) so Pydantic accepts any of the observed types.

> **Under the hood:** The `PydanticSchemaAdapter` converts your Pydantic model into LangCore's internal `SchemaConfig` — auto-generating the prompt description, JSON schema, and seed examples. You can use it directly for advanced scenarios: `from langcore.pydantic_schema import PydanticSchemaAdapter`.

#### Schema Validation Retries

When you pass a `schema`, validation retries are **auto-enabled** (1 retry by default). After extraction, each result is validated against the Pydantic model; extractions that fail trigger a re-extraction with the validation error feedback, following the Instructor-style "validate → re-ask" pattern.

```python
# Auto-enabled — retries once automatically when schema is provided
result = lx.extract(
    text="Invoice INV-2024-789 for $3,450 is due April 20th, 2024",
    schema=Invoice,
    model_id="gemini-2.5-flash",
)
```

You can increase or decrease the retry count, or disable retries entirely:

```python
# Allow up to 3 retry attempts
result = lx.extract(text="...", schema=Invoice, schema_validation_retries=3)

# Explicitly disable retries (schema extraction only, no validation)
result = lx.extract(text="...", schema=Invoice, schema_validation_retries=0)
```

**How it works:**

1. Valid extractions from the first pass are always preserved.
2. Invalid extractions are collected with their Pydantic validation errors.
3. A correction prompt containing the specific error messages is appended.
4. **Chunk-level retry:** Only the text regions surrounding failing extractions are re-sent to the LLM (using `char_interval` to identify positions), not the entire document. Overlapping regions are merged automatically.
5. Newly valid results are offset back to document coordinates and merged in.
6. Steps 2–5 repeat up to `schema_validation_retries` times.
7. Token usage from all retry regions is accumulated in the final `usage`.

Retries also work for **Document list** inputs — each document is validated and retried independently.

When `schema` is not provided, `schema_validation_retries` defaults to 0 (no-op).

### Multi-Model Consensus Extraction

Run the same extraction across multiple LLM providers and automatically merge results. Extractions confirmed by multiple models receive higher confidence scores, while unique findings from individual models are still preserved.

```python
import langcore as lx

result = lx.extract(
    text="Patient Jane Doe received Lisinopril 10mg for hypertension.",
    examples=[...],
    consensus_models=["gemini-2.5-flash", "gpt-4o", "litellm/anthropic/claude-sonnet-4"],
)

# Extractions agreed upon by 2+ models get higher confidence
for ext in result.extractions:
    model = ext.attributes.get("_consensus_model_id", "unknown")
    print(f"{ext.extraction_text} (confidence={ext.confidence_score}, from={model})")
```

**Accepted model IDs:**

Each string in `consensus_models` is resolved through the same provider router used by `model_id`. Any model identifier that works with `extract()` works here too:

| Provider | Format | Examples |
|---|---|---|
| **Gemini** (built-in) | `gemini-*` | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash` |
| **OpenAI** (built-in) | `gpt-*` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| **Ollama** (built-in) | Model family name | `llama3.2:1b`, `mistral:7b`, `qwen2.5:72b`, `deepseek-coder-v2` |
| **LiteLLM** (plugin) | `litellm/<provider>/<model>` | `litellm/anthropic/claude-sonnet-4`, `litellm/gpt-4o`, `litellm/ollama/llama3`, `litellm/bedrock/anthropic.claude-3` |
| **Custom plugins** | Any registered pattern | Any model ID matched by a `langcore.providers` entry-point plugin |

You can freely mix providers in a single list — e.g. `["gemini-2.5-flash", "gpt-4o", "litellm/anthropic/claude-sonnet-4"]`.

> **LiteLLM support:** Install the `langcore-litellm` plugin (`pip install langcore-litellm`) to access 100+ models via a unified interface. Prefix model IDs with `litellm/` so the provider router dispatches correctly. See the [langcore-litellm README](../langcore-litellm/README.md) for full details.

**How it works:**

1. Each model in `consensus_models` runs extraction independently on the same text.
2. Results are merged using overlap-aware deduplication (same logic as multi-pass extraction).
3. Confidence is computed as `agreement_ratio × alignment_confidence`, where `agreement_ratio = models_that_found_it / total_models`.
4. Each extraction is tagged with `_consensus_model_id` in its attributes.
5. Token usage from all models is summed in the result.

Consensus works with all other features — schema validation retries, reliability scoring, and hooks all apply to the merged result. Both the sync and async versions run all models **concurrently** — async via `asyncio.gather`, sync via `ThreadPoolExecutor` — for maximum throughput.

When only one model is in the list, it falls back to standard single-model extraction.

### Confidence Scoring

Every extraction is automatically assigned a `confidence_score` between 0.0 and 1.0 after alignment. The score is computed by `compute_alignment_confidence()` in the resolver and combines two signals:

- **Alignment quality** (`w_alignment`, default 70%) — how well the extraction text matched the source: exact match = 1.0, lesser = 0.8, greater = 0.7, fuzzy = 0.5, unaligned = 0.2.
- **Token overlap ratio** (`w_overlap`, default 30%) — how many tokens in the extraction text vs. the matched source span.

All weights are configurable — pass `w_alignment` and `w_overlap` keyword arguments to `compute_alignment_confidence()` to tune the balance for your use case. The unaligned default (0.2) can also be overridden via `unaligned_confidence` when you want a different baseline for extractions that couldn't be located in the source text.

```python
result = lx.extract(
    text="Patient Jane Doe received Lisinopril for hypertension.",
    examples=[...],
    model_id="gemini-2.5-flash",
)

for extraction in result.extractions:
    print(f"{extraction.extraction_class}: {extraction.extraction_text} "
          f"(confidence: {extraction.confidence_score})")

# Document-level average confidence
print(f"Average confidence: {result.average_confidence}")
```

For **multi-pass extraction**, confidence is further augmented by cross-pass appearance frequency — extractions confirmed across multiple passes receive higher scores (`cross_pass_ratio × alignment_confidence`).

### Extraction Reliability Score

While `confidence_score` measures alignment quality, the **reliability score** is a composite metric that combines multiple quality signals into a single `reliability_score` (0.0–1.0) per extraction:

| Signal | Default Weight | What it measures |
|--------|---------------|-----------------|
| **Confidence** | 40% | Alignment-based `confidence_score` |
| **Schema validity** | 20% | Does the extraction pass Pydantic validation? |
| **Field completeness** | 20% | Are all required schema fields non-empty? |
| **Source grounding** | 20% | Does the extraction have a valid `char_interval`? |

Reliability scoring is **automatic** — every call to `extract()` computes it. When a `schema` is provided, the schema validity and field completeness signals are evaluated against it; otherwise they default to 1.0 (neutral).

```python
import langcore as lx
from pydantic import BaseModel

class Invoice(BaseModel):
    text: str
    amount: float
    currency: str = "USD"

result = lx.extract(text, schema=Invoice, model_id="gemini-2.5-flash")

for ext in result.extractions:
    print(f"{ext.extraction_text}: reliability={ext.reliability_score:.2f}")

# Document-level average reliability
print(f"Average reliability: {result.average_reliability}")
```

**Custom weights** — pass a `ReliabilityConfig` to tune the balance:

```python
from langcore import ReliabilityConfig

# Emphasise confidence and grounding, ignore schema signals
config = ReliabilityConfig(
    w_confidence=0.6,
    w_schema_valid=0.0,
    w_completeness=0.0,
    w_grounding=0.4,
)
result = lx.extract(text, schema=Invoice, reliability_config=config)
```

To disable reliability scoring entirely, pass `reliability_config=False`.

### Extraction Hooks & Events

The `langcore.hooks` module provides a lightweight event system inspired by
[Instructor](https://python.useinstructor.com/) hooks to inject custom logic at
every stage of the extraction pipeline — without modifying core code.

**Lifecycle events** are defined by the `HookName` enum (you can also use plain strings):

| Event | `HookName` | Fires when | Payload keys |
|---|---|---|---|
| `extraction:start` | `HookName.START` | Pipeline begins (after components are built) | `text`, `examples`, `model_id` |
| `extraction:chunk` | `HookName.CHUNK` | A document chunk has been processed | `chunk_index`, `num_chunks`, `chunk_text`, `extractions` |
| `extraction:llm_call` | `HookName.LLM_CALL` | An LLM inference call completes | `prompt`, `response` |
| `extraction:alignment` | `HookName.ALIGNMENT` | Extraction alignment is performed | `extractions` |
| `extraction:complete` | `HookName.COMPLETE` | Pipeline finishes successfully | `result` |
| `extraction:error` | `HookName.ERROR` | An exception is raised | `error` |

**Quick example:**

```python
from langcore.hooks import Hooks, HookName

hooks = Hooks()
hooks.on(HookName.START, lambda payload: print("Starting extraction…"))
hooks.on(HookName.LLM_CALL, lambda payload: print(f"LLM responded"))
hooks.on(HookName.ERROR, lambda payload: alert_team(payload["error"]))

result = lx.extract(
    text="Patient received Lisinopril 10mg daily.",
    examples=[...],
    model_id="gemini-2.5-flash",
    hooks=hooks,
)
```

**Programmatic emission** — use `emit()` (sync) or `async_emit()` (async) to fire
events from your own code or custom providers:

```python
hooks.emit("extraction:start", {"text": "hello", "model_id": "gpt-4o"})

# In async contexts, async_emit() awaits coroutine handlers
await hooks.async_emit("extraction:complete", {"result": result})
```

**Composing hooks** — merge two `Hooks` instances with `+`:

```python
logging_hooks = Hooks().on("extraction:llm_call", log_llm_call)
metrics_hooks = Hooks().on("extraction:complete", record_metrics)
combined = logging_hooks + metrics_hooks
```

**Removing handlers** — use `off()` to remove a specific handler, or `clear()` to remove all:

```python
hooks.off(HookName.LLM_CALL, log_llm_call)
hooks.clear()
```

Callbacks are **fault-tolerant**: if a handler raises an exception it is logged
and swallowed so it never breaks the extraction pipeline.

**Global hooks via `lx.configure()`** — set hooks once and they apply to every
`extract()` / `async_extract()` call without passing `hooks=` each time:

```python
import langcore as lx
from langcore.hooks import Hooks, HookName

# Set up global observability hooks once at startup
global_hooks = Hooks()
global_hooks.on(HookName.EXTRACTION_START, lambda cfg: print("Starting:", cfg["model_id"]))
global_hooks.on(HookName.EXTRACTION_ERROR, lambda err: alert_team(err))
lx.configure(hooks=global_hooks)

# Every extract() call now emits to global hooks automatically
result = lx.extract(text="...", examples=[...])

# Per-call hooks still work and fire AFTER global hooks
per_call = Hooks().on(HookName.EXTRACTION_COMPLETE, lambda r: log_result(r))
result = lx.extract(text="...", examples=[...], hooks=per_call)

# Inspect or reset global config
lx.get_config()   # {"hooks": <Hooks instance>}
lx.reset()         # Clears all global configuration
```

### Quality Metrics & Evaluation

The `langcore.evaluation` module provides built-in quality metrics for measuring extraction accuracy against ground truth. Compute precision, recall, F1, and accuracy at both the extraction level and per-field level.

```python
from langcore.evaluation import ExtractionMetrics

# Quick static helpers
print(ExtractionMetrics.f1(predictions=results, ground_truth=expected))
print(ExtractionMetrics.precision(predictions=results, ground_truth=expected))
```

**Full evaluation with per-field breakdown** — pass a Pydantic schema for field-level metrics:

```python
from pydantic import BaseModel, Field
from langcore.evaluation import ExtractionMetrics

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    amount: str = Field(description="Total amount")
    due_date: str = Field(description="Due date YYYY-MM-DD")

metrics = ExtractionMetrics(schema=Invoice)
report = metrics.evaluate(predictions=results, ground_truth=expected)
print(report.f1)          # 0.92
print(report.per_field)   # {"invoice_number": FieldReport(...), "amount": ...}
```

**Convenience function** — `lx.evaluate()` wraps `ExtractionMetrics` for quick one-liners:

```python
import langcore as lx

report = lx.evaluate(predictions=results, ground_truth=expected, schema=Invoice)
```

**Averaging modes** — control how multi-document metrics are aggregated:

```python
from langcore.evaluation import ExtractionMetrics

# Macro (default) — pool all extractions, compute P/R/F1 once
metrics = ExtractionMetrics(schema=Invoice, averaging="macro")

# Micro — compute P/R/F1 per document, then take unweighted mean
metrics = ExtractionMetrics(schema=Invoice, averaging="micro")

# Weighted — per-document P/R/F1 weighted by ground-truth count
metrics = ExtractionMetrics(schema=Invoice, averaging="weighted")
```

**Fuzzy matching** — allow near-matches instead of exact string equality:

```python
# Match extractions with ≥80% string similarity (difflib.SequenceMatcher)
metrics = ExtractionMetrics(fuzzy_threshold=0.8)
report = metrics.evaluate(predictions=results, ground_truth=expected)

# Also available via lx.evaluate()
import langcore as lx
report = lx.evaluate(predictions=results, ground_truth=expected, fuzzy_threshold=0.8)
```

The `EvaluationReport` includes:

- Aggregate `precision`, `recall`, `f1`, `accuracy`
- `averaging` — the strategy used (`"macro"`, `"micro"`, or `"weighted"`)
- `per_document` — list of per-document metric dicts
- `per_field` — dict of `FieldReport` objects with field-level P/R/F1 and support counts
- `strict_attributes=True` mode for matching on attribute values (not just class + text)

## Installation

### From Source

LangCore uses modern Python packaging with `pyproject.toml` for dependency management:

```bash
git clone https://github.com/IgnatG/langcore.git
cd langcore

# For basic installation:
pip install -e .

# For development (includes linting tools):
pip install -e ".[dev]"

# For testing (includes pytest):
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langcore .
docker run --rm -e LANGCORE_API_KEY="your-api-key" langcore python your_script.py
```

## API Key Setup for Cloud Models

When using LangCore with cloud-hosted models (like Gemini or OpenAI), you'll need to
set up an API key. On-device models don't require an API key. For developers
using local LLMs, LangCore offers built-in support for Ollama and can be
extended to other third-party APIs by updating the inference endpoints.

### API Key Sources

Get API keys from:

- [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models
- [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) for enterprise use
- [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI models

### Setting up API key in your environment

**Option 1: Environment Variable**

```bash
export LANGCORE_API_KEY="your-api-key-here"
```

**Option 2: .env File (Recommended)**

Add your API key to a `.env` file:

```bash
# Add API key to .env file
cat >> .env << 'EOF'
LANGCORE_API_KEY=your-api-key-here
EOF

# Keep your API key secure
echo '.env' >> .gitignore
```

In your Python code:

```python
import langcore as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash"
)
```

**Option 3: Direct API Key (Not Recommended for Production)**

You can also provide the API key directly in your code, though this is not recommended for production use:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

**Option 4: Vertex AI (Service Accounts)**

Use [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) for authentication with service accounts:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    language_model_params={
        "vertexai": True,
        "project": "your-project-id",
        "location": "global"  # or regional endpoint
    }
)
```

## Adding Custom Model Providers

LangCore supports custom LLM providers via a lightweight plugin system. You can add support for new models without changing core code.

- Add new model support independently of the core library
- Distribute your provider as a separate Python package
- Keep custom dependencies isolated
- Override or extend built-in providers via priority-based resolution

See the detailed guide in [Provider System Documentation](langcore/providers/README.md) to learn how to:

- Register a provider with `@registry.register(...)`
- Publish an entry point for discovery
- Optionally provide a schema with `get_schema_class()` for structured output
- Integrate with the factory via `create_model(...)`

## Using OpenAI Models

LangCore supports OpenAI models (requires optional dependency: `pip install langcore[openai]`):

```python
import langcore as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",  # Automatically selects OpenAI provider
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False` because LangCore doesn't implement schema constraints for OpenAI yet.

## Using Local LLMs with Ollama

LangCore supports local inference using Ollama, allowing you to run models without API keys:

```python
import langcore as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",  # Automatically selects Ollama provider
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False
)
```

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## More Examples

Additional examples of LangCore in action:

### *Romeo and Juliet* Full Text Extraction

LangCore can process complete documents directly from URLs. This example demonstrates extraction from the full text of *Romeo and Juliet* from Project Gutenberg (147,843 characters), showing parallel processing, sequential extraction passes, and performance optimization for long document processing.

**[View *Romeo and Juliet* Full Text Example →](docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes of LangCore's baseline capability only. It does not represent a finished or approved product, is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice.

LangCore excels at extracting structured medical information from clinical text. These examples demonstrate both basic entity recognition (medication names, dosages, routes) and relationship extraction (connecting medications to their attributes), showing LangCore's effectiveness for healthcare applications.

**[View Medication Examples →](docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces that shows how LangCore can automatically structure radiology reports. Try it directly in your browser with no setup required.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Ecosystem Plugins

LangCore has a growing ecosystem of first-party plugins. Each one is an independent package you install as needed — no bloated dependencies.

### langcore-litellm — Universal Model Access

[![PyPI](https://img.shields.io/pypi/v/langcore-litellm)](https://pypi.org/project/langcore-litellm/)

Access 100+ language models through a single unified interface via [LiteLLM](https://docs.litellm.ai/docs/). OpenAI, Anthropic, Azure, Google, Mistral, Groq, Cohere, HuggingFace, Ollama, vLLM, and more.

- Native async with `asyncio.Semaphore` concurrency control
- Multi-pass cache bypass (fresh responses on repeat passes)
- Token usage tracking (`UsageStats`)
- Full parameter passthrough (temperature, top_p, timeout, etc.)

```python
result = lx.extract(text, examples=examples, model_id="litellm/anthropic/claude-sonnet-4")
```

### langcore-guardrails — Output Validation & Retry

[![PyPI](https://img.shields.io/pypi/v/langcore-guardrails)](https://pypi.org/project/langcore-guardrails/)

Wraps any LangCore model with output validation and automatic corrective retry. 7 built-in validators:

| Validator | Purpose |
|---|---|
| `JsonSchemaValidator` | Validate against JSON Schema with auto-repair |
| `RegexValidator` | Match output against regex patterns |
| `SchemaValidator` | Validate against Pydantic models (strict/lenient) |
| `ConfidenceThresholdValidator` | Reject extractions below a confidence cutoff |
| `FieldCompletenessValidator` | Ensure required fields are present and non-empty |
| `ConsistencyValidator` | Cross-check values using custom business rules |
| `GroundingValidator` | Reject hallucinated extractions via alignment quality + character coverage |

- 4 on-fail actions: `EXCEPTION`, `REASK`, `FILTER`, `NOOP`
- Validator chaining with `ValidatorChain`
- Validator registry with `@register_validator`
- Error-only correction mode to save tokens
- Batch-independent retries

```python
from langcore_guardrails import GuardrailLanguageModel, SchemaValidator, GroundingValidator, OnFailAction

guarded = GuardrailLanguageModel(
    inner=base_model, validators=[SchemaValidator(Invoice, on_fail=OnFailAction.REASK)]
)

# Post-alignment grounding check
validator = GroundingValidator(min_alignment_quality="MATCH_FUZZY", min_coverage=0.5)
passed, filtered = validator.validate_extractions(result.extractions, source_text=result.text)
```

### langcore-audit — Audit Logging

[![PyPI](https://img.shields.io/pypi/v/langcore-audit)](https://pypi.org/project/langcore-audit/)

Structured audit records for every LLM call — latency, token usage, prompt/response hashes, success/failure status.

- Pluggable sinks: Python logging, JSONL files, OpenTelemetry spans
- Thread-safe, fault-tolerant (errors never affect inference)
- Opt-in prompt/response sampling

```python
from langcore_audit import AuditLanguageModel, LoggingSink

audited = AuditLanguageModel(inner=base_model, sinks=[LoggingSink()])
```

### langcore-hybrid — Rules + LLM Fallback

[![PyPI](https://img.shields.io/pypi/v/langcore-hybrid-llm-regex)](https://pypi.org/project/langcore-hybrid-llm-regex/)

Evaluate deterministic rules (regex, Python functions) before falling back to an LLM — save 50–80% on API costs for documents with predictable patterns.

- Regex rules with named capture groups
- Callable rules for arbitrary logic
- Confidence thresholds for rule → LLM fallback
- Observability counters (`rule_hits` vs `llm_fallbacks`)

```python
from langcore_hybrid import HybridLanguageModel, RegexRule, RuleConfig

hybrid = HybridLanguageModel(
    inner=base_model,
    rule_config=RuleConfig(rules=[RegexRule(r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})")])
)
```

### langcore-dspy — Prompt Optimization

[![PyPI](https://img.shields.io/pypi/v/langcore-dspy)](https://pypi.org/project/langcore-dspy/)

Automatically optimize extraction prompts and few-shot examples using [DSPy](https://dspy.ai/). Given training data, searches for the best prompt to maximize precision and recall.

- MIPROv2 optimizer (fast, general-purpose)
- GEPA optimizer (reflective, feedback-driven)
- Persist/load optimized configs to disk
- Built-in evaluation (P/R/F1)
- `optimized_config` parameter in `lx.extract()`

```python
from langcore_dspy import DSPyOptimizer

optimizer = DSPyOptimizer(model_id="openai/gpt-4o-mini")
config = optimizer.optimize(prompt_description="...", examples=examples, train_texts=texts, expected_results=expected)
result = lx.extract(text, optimized_config=config)
```

### langcore-rag — RAG Query Parsing

[![PyPI](https://img.shields.io/pypi/v/langcore-rag)](https://pypi.org/project/langcore-rag/)

Parse natural-language queries into semantic search terms and structured metadata filters for hybrid RAG pipelines.

- Pydantic schema introspection (auto-discovers filterable fields)
- MongoDB-style operators (`$eq`, `$gte`, `$lte`, `$in`, etc.)
- Confidence scoring and human-readable explanation
- Query caching (LRU) and Jupyter-safe sync bridge

```python
from langcore_rag import QueryParser

parser = QueryParser(schema=Invoice, model_id="gemini/gemini-2.5-flash")
parsed = parser.parse("invoices over $5000 due in March 2024")
print(parsed.structured_filters)  # {"amount": {"$gte": 5000}, ...}
```

### langcore-api — Production HTTP Service

[![PyPI](https://img.shields.io/pypi/v/langcore-api)](https://pypi.org/project/langcore-api/)

Production-ready REST API wrapping the full LangCore ecosystem. Submit documents via HTTP and get structured entities back.

- FastAPI + Celery + Redis task queue
- Single and batch extraction endpoints
- Multi-tier caching (LLM response + extraction result)
- Webhook delivery with HMAC signing
- Prometheus metrics, structured logging, SSRF protection
- Full plugin integration (all of the above)
- Docker-ready with web, worker, and Flower profiles

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice INV-001 for $500", "model_id": "litellm/gpt-4o"}'
```

For detailed instructions on creating your own provider plugin, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started
with development, testing, and pull requests.

## Testing

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests
```

Or reproduce the full CI matrix locally with tox:

```bash
tox
```

### Ollama Integration Testing

If you have Ollama installed locally, you can run integration tests:

```bash
# Test Ollama integration (requires Ollama running with gemma2:2b model)
tox -e ollama-integration
```

## Development

### Code Formatting

```bash
# Auto-format all code
./autoformat.sh

# Or run formatters separately
isort langcore tests --profile google --line-length 80
pyink langcore tests --config pyproject.toml
```

### Pre-commit Hooks

```bash
pre-commit install  # One-time setup
pre-commit run --all-files  # Manual run
```

### Linting

```bash
pylint --rcfile=.pylintrc langcore tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full terms.

This project includes code originally developed by Google LLC as [LangCore](https://github.com/google/langextract). See [NOTICE](NOTICE) for attribution details.
