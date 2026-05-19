# ADR 0011 — Adopt Beam's `VLLMCompletionsModelHandler` for §9

- **Status**: accepted (2026-05-20)

## Context

M1 §9 needs a `ModelHandler` that serves Gemma 4 weights on Dataflow L4 workers and exposes them to the synthesis engines through our `ModelClient` Protocol ([ADR 0006](0006-generation-engine-abc.md)). Two viable shapes:

1. **Custom in-process handler** — import `vllm.LLM` inside the handler, call `llm.generate(prompts, sampling_params=SamplingParams(guided_decoding=GuidedDecodingParams(json=schema)))` directly. No subprocess, lowest call latency, direct access to vLLM's Python API.

2. **Beam's native `apache_beam.ml.inference.vllm_inference.VLLMCompletionsModelHandler`** — shipped since Beam 2.60.0 (we pin `apache-beam[gcp]>=2.60.0,<3.0`). The handler spawns `python -m vllm.entrypoints.openai.api_server` as a subprocess inside the worker and talks to it via the `openai` client on `http://localhost:<port>/v1`. Guided JSON via the OpenAI extension `extra_body={"guided_json": schema}` (vLLM's OpenAI server understands this).

The Beam team's Dataflow GPU notebook, the 2.60 docs, and the Beam Summit 2025 deck all converge on Option 2 — it is the documented path for vLLM-on-Dataflow.

## Decision

**Use Beam's `VLLMCompletionsModelHandler` as the in-pipeline handler.** Wrap it inside a thin `VLLMModelClient` that implements our `ModelClient` Protocol and is responsible for:

- formatting prompts with Gemma 4's tool-use template,
- building the OpenAI completion request, including `extra_body={"guided_json": schema, "guided_decoding_backend": "outlines"}` for schema conformance,
- parsing `PredictionResult.inference.choices[0].text` back into a Pydantic record,
- the Pydantic repair loop (`max_retries=2`) when guided decoding edge-cases produce malformed JSON.

The handler is constructed once per DoFn `setup()` with:

```python
VLLMCompletionsModelHandler(
    model_name="/local-ssd/model",          # local filesystem path after gsutil cp from GCS
    vllm_server_kwargs={
        "quantization": "awq",
        "max-model-len": "8192",
        "gpu-memory-utilization": "0.85",
        "max-num-seqs": "16",
        "dtype": "auto",
    },
    max_batch_size=16,
)
```

`model_name` accepting a local path is the bridge between [ADR 0001](0001-no-managed-gcp-services.md) (no HF Hub at runtime) and Beam's handler: vLLM's server CLI takes a path the same way it takes a Hub identifier.

## Consequences

- **Enables**: ~150 fewer lines of handler code; Beam owns subprocess lifecycle, port allocation, health checks, retry-on-startup-failure, and the OpenAI-compat client. We pick up Beam's `min_batch_size` / `max_batch_size` / `max_batch_duration_secs` batching knobs for free. Future Dataflow GPU improvements (driver bumps, container optimizations) land in upstream Beam first.
- **Costs**: +30–60 s subprocess warmup per worker on top of the GCS warm-pull. Guided JSON moves from the in-process `GuidedDecodingParams` API to OpenAI `extra_body` — slightly more ceremony but documented and stable. Bounded by vLLM's OpenAI-server compatibility (any breaking change there breaks us, but it would break the broader vLLM community first).
- **Forbids**: importing `vllm.LLM` directly inside engine code or handler code (engines never import vllm; the handler now only configures the subprocess). The constrained-decoding fallback chain in [`.claude/skills/model-handler.md`](../../.claude/skills/model-handler.md) is unchanged: vLLM guided JSON primary → `outlines` → Pydantic repair loop.

## Related

- `apache_beam.ml.inference.vllm_inference` — upstream module: <https://beam.apache.org/releases/pydoc/2.60.0/apache_beam.ml.inference.vllm_inference.html>
- Beam example notebook (the canonical recipe): <https://github.com/apache/beam/blob/master/examples/notebooks/beam-ml/run_inference_vllm.ipynb>
- Dataflow tutorial: <https://cloud.google.com/dataflow/docs/notebooks/run_inference_vllm>
- Beam Summit 2025 deck on serving with vLLM: <https://beamsummit.org/slides/2025/how-beam-serves-models-with-vllm.pdf>
- Benchmark page (Gemma 2B / T4 baseline): <https://beam.apache.org/performance/vllmgemmabatchtesla/>
- vLLM OpenAI-server extra parameters (where `extra_body` is documented): <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-completions-api>
- vLLM structured outputs: <https://docs.vllm.ai/en/latest/usage/structured_outputs.html>
- [ADR 0001](0001-no-managed-gcp-services.md) — local-path `model_name` is the bridge that keeps us off HF Hub.
- [ADR 0006](0006-generation-engine-abc.md) — `ModelClient` Protocol kept identical; only the implementation behind it changes.
- [ADR 0009](0009-single-flex-template-image.md) — the single image already bundles `vllm` and `apache-beam[gcp]` 2.73.0 via `[gpu]` extra.
- `.claude/skills/model-handler.md` — implementation recipe (single source of truth for the handler/client code skeleton).
- `.claude/skills/gpu-dockerfile.md` — driver pin (5xx for vLLM compatibility).