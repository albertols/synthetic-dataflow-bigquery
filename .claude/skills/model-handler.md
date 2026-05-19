---
name: model-handler
description: Recipe for the vLLM-backed `ModelHandler` and `ModelClient` Protocol — how `apache_beam.ml.inference.RunInference` ties into the synthesis engines without exposing vLLM internals to engine code. Load when wiring vLLM, adding a new model family, or debugging structured-output behavior.
---

# Skill — `ModelHandler` + `ModelClient`

The engines never import `vllm`. They call `ModelClient.generate_json(prompt, json_schema)`. The mapping from there to vLLM goes:

```
engine.generate_batch(n)
    └─> model_client.generate_json(prompt, schema)
            └─> RunInference with VLLMModelHandler
                    └─> vllm.LLM.generate(prompts, sampling_params=guided_json_params)
```

## Files

- `packages/sdfb-beam/src/sdfb_beam/handlers/vllm_handler.py` — `ModelHandler` subclass for vLLM (M4 only).
- `packages/sdfb-beam/src/sdfb_beam/handlers/vllm_client.py` — `ModelClient` impl invoking the handler.
- `packages/sdfb-beam/src/sdfb_beam/handlers/fake_client.py` — fixture-driven test impl (laptop).

## vLLM guided JSON

vLLM has native JSON-schema-guided decoding. Use it — do NOT rely on prompt-only JSON for production.

```python
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

guided_params = GuidedDecodingParams(json=schema_dict)
sampling = SamplingParams(
    temperature=0.7,
    max_tokens=2048,
    guided_decoding=guided_params,
)
outputs = llm.generate(prompts=[prompt], sampling_params=sampling)
```

REF: https://docs.vllm.ai/en/latest/usage/structured_outputs.html

## Gemma 4 native function calling

Gemma 4 ships with JSON-schema function calling trained into pretraining (not patched). Format prompts using Gemma's tool-use template (see https://ai.google.dev/gemma/docs/core). With guided decoding ON TOP, schema conformance should exceed 99% — we measure this as a CI gate.

REF: https://ai.google.dev/gemma/docs/core

## Model load — from GCS, not HF Hub

Worker `setup()` does:
```python
def setup(self):
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", self.model_uri, "/local-ssd/model/"],
        check=True,
    )
    from vllm import LLM
    self.llm = LLM(model="/local-ssd/model", quantization="awq", ...)
```

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the Dockerfile to make any accidental Hub call fail fast.

## Constrained-decoding fallback chain

If vLLM guided decoding hits a schema edge case (deeply recursive types, complex `oneOf`/`anyOf`), the fallback chain is:

1. **vLLM guided JSON** (primary — fastest, schema-aware at the token level).
2. **`outlines`** regex/JSON-schema generator (secondary).
3. **Pydantic repair loop** (`max_retries=2`) — re-prompt with the validation error injected into the next prompt.

Don't reach for the fallback unless you've reproduced the failure with a fixture test.

## RunInference batching

`RunInference` batches across Beam bundles via `min_batch_size`/`max_batch_size`. Tune so each vLLM call sees ≥ 4 prompts — single-prompt calls leave the GPU mostly idle. Start with `max_batch_size=16` and measure GPU utilization in Cloud Monitoring (Dataflow GPU metrics).

REF: https://docs.cloud.google.com/dataflow/docs/gpu/gpu-metrics

## Current implementation

| Concern | File | Status |
|---|---|---|
| `ModelClient` Protocol | `packages/sdfb-core/src/sdfb_core/engines/base.py` | ✅ done |
| `FakeModelClient` (canned + echo modes) | `packages/sdfb-beam/src/sdfb_beam/handlers/fake_client.py` | ✅ done |
| `FakeModelClient` tests | `packages/sdfb-tests/tests/unit/handlers/test_fake_client.py` | ✅ done |
| `VLLMModelClient` + `VLLMModelHandler` | `packages/sdfb-beam/src/sdfb_beam/handlers/vllm_*.py` | 🔒 M1 §9 |

Model registry (GCS URIs, vLLM args, license): `config/models.yml`. Layout / download procedure: [`docs/MODEL_LAYOUT.md`](../../docs/MODEL_LAYOUT.md).

## References

- vLLM structured outputs: https://docs.vllm.ai/en/latest/usage/structured_outputs.html
- vLLM install: https://docs.vllm.ai/en/latest/getting_started/installation.html
- outlines: https://github.com/outlines-dev/outlines
- lm-format-enforcer: https://github.com/noamgat/lm-format-enforcer
- Beam RunInference: https://beam.apache.org/releases/pydoc/current/apache_beam.ml.inference.base.html
- Generative AI inference on Dataflow: https://docs.cloud.google.com/dataflow/docs/notebooks/run_inference_generative_ai
- Gemma 4: https://ai.google.dev/gemma/docs/core
- JSONSchemaBench (methodology): https://arxiv.org/pdf/2501.10868