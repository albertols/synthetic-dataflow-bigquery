"""vLLM-backed `ModelClient` — stub awaiting M1 §9 implementation.

This stub exists so that:
  1. `docker/Dockerfile` builds without M1 §9 being complete (the CI gate
     and the Flex Template image both need to ship before the real vLLM
     handler lands).
  2. `from sdfb_beam.handlers.vllm_client import VLLMModelClient` succeeds
     on the M4 laptop (where vllm isn't installed) — the real
     `apache_beam.ml.inference.vllm_inference` import is deferred to
     `setup()`, which only runs on a Dataflow worker with the `[gpu]`
     extra installed.

The real implementation (per ADR 0011 + ADR 0012) will:
  - Pull weights in `setup()` via the `google-cloud-storage` Python client
    (NOT `gsutil` — the CLI would need a packages.cloud.google.com apt
    install the enterprise build can't reach; ADC authenticates the client).
  - Construct a single `VLLMCompletionsModelHandler(model_name=
    "/local-ssd/model", vllm_server_kwargs={...})` per worker. Beam's
    handler owns the subprocess server lifecycle.
  - On `generate_json()`, call the handler's OpenAI client with
    `extra_body={"guided_json": schema, "guided_decoding_backend":
    "outlines"}` and parse `PredictionResult.inference.choices[0].text`.

REFs:
  - docs/adr/0011-adopt-beam-vllm-model-handler.md (handler decision)
  - docs/adr/0012-enterprise-image-build.md (JFrog bases, GCS-client pull)
  - docs/MODEL_LAYOUT.md (weight layout)
  - .claude/skills/model-handler.md (recipe)
  - https://beam.apache.org/releases/pydoc/2.60.0/apache_beam.ml.inference.vllm_inference.html
  - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-completions-api
"""

from __future__ import annotations


class VLLMModelClient:
    """Stub `ModelClient` — real impl in M1 §9."""

    def __init__(self, model_uri: str, **kwargs) -> None:
        self.model_uri = model_uri
        self._kwargs = kwargs
        self._llm = None  # vllm.LLM instance, populated by setup()

    def setup(self) -> None:
        """Per-worker init. Real impl: gsutil cp + vllm.LLM(...)."""
        raise NotImplementedError(
            "VLLMModelClient.setup() — implemented in M1 §9. "
            f"model_uri={self.model_uri!r}"
        )

    def generate_json(
        self,
        prompt: str,
        json_schema: dict,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        n: int = 1,
        seed: int | None = None,
    ) -> list[dict]:
        raise NotImplementedError(
            "VLLMModelClient.generate_json() — implemented in M1 §9."
        )
