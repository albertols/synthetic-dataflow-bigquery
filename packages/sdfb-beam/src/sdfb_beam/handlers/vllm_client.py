"""vLLM-backed `ModelClient` — stub awaiting M1 §9 implementation.

This stub exists so that:
  1. `docker/Dockerfile` builds without M1 §9 being complete (the CI gate
     and the Flex Template image both need to ship before the real vLLM
     handler lands).
  2. `from sdfb_beam.handlers.vllm_client import VLLMModelClient` succeeds
     on the M4 laptop (where vllm isn't installed) — the real `import vllm`
     call is deferred to `setup()`, which only runs on a Dataflow worker
     with the `[gpu]` extra installed.

The real implementation will:
  - `gsutil -m cp -r {model_uri}/ /local-ssd/model/` in `setup()`.
  - `vllm.LLM(model="/local-ssd/model", quantization=...)` once per worker.
  - `vllm.SamplingParams(guided_decoding=GuidedDecodingParams(json=schema))`
    on every `generate_json()` call.

REFs:
  - docs/MODEL_LAYOUT.md (weight layout)
  - .claude/skills/model-handler.md
  - https://docs.vllm.ai/en/latest/usage/structured_outputs.html
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
