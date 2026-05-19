"""MLX-backed `ModelClient` for M4 Apple-Silicon local smoke tests.

This is the M4 counterpart to `VLLMModelClient` (Linux + L4 GPU). The
two satisfy the same `ModelClient` Protocol; engines never know which
one they have. See [ADR 0010](../../../../docs/adr/0010-m4-local-smoke-mlx.md)
for the rationale.

Install: `uv sync --extra mlx` on the M4. The marker `sys_platform == 'darwin'`
in pyproject.toml prevents accidental install on Linux.

REFs:
  - mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms
  - docs/M4_LOCAL_SMOKE.md
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MLXModelClient:
    """`ModelClient` impl backed by `mlx-lm` on Apple Silicon.

    Loads weights from a local directory (HF-layout safetensors). Falls back
    to a JSON-schema instruction in the prompt for structured output — MLX
    does not have native guided-JSON decoding like vLLM does, so we rely on
    Gemma 4's native function-calling training + post-validate-and-retry.

    Lifecycle:
        c = MLXModelClient(model_uri="./models/gemma4/e4b/v1/")
        c.setup()                       # downloads / loads model (~30s)
        rows = c.generate_json(prompt, schema, n=5)
    """

    def __init__(self, model_uri: str, max_tokens: int = 2048) -> None:
        # Strip gs:// scheme if present — MLX wants a local path.
        self.model_uri = model_uri.removeprefix("gs://")
        self.default_max_tokens = max_tokens
        self._model = None  # mlx_lm.Model
        self._tokenizer = None
        self._sampler = None

    def setup(self) -> None:
        """Lazy import + load. Idempotent."""
        if self._model is not None:
            return
        try:
            from mlx_lm import load, sample_utils
        except ImportError as e:
            raise ImportError(
                "mlx-lm is not installed. Run `uv sync --extra mlx` on Apple "
                "Silicon. MLXModelClient is not available on Linux/x86."
            ) from e

        logger.info("Loading MLX model from %s", self.model_uri)
        self._model, self._tokenizer = load(self.model_uri)
        self._sampler = sample_utils
        logger.info("MLX model loaded.")

    def generate_json(
        self,
        prompt: str,
        json_schema: dict,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        n: int = 1,
        seed: int | None = None,
    ) -> list[dict]:
        """Call the LLM `n` times, parse each output as JSON.

        Schema-conformance is best-effort here (no token-level grammar
        constraint). Use `mlx_lm`'s sampler with low temperature + the
        json_schema injected into the prompt; post-validate via Pydantic
        downstream and reroll on failure (engine's repair loop).
        """
        if self._model is None:
            self.setup()
        from mlx_lm import generate

        max_tokens = max_tokens or self.default_max_tokens
        wrapped_prompt = self._wrap_with_schema(prompt, json_schema)

        out: list[dict] = []
        for _ in range(n):
            text = generate(
                self._model,
                self._tokenizer,
                prompt=wrapped_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )
            payload = self._extract_first_json_object(text)
            if payload is None:
                logger.warning("MLX output not parseable as JSON; skipping.")
                continue
            out.append(payload)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_with_schema(prompt: str, json_schema: dict) -> str:
        """Embed the JSON schema in the prompt — MLX has no guided decoding."""
        return (
            f"{prompt}\n\n"
            "Respond with ONE JSON object conforming to this schema, and "
            "nothing else (no markdown, no commentary):\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            "JSON:"
        )

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        """Pull the first balanced `{…}` substring out of an LLM response."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None
