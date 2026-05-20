"""MLX-backed `ModelClient` for M4 Apple-Silicon local smoke tests.

This is the M4 counterpart to `VLLMModelClient` (Linux + L4 GPU). The
two satisfy the same `ModelClient` Protocol; engines never know which
one they have. See [ADR 0010](../../../../docs/adr/0010-m4-local-smoke-mlx.md)
for the rationale.

Install: `uv sync --package sdfb-beam --extra mlx` on the M4. The marker
`sys_platform == 'darwin'` in pyproject.toml prevents accidental install
on Linux. The `--package sdfb-beam` selector is required because the
`[mlx]` extra lives on the workspace member, not on the root.

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
            from pathlib import Path

            from mlx_lm import sample_utils
            from mlx_lm.utils import load_model, load_tokenizer
        except ImportError as e:
            raise ImportError(
                "mlx-lm is not installed. Run "
                "`uv sync --package sdfb-beam --extra mlx` on Apple Silicon. "
                "MLXModelClient is not available on Linux/x86."
            ) from e

        logger.info("Loading MLX model from %s", self.model_uri)
        # We deliberately bypass `mlx_lm.load()` (which forces strict=True) and
        # call the lower-level loaders with strict=False. Gemma 4 multimodal
        # checkpoints (Gemma4ForConditionalGeneration) carry redundant
        # k_proj/v_proj/k_norm weights for the shared-KV layers
        # (text_config.num_kv_shared_layers=18 in E4B → layers 24-41). mlx-lm's
        # Gemma 4 implementation correctly omits these from the model graph,
        # so they are never read at inference. The strict load just rejects
        # them on principle; relaxing it is safe.
        model_path = Path(self.model_uri)
        self._model, config = load_model(model_path, lazy=False, strict=False)
        self._tokenizer = load_tokenizer(
            model_path,
            eos_token_ids=config.get("eos_token_id"),
        )
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
        # mlx-lm ≥0.20 routes sampling params through a sampler callable
        # instead of accepting temp= directly on generate().
        sampler = self._sampler.make_sampler(temp=temperature)

        out: list[dict] = []
        for _ in range(n):
            text = generate(
                self._model,
                self._tokenizer,
                prompt=wrapped_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
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
