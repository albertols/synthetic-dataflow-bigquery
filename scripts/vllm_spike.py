#!/usr/bin/env python
"""vLLM serving spike — de-risk the Dataflow/L4 path for M1 §11.

Answers three questions WITHOUT building the Beam image or submitting a
Dataflow job. It mirrors exactly how Beam's `VLLMCompletionsModelHandler` /
`VLLMChatModelHandler` talk to vLLM (a subprocess OpenAI-compatible server +
the `openai` client), so a PASS here transfers straight to the worker.

  Q1  Does vLLM accept Gemma 4 E4B-IT? The Kaggle checkpoint is the multimodal
      `Gemma4ForConditionalGeneration` arch (see the project memory). If vLLM's
      model loader rejects it, the server exits during startup and this script
      prints the captured error.
  Q2  Can we suppress Gemma 4's chain-of-thought channel server-side? On the
      MLX path we used `enable_thinking=False` via the chat template; here we
      pass it through `extra_body={"chat_template_kwargs": {...}}`.
  Q3  Does guided JSON (`extra_body={"guided_json": schema}`, per ADR 0011)
      produce schema-conformant output?

VENUE — read this:
  Run on a CUDA box, ideally a throwaway GCP g2-standard-8 (1x L4) VM. Do NOT
  run on the M4: vLLM officially targets CUDA; Apple Silicon support is
  experimental (ADR 0010 — that's why local smoke uses MLX, not vLLM).

  On the VM:
    pip install "vllm>=0.6.0" "openai>=1.52" jsonschema   # jsonschema optional
    # copy the model dir down first, e.g.:
    #   gcloud storage cp -r gs://<project>-models/gemma4/e4b-it/v1 ./model
    python vllm_spike.py --model_path ./model

REFs:
  - docs/adr/0011-adopt-beam-vllm-model-handler.md  (Beam handler choice)
  - docs/adr/0012-enterprise-image-build.md         (enterprise build)
  - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
  - https://docs.vllm.ai/en/latest/usage/structured_outputs.html
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

# Representative single-table schema — mixed types, an enum, a date string.
# Self-contained so the spike runs on a bare VM with no workspace install.
# Swap in the real 67-column schema with --schema_json once Q1-Q3 pass.
PROBE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "account_id": {"type": "integer"},
        "customer_name": {"type": "string"},
        "country_code": {"type": "string"},
        "currency_iso": {"type": "string"},
        "balance": {"type": "number"},
        "is_active": {"type": "boolean"},
        "opened_date": {"type": "string"},
        "status": {"type": "string", "enum": ["OPEN", "CLOSED", "FROZEN"]},
        "uuid": {"type": "string"},
    },
    "required": [
        "account_id", "customer_name", "country_code", "currency_iso",
        "balance", "is_active", "opened_date", "status", "uuid",
    ],
}

PROMPT = (
    "Generate ONE synthetic row for a bank account table. Use realistic but "
    "fictitious values. Respond with a single JSON object and nothing else."
)

# Markers that betray Gemma 4's chain-of-thought channel leaking into output.
THINKING_MARKERS = ("<|channel>", "<channel|>", "channel>thought")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM serving spike for §11")
    p.add_argument("--model_path", required=True,
                   help="Local path to the model dir (HF layout safetensors).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--quantization", default=None,
                   help="e.g. awq for the 26B-A4B-AWQ build; omit for FP16 E4B.")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--chat_template", default=None,
                   help="Path to chat_template.jinja if vLLM doesn't auto-pick it.")
    p.add_argument("--trust_remote_code", action="store_true",
                   help="Set if vLLM needs the checkpoint's custom modeling code.")
    p.add_argument("--startup_timeout", type=int, default=420,
                   help="Seconds to wait for the server to load the model.")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--schema_json", default=None,
                   help="Path to a JSON Schema file; defaults to the built-in probe schema.")
    return p.parse_args(argv)


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.chat_template:
        cmd += ["--chat-template", args.chat_template]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    return cmd


def start_server(cmd: list[str], log_tail: collections.deque) -> subprocess.Popen:
    print("[spike] launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def drain() -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="backslashreplace").rstrip()
            log_tail.append(line)
            print("    [vllm]", line, flush=True)

    threading.Thread(target=drain, daemon=True).start()
    return proc


def wait_until_ready(client, proc, timeout: int) -> str | None:
    """Poll /v1/models. Returns the served model id, or None if the server died."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:  # server exited during startup
            return None
        try:
            models = client.models.list().data
            if models:
                return models[0].id
        except Exception:  # server not up yet
            pass
        time.sleep(5)
    return None


def diagnose_startup_failure(log_lines: list[str]) -> str:
    """Pattern-match the captured server log to a likely cause. The server can
    die for many reasons BEFORE the architecture is even loaded (config parsing,
    OOM, CPU-only) — don't assume 'architecture rejected'."""
    blob = "\n".join(log_lines).lower()
    hints: list[str] = []
    if "platform cpu" in blob:
        hints.append(
            "vLLM ran in CPU mode (no CUDA detected) — NOT the L4 path. The "
            "result is not authoritative for GPU; rerun on a CUDA box."
        )
    if "rope_scaling" in blob or "rope_type" in blob or "rope_parameters" in blob:
        hints.append(
            "Config-parsing error on the model's ROPE settings — this vLLM "
            "version does not understand the model's rope schema (Gemma 4 uses "
            "nested `rope_parameters`, not a flat `rope_scaling.rope_type`). "
            "This is a vLLM-version support gap, NOT an architecture rejection, "
            "and it fails identically on GPU. Fix: a vLLM build with explicit "
            "Gemma 4 support, or try `--model-impl transformers`."
        )
    if "is not supported" in blob or "are not supported" in blob or "no model" in blob:
        hints.append("vLLM reports the model architecture itself is unsupported.")
    if "out of memory" in blob or "cuda out of memory" in blob:
        hints.append("Out of memory — lower --gpu_memory_utilization / --max_model_len.")
    if not hints:
        hints.append("Unrecognized startup error — read the traceback above.")
    return "\n".join(f"  - {h}" for h in hints)


def detect_thinking(text: str, message) -> bool:
    if any(m in text for m in THINKING_MARKERS):
        return True
    # Some vLLM reasoning parsers split CoT into a separate field.
    return bool(getattr(message, "reasoning_content", None))


def validate_against_schema(payload: dict, schema: dict) -> tuple[bool, str]:
    try:
        import jsonschema  # noqa: PLC0415 — optional dep, deferred on purpose
        jsonschema.validate(payload, schema)
        return True, "jsonschema: valid"
    except ImportError:
        missing = [k for k in schema.get("required", []) if k not in payload]
        if missing:
            return False, f"missing required keys: {missing}"
        return True, "key-presence check passed (install jsonschema for full validation)"
    except Exception as e:  # jsonschema.ValidationError
        return False, f"jsonschema: {e}"


def chat(client, model_id: str, args, *, enable_thinking: bool, guided: bool):
    extra_body: dict = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    if guided:
        extra_body["guided_json"] = args.schema
        extra_body["guided_decoding_backend"] = "outlines"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    return msg, msg.content or "", elapsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.schema = (
        json.loads(Path(args.schema_json).read_text()) if args.schema_json else PROBE_SCHEMA
    )

    try:
        from openai import OpenAI  # noqa: PLC0415 — VM-only dep, friendly error if absent
    except ImportError:
        print("[spike] pip install 'openai>=1.52' on this box first.", file=sys.stderr)
        return 2

    log_tail: collections.deque = collections.deque(maxlen=60)
    proc = start_server(build_server_cmd(args), log_tail)
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")

    results: dict[str, str] = {}
    try:
        model_id = wait_until_ready(client, proc, args.startup_timeout)
        if model_id is None:
            print("\n" + "=" * 70)
            print("Q1 FAIL — vLLM server did not come up. Diagnosis:")
            print(diagnose_startup_failure(list(log_tail)))
            print("\nLast server log lines:")
            for line in log_tail:
                print("   ", line)
            print("=" * 70)
            print("If the model family is genuinely unsupported by this vLLM "
                  "build, the fallback per the memo is Qwen 2.5 7B (ADR-0002).")
            return 1

        results["Q1 arch accepted"] = f"PASS — vLLM serving '{model_id}'"

        # Q2 — does enable_thinking=False suppress the CoT channel?
        _, base_text, _ = chat(client, model_id, args, enable_thinking=True, guided=False)
        msg_off, off_text, _ = chat(client, model_id, args, enable_thinking=False, guided=False)
        base_think = detect_thinking(base_text, msg_off)
        off_think = detect_thinking(off_text, msg_off)
        results["Q2 thinking suppressible"] = (
            f"PASS — thinking present default={base_think}, with enable_thinking=False={off_think}"
            if not off_think else
            f"CHECK — still detected with enable_thinking=False (default={base_think}). "
            "May need a different chat_template kwarg or --chat-template override."
        )

        # Q3 — guided JSON conformance on the production path.
        _, gtext, gsecs = chat(client, model_id, args, enable_thinking=False, guided=True)
        try:
            payload = json.loads(gtext)
            ok, detail = validate_against_schema(payload, args.schema)
            results["Q3 guided JSON valid"] = (
                f"{'PASS' if ok else 'FAIL'} — {detail} ({gsecs:.1f}s)"
            )
        except json.JSONDecodeError as e:
            results["Q3 guided JSON valid"] = f"FAIL — output not JSON: {e}; raw[:200]={gtext[:200]!r}"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    print("\n" + "=" * 70)
    print("vLLM spike summary")
    print("=" * 70)
    for q, verdict in results.items():
        print(f"  {q:28} {verdict}")
    print("=" * 70)
    all_pass = all(v.startswith("PASS") for v in results.values())
    print("RESULT:", "ALL PASS — §9/§11 vLLM path is clear." if all_pass
          else "Review the non-PASS lines above before committing to §11.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
