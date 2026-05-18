# Model storage layout

Where model weights live, in what format, for which runner.

## TL;DR

| Runner | Model source | Inference backend | Weight format |
|---|---|---|---|
| **DirectRunner (this laptop)** | `FakeModelClient` (no real model) | n/a — test fake | n/a |
| **DirectRunner (M4 — stretch / optional)** | `./models/{family}/{model}/{version}/` | MLX, llama.cpp, or vLLM-CPU | safetensors (or GGUF) |
| **Dataflow (L4 GPU workers)** | `gs://{project}-models/{family}/{model}/{version}/` | vLLM with CUDA | safetensors + AWQ Q4 |

**For M1 §8 specifically** (the DAG end-to-end on DirectRunner), no real model is needed — `FakeModelClient` substitutes. Everything below is for §9–§11 when real LLMs come online.

---

## Canonical GCS layout (Dataflow workers)

```
gs://{project}-models/
├── gemma4/
│   ├── e4b/v1/                  # Gemma 4 E4B (4.5B effective) — dev / cost-floor
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── model.safetensors.index.json
│   └── 26b-a4b-awq/v1/          # Gemma 4 26B A4B MoE, Q4-AWQ — primary production
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── generation_config.json
│       ├── quant_config.json
│       └── model.safetensors    # AWQ-quantized; may be sharded depending on packer
├── qwen2.5/7b-it/v1/            # Optional cross-family check (Apache-2.0)
│   └── … (same layout)
└── embedders/
    └── bge-small-en-v1.5/v1/    # For B.1 RAG (M1 §7)
        ├── config.json
        ├── tokenizer.json
        └── model.safetensors
```

Rules:
- The addressable unit is the **`{family}/{model}/{version}/` triple**. The pipeline flag is `--model_uri=gs://{project}-models/gemma4/e4b/v1/`.
- Version directories are **immutable**. New version = new directory. Never overwrite `v1/` — bump to `v2/`.
- Worker `setup()` runs **one** `gsutil -m cp -r {model_uri}/ /local-ssd/model/` per worker lifetime; vLLM loads from the local-SSD path.

## Local layout on the M4

Mirror the GCS structure under the repo's `./models/` (gitignored — see `.gitignore`):

```
~/IdeaProjects/synthetic-dataflow-bigquery/
└── models/                       # gitignored
    └── gemma4/
        ├── e4b/v1/               # ~9 GB at FP16 — fits comfortably on M4 24 GB
        └── 26b-a4b-awq/v1/       # ~13 GB at Q4-AWQ — tight but fits
```

The pipeline flag accepts both `gs://…` and local paths, so the same code path works for laptop/M4-local and Dataflow.

## Apple Silicon (M4) caveat

vLLM targets CUDA. On the M4's unified-memory GPU it will fall back to CPU (slow) — for real local inference on M4 you'd typically use one of:

- **MLX / mlx-lm** — Apple's framework, fast on M-series. Reads HuggingFace-layout safetensors directly.
- **llama.cpp / GGUF** — needs converted GGUF weights (`convert-hf-to-gguf.py`).
- **Ollama** — wraps llama.cpp.

A future `MLXModelClient` (out of M1 scope) would slot into the `ModelClient` Protocol exactly the same as `VLLMModelClient`. The Beam DAG doesn't care.

For M1 specifically, **on M4 you have two practical paths**:
1. `FakeModelClient` + DirectRunner — exercise the pipeline locally with no GPU.
2. `VLLMModelClient` on Dataflow with L4 workers — the production path (M1 §11).

A real-model DirectRunner run on M4 with MLX is a nice-to-have for §6 (B.2 spike) and §7 (B.1 spike), not a blocker.

## How to download (canonical procedure)

The non-HuggingFace source for Gemma is **Kaggle** (Google-hosted, license-clean).

1. On the M4, install the Kaggle CLI and authenticate:
   ```bash
   pip install kaggle
   # Drop ~/.kaggle/kaggle.json with your API token (Kaggle → Settings → API)
   chmod 600 ~/.kaggle/kaggle.json
   ```
2. Accept the Gemma model license once on the Kaggle model page (https://kaggle.com/models/google/gemma-4).
3. Download:
   ```bash
   mkdir -p models/gemma4/e4b/v1
   kaggle models instances versions download google/gemma-4/transformers/e4b/1 \
     -p models/gemma4/e4b/v1
   # Kaggle delivers as a zip — extract in place
   unzip models/gemma4/e4b/v1/*.zip -d models/gemma4/e4b/v1/
   rm models/gemma4/e4b/v1/*.zip
   ```
4. Verify the file-level checklist below.
5. Upload to GCS for Dataflow workers:
   ```bash
   gsutil -m cp -r models/gemma4/e4b/v1/ gs://{project}-models/gemma4/e4b/v1/
   ```

For Qwen 2.5 (not on Kaggle), use the official Qwen GitHub release tarball: https://github.com/QwenLM/Qwen2.5/releases (download → extract → same layout).

## File-level checklist

Every `{family}/{model}/{version}/` directory MUST contain, at minimum:

- `config.json` — model architecture + hyperparameters
- `tokenizer.json` (or `tokenizer.model` for SentencePiece-based tokenizers)
- `tokenizer_config.json` — tokenizer wrapper config
- `special_tokens_map.json` — BOS / EOS / PAD token ids
- One or more `*.safetensors` files — model weights
- `model.safetensors.index.json` — required when weights are sharded across multiple `*.safetensors` files

Recommended:
- `generation_config.json` — default sampling params (vLLM picks these up)
- `chat_template.jinja` or `chat_template.json` — for instruction-tuned chat models

For AWQ-quantized variants, additionally:
- `quant_config.json` — AWQ quantization parameters (`zero_point`, `q_group_size`, `w_bit`, etc.)

For GGUF (llama.cpp / Ollama path):
- Single `*.gguf` file is sufficient. No accompanying `config.json` — GGUF is self-describing. Tokenizer is embedded.

## Runtime load — Dataflow / vLLM

Inside `sdfb_beam/handlers/vllm_handler.py` (M1 §9):

```python
def setup(self):
    # One-time per worker — copy from GCS to local SSD.
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", self.model_uri, "/local-ssd/model/"],
        check=True,
    )
    from vllm import LLM
    self.llm = LLM(
        model="/local-ssd/model",
        quantization="awq",           # for the AWQ-quantized variants
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
```

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the GPU `Dockerfile.gpu` (M1 §10) so any accidental Hub call fails loudly. The model directory must be self-contained.

## Runtime load — local M4 (stretch goal, MLX example)

```python
# An MLXModelClient would look roughly like this — not in M1 scope.
def setup(self):
    from mlx_lm import load
    self.model, self.tokenizer = load(str(self.local_model_path))
```

## What NOT to do

- ❌ Do not commit model weights — they're in `.gitignore` (`models/`, `*.safetensors`, `*.gguf`, `*.bin`).
- ❌ Do not call `from_pretrained("org/repo")` against the Hub at runtime. `HF_HUB_OFFLINE=1` in production is there to make this fail loudly.
- ❌ Do not store whole-archive model blobs in BigQuery / GCR / Artifact Registry. GCS-as-a-flat-directory-prefix is the contract; vLLM expects a directory layout.
- ❌ Do not mix multiple model versions in one directory. New version = new `vN/` subdirectory.
- ❌ Do not put models inside `packages/` — they're operational artifacts, not source code. `./models/` at the repo root is the convention.
