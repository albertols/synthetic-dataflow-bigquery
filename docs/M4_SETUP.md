# M4 Pro onboarding — synthetic-dataflow-bigquery

End-to-end setup for picking up the project on the M4 Pro after the laptop checkpoint. Covers venv, GCP auth, PyCharm wiring, the recommended task order, and the gotchas that bite later.

Companion to `MODEL_LAYOUT.md` — that doc covers weights only; this one covers everything else.

## TL;DR

```bash
brew install uv
cd ~/IdeaProjects/synthetic-dataflow-bigquery
git pull --rebase
uv sync --group dev
uv run pytest -m "not gpu and not gcp" -q          # expect "68 passed"
gcloud auth application-default login
gcloud config set project <your-dev-project>
```

Point PyCharm at `.venv/bin/python`. Done.

## Venv recommendation — `uv`, not PyCharm's built-in

Use **`uv`** to manage the venv (matches the repo's pyproject layout — three-member uv virtual workspace with dependency groups), and **point PyCharm at the resulting venv** as its interpreter. Don't use PyCharm's "New Environment → Virtualenv" wizard — it doesn't understand the workspace.

Why uv-first:
- One `uv sync` brings all three packages + dev deps + the pinned Python into `.venv/` deterministically.
- Same command runs in CI later — zero environment skew.
- PyCharm 2024.3+ has native uv support.

## Step-by-step

### 0. Prerequisites

```bash
which git python3 gcloud   # all three should resolve
python3 --version          # 3.11 or 3.12 (pyproject requires >=3.11,<3.13)
```

Install missing tools:
- `git` — should be there on macOS.
- `python3` — `brew install python@3.12` if missing.
- `gcloud` — https://cloud.google.com/sdk/docs/install-sdk

### 1. Clone (or pull)

```bash
cd ~/IdeaProjects
git clone <repo-url> synthetic-dataflow-bigquery   # skip if already cloned
cd synthetic-dataflow-bigquery
git pull --rebase                                   # if already cloned
```

### 2. Install `uv`

```bash
brew install uv
# OR: curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version   # confirm >= 0.4
```

### 3. Bootstrap the workspace

```bash
uv sync --group dev
```

One-time ~2–3 min on first install, ~600 MB on disk. Installs `sdfb-core` + `sdfb-beam[gcp]` + `sdfb-tests` + dev tools (ruff, mypy, pytest, hypothesis). Creates `.venv/` at the repo root.

To install the optional extras later (only needed when you start working on the GPU / RAG / library pieces):

```bash
uv sync --group dev --extra gpu --extra embedding --extra library
```

Pulls vLLM + torch + faiss + sdgx — much larger, only worth it when you start §6 / §7 / §9.

### 4. Sanity check — the 68 laptop tests must pass on the M4

```bash
uv run pytest -m "not gpu and not gcp" -q
```

Expected last line: `68 passed in ~25s`.

If any test fails: paste the last ~30 lines back — that's the fastest signal something diverges between the laptop and M4 environments.

### 5. PyCharm wiring (one-time)

1. Settings → Project → Python Interpreter → Add Local Interpreter.
2. Choose the "uv" tab → "Existing environment" → select `<repo>/.venv/bin/python`.
3. In the Project view, right-click `packages/sdfb-tests/tests` → Mark Directory As → "Test Sources Root".
4. (Optional) Right-click `packages/sdfb-core/src` and `packages/sdfb-beam/src` → Mark Directory As → "Sources Root". PyCharm then resolves imports correctly without the `pythonpath` hack.

### 6. GCP auth

```bash
gcloud auth application-default login    # browser flow
gcloud config set project <your-dev-project>
gcloud auth list                          # confirm active account
gcloud config get-value project           # confirm project
```

Application Default Credentials are what `google-cloud-bigquery` and `apache_beam[gcp]` pick up automatically — no env var plumbing needed.

## What to send back from the M4

After each step, paste these into chat (terse is fine):

1. **After step 2**: `uv --version` output (one line).
2. **After step 3**: last 5 lines of `uv sync --group dev` output, especially if it errored.
3. **After step 4**: the pytest summary line. If anything fails: the last ~30 lines.
4. **After step 6**: output of `gcloud config get-value project` (so we can use the right project ID in subsequent commands).

Once those four come back green, we pick the next task.

## Recommended task order from here

Three M4-doable tasks are unblocked. Suggested order with rationale:

### Step A — Extract a real `_ddl.json` (~10 min)

Sanity-checks the laptop → M4 transfer and confirms the canonical schema shape against a real table, not just our fixture mocks.

```bash
uv run python scripts/extract_ddl.py \
    --project "$(gcloud config get-value project)" \
    --dataset <some_small_dev_dataset> \
    --table <some_small_table> \
    --runner DirectRunner
```

Output lands in `./output/<dataset>/ddl_metadata_<dataset>_<table>.json`.

Paste back the `head -50` of the file (or the full JSON if small). If `TableSchema.model_validate()` accepts it cleanly on the M4, the whole DDL extractor chain is verified against real BigQuery — not just mocked clients.

### Step B — pick one of #10 (`Dockerfile.gpu`) or #6 (B.2 library spike)

**Recommend #10 first.** The GPU container is the longest pole — once it builds and a 1-row probe job succeeds on a real L4 worker, every other M4 task gets faster to iterate.

**#6 (sdgx vs DataDreamer bake-off)** is technically laptop-doable, but realistic fit timing needs the M4. Run it after #10 if you want to interleave.

### Step C — #9 (vLLM `ModelHandler`)

Depends on having weights in GCS, so the prerequisite chain is:

1. Pull Gemma 4 E4B per `MODEL_LAYOUT.md` § "How to download".
2. `gsutil -m cp -r` weights to `gs://<project>-models/gemma4/e4b/v1/`.
3. Then #9.

### Step D — #11 (end-to-end on Dataflow) + #12 (thresholds + `validation_runs` table)

After #9 + #10 are stable. This is the M1 finish line.

## Gotchas worth flagging now

- **`apache-beam[gcp]` on M4 ARM**: works fine for DirectRunner and for *submitting* Dataflow jobs. Dataflow workers run in their own Linux/x86 container regardless of where the submitter runs, so the M4 architecture is irrelevant to job execution. Just don't try to *locally simulate* GPU workers on the M4.

- **L4 GPU quota** (before #10 / #11): verify quota *before* you build the container — quota requests can take days.

  ```bash
  gcloud compute project-info describe \
      --project="$(gcloud config get-value project)" \
      --flatten='quotas[]' \
      --format='table(quotas.metric,quotas.limit,quotas.usage)' \
    | grep -iE 'l4|gpu|nvidia'
  ```

  If the limit for the L4 metric is 0 in your target region, file a quota request via the Cloud Console (IAM & Admin → Quotas).

- **Region selection**: pick a region where `g2-standard-*` machine types and L4 GPUs exist. Safe defaults: `europe-west4`, `us-central1`. The Dataflow GPU support matrix changes — verify at submission time via the docs in `.claude/skills/gpu-dockerfile.md`.

- **vLLM is CUDA-only** — don't try to run vLLM on the M4's GPU. For real local inference on M4 the practical backends are MLX / llama.cpp / Ollama (see `MODEL_LAYOUT.md` § "Apple Silicon caveat"). For M1, `FakeModelClient` on M4 + `VLLMModelClient` on Dataflow is the complete path; a local MLX backend is a stretch goal.

- **Kaggle CLI auth**: needed for downloading Gemma weights. Drop `~/.kaggle/kaggle.json` (from Kaggle Settings → API) with `chmod 600`. The license must be accepted in the browser once per model — `pip install kaggle` doesn't help if you haven't clicked "Accept" on https://kaggle.com/models/google/gemma-4.

- **The `whylogs` warning the laptop venv showed**: was an artifact of using `--no-deps` to skip heavy packages. On the M4 with the full `uv sync`, whylogs installs cleanly — no warning.

## What lives where (quick map)

```
CLAUDE.md                         the session-spanning project context
docs/MODEL_LAYOUT.md              where model weights live
docs/M4_SETUP.md                  THIS FILE
.claude/skills/*.md               recipe cards (load on demand)
.claude/agents/*.md               sub-agent definitions
packages/sdfb-core/               pure-Python contracts + ABC + codegen
packages/sdfb-beam/               Beam pipeline + DoFns + DDL extractor
packages/sdfb-tests/              68 unit + integration tests
scripts/extract_ddl.py            DDL extraction CLI shim
config/                           thresholds.yml + models.yml (M1 §12 — TBD)
docker/                           Dockerfile.gpu + entrypoint (M1 §10 — TBD)
```

When in doubt: `CLAUDE.md` + `memory/MEMORY.md` in the project memory directory carry the locked decisions from planning. Re-read them if anything ever feels ambiguous.
