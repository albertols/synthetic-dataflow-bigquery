# synthetic-dataflow-bigquery — Claude Code project context

> **Always-loaded.** Keep tight. Detailed recipes live in `.claude/skills/`; specialized work belongs to sub-agents in `.claude/agents/`. Locked decisions from planning live in the project memory directory.

## Mission

Generate fictitious-but-realistic synthetic rows for a target BigQuery table, driven by its DDL plus a live reference sample. Pipeline runs on Apache Beam (Python SDK) on Google Cloud Dataflow with L4 GPU workers; LLM inference happens **inside** the DAG via `apache_beam.ml.inference.RunInference` with a custom `ModelHandler`. Two engines coexist behind one interface: B.1 (RAG) and B.2 (library-wrapper).

## Hard constraints — do not violate without explicit user confirmation

1. **No Vertex AI in the serving path.** Everything LLM happens inside Beam DoFns. If a design step is reaching for Vertex, stop and propose a self-hosted alternative on L4 workers.
2. **No HuggingFace Hub at runtime.** Weights live in `gs://{project}-models/{family}/{model}/{version}/`, pulled once on the M4. The `transformers` / `safetensors` libraries are fine as file-format readers — never call `from_pretrained("org/repo")` against the Hub. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the GPU container.
3. **No Dataplex, no Looker.** Validation results land in BigQuery `synthetic_data_quality.*` tables and GCS HTML/JSON artifacts only. Do not propose dashboards or DQ scans.
4. **No external LLM APIs.** GPT / Claude / Grok / Deepseek API calls violate the self-hosting contract. Off-pipeline benchmarking scripts are M2+ and out of M1 scope.
5. **Single-table only for M1.** Primary-key awareness is in scope; FK / multi-table joins are M2+.

Locked rationale: `~/.claude/projects/-Users-serna-IdeaProjects-synthetic-dataflow-bigquery/memory/feedback_no_managed_gcp_services.md` and `…/project_synthetic_dataflow_m1_stack.md`.

## Hardware split

- **This laptop** (no GPU, no GCP access): pure-Python work — `sdfb-core`, fake handlers, fixtures, unit tests, DirectRunner with mocked inference. Anything that imports `apache_beam[gcp]`, `vllm`, `torch`, `google.cloud.*` runs here **only with mocks**.
- **M4 Pro 24GB** (clones the repo, has GCP creds, runs Dataflow): GPU container builds, vLLM benchmarks, real Dataflow submissions, BigQuery reads, model fit timing for B.2.

Mark M4-only tests with `@pytest.mark.gpu` or `@pytest.mark.gcp`. The default `pytest` run excludes them.

## Package map

```
packages/sdfb-core/    no Beam, no GCP, no torch. Pydantic, codegen, ABC, prompt templates.
packages/sdfb-beam/    imports sdfb-core; adds apache-beam[gcp], pandera, whylogs, pandas.
                       Optional extras: [gpu] (vllm+torch), [embedding] (faiss+transformers),
                                        [library] (sdgx — B.2 candidate).
packages/sdfb-tests/   imports both; pytest + hypothesis + DirectRunner fixtures.
```

Import direction is **strict**: `sdfb-beam` depends on `sdfb-core`, never the other way. Engines live in `sdfb-core` (pure-Python); only the DoFn wrappers and ModelHandler live in `sdfb-beam`.

## What lives where in `.claude/`

`.claude/skills/` — recipe cards for recurring tasks. Load on demand.
- `engine-contract.md` — implementing the `GenerationEngine` ABC
- `beam-dofn.md` — writing/testing Beam DoFns (lifecycle, tagged outputs, side inputs, metrics)
- `model-handler.md` — RunInference + `ModelHandler` + the `ModelClient` Protocol
- `ddl-codegen.md` — Pydantic ↔ Pandera ↔ BQ DDL derivation
- `validation-mode-a.md` — three lines of defense, DLQ, whylogs merge
- `reference-data.md` — live BQ SELECT + canonical provenance digest
- `gpu-dockerfile.md` — L4 + vLLM custom-container recipe

`.claude/agents/` — bounded sub-agent definitions (use the `Agent` tool with `subagent_type` matching the file's `name:`). Each agent owns one concern; do not let them sprawl.
- `ddl-codegen-agent` — schema → contracts, pure laptop work
- `beam-pipeline-author` — DAG composition (does not write DoFns from scratch)
- `b1-rag-engineer` — owns `worktrees/b1-rag`
- `b2-library-engineer` — owns `worktrees/b2-library`
- `gpu-image-builder` — owns `docker/Dockerfile.gpu` and the vLLM handler

## Critical path (M1 §9)

1. Bootstrap (this commit)                              — laptop
2. Contracts (Pydantic → Pandera + BQ DDL codegen)      — laptop
3. DDL refactor (`bigquery_ddl_metadata.py` → core)     — laptop
4. Fake model client + fixtures                          — laptop
5. Engine ABC + tests                                    — laptop
6. B.2 engine (worktree)                                 — M4 (library fit timing)
7. B.1 engine (worktree)                                 — M4 (FAISS perf)
8. Beam DAG (DirectRunner with FakeModelClient)         — laptop
9. vLLM `ModelHandler`                                   — M4
10. `Dockerfile.gpu` + entrypoint + image push          — M4
11. End-to-end on Dataflow with Gemma 4 E4B then 26B-MoE — M4
12. Threshold tuning + `validation_runs` metadata table  — M4

§1–§5 + §8 parallelize on the laptop. §6, §7, §9–§12 require the M4.

## Anti-patterns — do not do these

- Do not import `apache_beam` inside `sdfb-core`. Engines must be testable without Beam.
- Do not write a DoFn that constructs the engine inside `process()`. Engines are heavy; build them in `setup()`.
- Do not call `WriteToBigQuery` with `STREAMING_INSERTS` for batch synthetic. Use `FILE_LOADS` (cheaper, batch-shaped).
- Do not catch `ValidationError` and silently drop. Always route to a tagged DLQ output with full error context.
- Do not add feature flags or abstractions for hypothetical future cases. Three similar lines beat a premature abstraction.
- Do not propose Vertex AI, Dataplex, Looker, OpenLineage, Marquez, Dagster, or external LLM APIs. Out of scope.

## When in doubt

Re-read `memory/MEMORY.md` (in the project memory directory). It indexes the two locked decision memos from planning.
