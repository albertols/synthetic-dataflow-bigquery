# synthetic-dataflow-bigquery — roadmap

Single source of truth for milestone scope. Locked decisions per milestone live as ADRs in [`adr/`](adr/); operational runbooks live in this directory's other docs.

## M1 — laptop-spine + L4 GPU + first Dataflow E2E

**Goal**: produce schema-conformant synthetic rows for one BigQuery table on real Dataflow with L4 GPU workers, validated in-pipeline (Mode A).

| # | Task | Status | Where |
|---|---|---|---|
| 1 | Bootstrap workspace + harness | ✅ done | laptop |
| 2 | Pydantic contracts + Pandera / BQ-DDL codegen | ✅ done | laptop |
| 3 | DDL extractor `sdfb_beam/ddl/` | ✅ done | laptop |
| 4 | `FakeModelClient` + fixtures | ✅ done | laptop |
| 5 | `GenerationEngine` ABC + 5 contract tests | ✅ done | laptop |
| 6 | B.2 library-wrapper engine + spike (sdgx vs DataDreamer) | 🔒 pending | M4 |
| 7 | B.1 RAG engine | 🔒 pending | M4 |
| 8 | Beam DAG end-to-end on DirectRunner + `FakeModelClient` | ✅ done | laptop |
| 9 | vLLM `ModelHandler` + `ModelClient` | 🔒 pending | M4 |
| 10 | `Dockerfile.gpu` + image build/push | 🟡 in progress | M4 |
| 11 | E2E Dataflow run: Gemma 4 E4B → 26B-A4B MoE | 🔒 pending | M4 |
| 12 | `thresholds.yml` wiring + `validation_runs` BQ table | 🔒 pending | M4 |

**Hard constraints (immutable for M1)** — see [`adr/0001-no-managed-gcp-services.md`](adr/0001-no-managed-gcp-services.md):
- No Vertex AI, Dataplex, Looker.
- No HuggingFace Hub at runtime.
- No external LLM APIs (GPT / Claude / Grok / Deepseek).
- Single-table only; FK / multi-table is M2.

**M1 deliverables**:
- Working DAG that ingests a `_ddl.json`, pulls a live BQ reference sample, generates N rows via Gemma 4 on L4 workers, validates with Pydantic + Pandera + (optional) whylogs profile merge, writes valid rows to BigQuery via `FILE_LOADS` and invalid rows to a partitioned DLQ table.
- All laptop tests passing in CI on every push.
- Reference digest and run metadata in `synthetic_data_quality.validation_runs`.

## M2 — Mode B validation + scale + breadth

**Goal**: production-grade validation, broader use cases, foundations for managed adoption.

Themes (order TBD; targets are 4–6 weeks after M1 ships):
- **Mode B validation pipeline** — GX 1.x Checkpoint + Soda Core scan + SDMetrics fidelity (`QualityReport` + `DiagnosticReport`) + Evidently drift report. Results to `synthetic_data_quality.*` tables; HTML / JSON artifacts to GCS.
- **Multi-table mode** — FK awareness, cross-table referential checks (`fk.exists`, `logic.cross`), DAG composition for parent-first generation.
- **Constrained-decoding fallback chain** — `outlines` and `lm-format-enforcer` for schema edge cases that beat vLLM's guided JSON.
- **Reference snapshot pattern** — cached parquet under `gs://{project}-dataflow/reference/{table}/sample.parquet` as a deterministic alternative to live SELECT.
- **PII allow-list** — mask / format-template sensitive columns before they touch embeddings or validation reports.
- **B.2 spike formalization** — lock the library choice (`sdgx` vs `DataDreamer`) in `config/models.yml` after head-to-head fidelity numbers.
- **Apple Silicon MLX `ModelClient`** — drop-in for `VLLMModelClient` so M4-local runs become a full alternative to the FakeClient.
- **CI gating** — Mode B checkpoint blocks PR merges to main; thresholds.yml is the source of truth.

**Explicit non-goals for M2**:
- Dataplex / Looker dashboards (the no-managed-services rule from ADR-0001 still applies).
- OpenLineage / Marquez / Dagster.
- External LLM APIs.

## M3 — production hardening

**Goal**: operations team can adopt without us.

- **IaC / Terraform** — buckets, datasets, secrets, IAM, AR repos (if we migrate from JFrog).
- **Run metadata schema v2** — full audit trail, signed-URL report links, OpenLineage-compatible structure (without taking the OpenLineage dependency).
- **Multi-region** — at least `us-central1` parity with `europe-west3`.
- **Quota-aware autoscaling** — pipeline reads quota before requesting workers, fails gracefully.
- **B.1 / B.2 head-to-head report** — fidelity (SDMetrics), cost-per-1k-rows, throughput, suitability matrix per column shape.
- **B.3 candidates** — diffusion-based tabular or GAN-only engines if research warrants.
- **Documentation site** — render `docs/` as a static site (mdBook / Docusaurus); not in lieu of the markdown source.

## How milestones evolve

- Add new tasks to the relevant milestone's table; never silently move tasks between milestones.
- Add new locked decisions as ADRs under [`adr/`](adr/); don't sneak them into a memory file or a skill.
- When an "explicit non-goal" feels like a goal, open a discussion before changing the roadmap — and update the relevant ADR.
