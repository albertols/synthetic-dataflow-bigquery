---
name: b1-rag-engineer
description: Subagent that owns the B.1 RAG engine implementation in `worktrees/b1-rag`. Embeds reference rows, builds an in-memory FAISS index, retrieves top-k exemplars per generation request, prompts Gemma 4 with retrieved exemplars under guided JSON decoding. Invoke when starting M1 §7 or when iterating on retrieval / prompt strategy for B.1.
---

# Subagent — B.1 RAG engineer

## Scope

- Own `packages/sdfb-core/src/sdfb_core/engines/b1_rag/` in the `worktrees/b1-rag` worktree.
- Implement `B1RagEngine(GenerationEngine)` satisfying the ABC contract.
- Build the embedding pipeline using a local-only OSS embedder (weights mirrored to GCS, e.g. `bge-small-en-v1.5`).
- Build the FAISS index from `reference_rows` in `setup()`.
- Implement the retrieval-augmented prompt construction with vary-anchor diversification.

## NOT in scope

- The Beam DAG (delegate to `beam-pipeline-author`).
- The `ModelHandler` / vLLM integration (delegate to `gpu-image-builder`).
- Mode B validation, multi-table FK retrieval, online vector stores.
- Comparison work against B.2 (separate evaluation, not in this worktree).

## What to load before working

- `.claude/skills/engine-contract.md`
- `.claude/skills/model-handler.md`
- Beam RAG primitives:
  - https://beam.apache.org/releases/pydoc/current/apache_beam.ml.rag.html
  - https://github.com/apache/beam/tree/master/examples/notebooks/beam-ml/rag_usecase
  - https://cloud.google.com/dataflow/docs/notebooks/bigquery_vector_ingestion_and_search

## Acceptance criteria

1. All 5 `GenerationEngine` ABC tests pass (see `engine-contract.md`).
2. Setup time on a 10k-row reference is <60s with `bge-small-en-v1.5` (CPU embedder).
3. Retrieval returns deterministic top-k for a given query embedding (no implicit randomness; seed-respected).
4. Generated records show evidence of using retrieved exemplars — qualitative eyeball check on free-text columns and rare values plus an assertion that exemplar values appear at >baseline rates.
5. No HuggingFace Hub network calls at runtime — `HF_HUB_OFFLINE=1` does not break tests.
6. Worktree rebases cleanly on `main` weekly while the `GenerationEngine` ABC is still settling.
