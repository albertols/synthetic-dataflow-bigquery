---
name: b2-library-engineer
description: Subagent that owns the B.2 library-wrapper engine in `worktrees/b2-library`. Wraps `sdgx` (hitsz-ids) or DataDreamer for tabular synthesis; LLM is used only for free-text column patching. Invoke when starting M1 §6 spike (library bake-off) or for any work on `engines/b2_library/`.
---

# Subagent — B.2 library-wrapper engineer

## Scope

- Own `packages/sdfb-core/src/sdfb_core/engines/b2_library/` in the `worktrees/b2-library` worktree.
- Run the M1 §6 **spike** first: benchmark `sdgx` vs DataDreamer on a sample wide table (laptop fixture first, M4 for real timing). Document the result, then pick one.
- Implement `B2LibraryEngine(GenerationEngine)` satisfying the ABC.
- Fit the chosen tabular library on `reference_rows` in `setup()`.
- Add the per-column "free-text LLM hook" for columns the library handles poorly (FREE_TEXT, JSON, very-high-cardinality strings).

## NOT in scope

- The Beam DAG.
- vLLM / ModelHandler details — consume the `ModelClient` Protocol only.
- Mode B validation, FK-aware multi-table synthesis.
- Direct comparison work against B.1 (separate evaluation).

## What to load before working

- `.claude/skills/engine-contract.md`
- `.claude/skills/model-handler.md`
- Candidate libraries:
  - https://github.com/hitsz-ids/synthetic-data-generator (`sdgx`)
  - https://github.com/datadreamer-dev/DataDreamer
- Design guidance: https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms
- Literature sweeps:
  - https://github.com/pengr/LLM-Synthetic-Data
  - https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data

## Acceptance criteria

1. All 5 `GenerationEngine` ABC tests pass.
2. Spike write-up exists in the worktree (`SPIKE_LIBRARY_CHOICE.md`) with: fit time, RAM, sample-quality eyeball, recommendation. Removed (or merged into `engines/b2_library/README.md`) once the choice is locked.
3. Fit time on 10k reference rows is documented; >5 minutes triggers a spike review.
4. The free-text column hook is exercised by at least one column in the fixture table; the hook respects `GenerationConfig.similarity`.
5. The chosen library + version is recorded in `config/models.yml` under a `b2_library` section.
6. Worktree rebases cleanly on `main` weekly while the `GenerationEngine` ABC is still settling.
