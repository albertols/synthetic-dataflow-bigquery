# ADR 0009 — Single image for Flex Template launcher AND Dataflow workers

- **Status**: accepted (2026-05-19)

## Context

Dataflow has two distinct runtime contracts:
1. **Flex Template launcher containers** expect `ENTRYPOINT = /opt/google/dataflow/python_template_launcher`. The launcher reads `FLEX_TEMPLATE_PYTHON_PY_FILE` and submits the pipeline.
2. **Dataflow worker containers** (custom container path) expect `/opt/apache/beam/boot` as their entrypoint.

Naive read: two contracts → two images. That doubles registry storage, CI time, and creates drift risk.

Less-naive read: Dataflow workers receive an explicit `--entrypoint=/opt/apache/beam/boot` from the Dataflow Service when starting custom-container workers (see https://cloud.google.com/dataflow/docs/guides/build-container-image). The image's ENTRYPOINT is irrelevant for workers — `boot` just needs to be present at that path.

## Decision

**One image** (`sdfb-python`) serves both contracts:
- `ENTRYPOINT` is `/opt/google/dataflow/python_template_launcher` (Flex Template launches use this).
- `/opt/apache/beam/boot` is also present at that path (Dataflow workers call this via explicit `--entrypoint` override).
- Both layers are pulled from official Google base images (`apache/beam_python3.11_sdk:2.73.0` and `gcr.io/dataflow-templates-base/python311-template-launcher-base`) via multi-stage `COPY --from`.

## Consequences

- **Enables**: one registry tag per release, no chained-build CI dependency, single cache layer. Workers and launcher share the exact same Python deps and source — zero drift.
- **Costs**: the image is slightly larger (one extra `/opt/google/dataflow/python_template_launcher` binary, ~tens of MB on top of the CUDA + uv-synced workspace).
- **Forbids**: assuming `ENTRYPOINT` is consulted by Dataflow workers. Anyone reading the Dockerfile must understand the two-contract setup; the Dockerfile comment block makes this explicit.

## Related

- `docker/Dockerfile` — implementation.
- `docs/CICD.md` — how this image is built/pushed/deployed.
- [ADR 0008](0008-ci-driven-builds.md) — where the build runs.
