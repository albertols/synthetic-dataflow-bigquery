---
name: gpu-dockerfile
description: Recipe for the L4 GPU Dataflow custom container — CUDA base + Beam SDK overlay + Flex Template launcher + vLLM, with model warm-pull from GCS at worker startup (not container build). Load when working on `docker/Dockerfile` or the CI workflows that build it.
---

# Skill — GPU Dockerfile

> **Source of truth** for the actual image is **`docker/Dockerfile`** in this repo. The CI build/push pipeline is **`.github/workflows/1_build_python_beam.yaml`** and the operational runbook is **[`docs/CICD.md`](../../docs/CICD.md)**. This skill is a recipe for working on those files — not a copy of their content.

## The two-contract trick

Dataflow runs the same image in two different ways depending on what called it:

- **Flex Template launch** → uses the image's `ENTRYPOINT`, which is `/opt/google/dataflow/python_template_launcher`. The launcher reads `FLEX_TEMPLATE_PYTHON_PY_FILE` (set to `sdfb_beam.cli.run_pipeline`) and submits the job.
- **Dataflow workers** → invoked with an explicit `--entrypoint=/opt/apache/beam/boot` override; image ENTRYPOINT is irrelevant. `boot` just needs to exist at that path.

Both binaries are copied into the final image via multi-stage `COPY --from`. See [ADR 0009](../../docs/adr/0009-single-flex-template-image.md) for the rationale.

## Launch flags

```bash
# REF: https://docs.cloud.google.com/dataflow/docs/gpu/use-gpus
--dataflow_service_options="worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver"
--worker_machine_type="g2-standard-8"
--experiments=use_runner_v2
--sdk_container_image="${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-python:${VERSION}"
--worker_disk_type=pd-ssd
--worker_disk_size_gb=200
--image-repository-username-secret-id="projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_USERNAME"
--image-repository-password-secret-id="projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_PASSWORD"
```

`g2-standard-8` = 1×L4 (24 GB VRAM), 8 vCPU, 32 GB RAM. Upgrade to `g2-standard-24` if NVIDIA MPS is needed for multi-process GPU sharing.

## Model warm-pull

The image is intentionally weights-free. Model weights live in `gs://{project}-models/{family}/{model}/{version}/` and are pulled once per worker lifetime inside the `ModelClient.setup()` method:

```python
# packages/sdfb-beam/src/sdfb_beam/handlers/vllm_client.py (real impl in M1 §9)
def setup(self):
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", self.model_uri, "/local-ssd/model/"],
        check=True,
    )
    from vllm import LLM
    self.llm = LLM(model="/local-ssd/model", quantization="awq", ...)
```

This keeps the image ~3 GB instead of ~15 GB with weights baked in. Worker cold-start budget: ≤ 3 min total, including this pull. See [`docs/MODEL_LAYOUT.md`](../../docs/MODEL_LAYOUT.md) for the GCS layout contract.

## NVIDIA MPS — when

Single-process per worker first. Move to MPS only if the Dataflow GPU metric `gpu_utilization` shows <60% at steady state.

REF: https://docs.cloud.google.com/dataflow/docs/gpu/use-nvidia-mps · https://docs.cloud.google.com/dataflow/docs/gpu/gpu-metrics

## Troubleshooting

If the job fails on driver install, check the satellite-images tutorial — it's the canonical working Python+GPU example:

REF: https://docs.cloud.google.com/dataflow/docs/tutorials/satellite-images-gpus

For other GPU job issues:

REF: https://docs.cloud.google.com/dataflow/docs/gpu/troubleshoot-gpus

## Current implementation

| Concern | File | Status |
|---|---|---|
| Single image (launcher + workers) | `docker/Dockerfile` | ✅ |
| Build context exclusions | `docker/.dockerignore` | ✅ |
| Flex Template parameter schema | `docker/flex_template_metadata.json` | ✅ |
| CI build + push workflow | `.github/workflows/1_build_python_beam.yaml` | ✅ |
| CI flex-template deploy | `.github/workflows/2_deploy_flex_template_python_beam.yaml` | ✅ |
| 1-row Dataflow probe (image already in JFrog) | `scripts/probe_gpu_dataflow.sh` | ✅ |
| Operational runbook | [`docs/CICD.md`](../../docs/CICD.md) | ✅ |

Per [ADR 0008](../../docs/adr/0008-ci-driven-builds.md), developers do not build images locally. The retired local-build scripts have been removed.

ADRs: [`0003`](../../docs/adr/0003-jfrog-image-registry.md), [`0004`](../../docs/adr/0004-europe-west3-region.md), [`0008`](../../docs/adr/0008-ci-driven-builds.md), [`0009`](../../docs/adr/0009-single-flex-template-image.md).

## References

- Overview: https://docs.cloud.google.com/dataflow/docs/gpu
- GPU support matrix: https://docs.cloud.google.com/dataflow/docs/gpu/gpu-support
- Develop with GPUs: https://docs.cloud.google.com/dataflow/docs/gpu/develop-with-gpus
- Use GPUs: https://docs.cloud.google.com/dataflow/docs/gpu/use-gpus
- L4-specific: https://docs.cloud.google.com/dataflow/docs/gpu/use-l4-gpus
- NVIDIA MPS: https://docs.cloud.google.com/dataflow/docs/gpu/use-nvidia-mps
- GPU metrics: https://docs.cloud.google.com/dataflow/docs/gpu/gpu-metrics
- Troubleshooting: https://docs.cloud.google.com/dataflow/docs/gpu/troubleshoot-gpus
- Tutorial: https://docs.cloud.google.com/dataflow/docs/tutorials/satellite-images-gpus
- vLLM install: https://docs.vllm.ai/en/latest/getting_started/installation.html
- Flex Templates with custom containers: https://cloud.google.com/dataflow/docs/guides/templates/configuring-flex-templates
