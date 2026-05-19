---
name: gpu-dockerfile
description: Recipe for the L4 GPU Dataflow custom container — CUDA base + Beam SDK overlay + vLLM, with model warm-pull from GCS at worker startup (not container build). Load when working on `docker/Dockerfile.gpu` or the deploy scripts.
---

# Skill — GPU Dockerfile

Dataflow runs custom containers for Python jobs. The L4 GPU path requires CUDA 12 + the Beam Python SDK worker bits + vLLM, plus a runtime model warm-pull from GCS at worker startup. Container size matters for cold-start latency — keep weights OUT of the image.

## File: `docker/Dockerfile.gpu`

Skeleton (write the real file when picking up this skill; pin versions before building):

```dockerfile
# REF: https://docs.cloud.google.com/dataflow/docs/gpu/use-l4-gpus
# REF: https://docs.cloud.google.com/dataflow/docs/gpu/develop-with-gpus
# REF: https://docs.cloud.google.com/dataflow/docs/guides/build-container-image

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Overlay Beam Python SDK worker bits onto the CUDA base.
COPY --from=apache/beam_python3.11_sdk:2.60.0 /opt/apache/beam /opt/apache/beam

RUN pip install --no-cache-dir uv
WORKDIR /workspace
COPY pyproject.toml uv.lock packages/ ./
RUN uv sync --frozen --no-dev \
    --package sdfb-beam \
    --extra gpu --extra embedding --extra library

# Hard-block any HF Hub network call — weights come from GCS only.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Runner v2 is required for Python ML inference paths on Dataflow.
ENV BEAM_PYTHON_SDK_OPTIONS="--experiments=use_runner_v2"

ENTRYPOINT ["/opt/apache/beam/boot"]
```

## Launch flags

```bash
# REF: https://docs.cloud.google.com/dataflow/docs/gpu/use-gpus
--dataflow_service_options="worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver"
--worker_machine_type="g2-standard-8"
--experiments=use_runner_v2
--sdk_container_image="${ARTIFACTORY}/sdfb-gpu:${VERSION}"
--worker_disk_type=pd-ssd
--worker_disk_size_gb=200
```

`g2-standard-8` = 1×L4 (24GB VRAM), 8 vCPU, 32GB RAM. Upgrade to `g2-standard-24` if NVIDIA MPS is needed for multi-process GPU sharing.

## Model warm-pull

Done in `DoFn.setup()`, not the container:

```python
def setup(self):
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", self.model_uri, "/local-ssd/model/"],
        check=True,
    )
    from vllm import LLM
    self.llm = LLM(model="/local-ssd/model", quantization="awq", ...)
```

This keeps the container ~3GB instead of ~15GB with weights baked in. Worker cold-start budget: ≤ 3 min total, including this pull.

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
| 1-row Dataflow probe (image already in JFrog) | `scripts/probe_gpu_dataflow.sh` | ✅ |
| CI build workflow | `.github/workflows/1_build_python_beam.yaml` | ✅ |
| Operational runbook | [`docs/CICD.md`](../../docs/CICD.md) | ✅ |

Build/push scripts that used to live on the M4 (`build_gpu_image.sh`, `push_gpu_image.sh`) were retired per [ADR 0008](../../docs/adr/0008-ci-driven-builds.md). CI is the only image publisher.

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
