# synthetic-dataflow-bigquery — single image for BOTH the Flex Template
# launcher and the Dataflow custom-container workers.
#
# How Dataflow uses this image:
#   - Flex Template launch:  ENTRYPOINT (this file's last line) runs
#     /opt/google/dataflow/python_template_launcher, which reads
#     FLEX_TEMPLATE_PYTHON_PY_FILE and submits the job.
#   - Dataflow workers:       Dataflow Service invokes /opt/apache/beam/boot
#     with an explicit --entrypoint override, so the image's ENTRYPOINT is
#     irrelevant for workers — `boot` just needs to exist at this path.
#
# REFs:
#   https://cloud.google.com/dataflow/docs/guides/templates/configuring-flex-templates
#   https://cloud.google.com/dataflow/docs/guides/build-container-image
#   https://docs.vllm.ai/en/latest/getting_started/installation.html
#
# ADR-0009 (single image, two entrypoint paths) and ADR-0008 (CI-driven build)
# carry the locked decisions.

# ---------------------------------------------------------------------------
# Base images — ALL pulled from corporate JFrog (ARC build runners cannot
# reach Docker Hub or gcr.io). Refs are build-args so CI injects the exact
# JFrog paths; the defaults below are best-guesses — confirm the precise
# pull strings with the platform team. See ADR 0012.
#
#   CUDA:     com/db/awp/cuda  tag 12.2.2-cudnn-runtime-ubuntu22.04
#   Beam SDK: dkr-io/apache/beam_python3.11_sdk:2.71.0 (only ver in JFrog →
#             apache-beam pinned ==2.71.0 in packages/sdfb-beam/pyproject.toml)
#   Launcher: gcr.io/dataflow-templates-base/python311-template-launcher-base
#             — MIRROR to JFrog if gcr.io is unreachable from ARC runners.
# ---------------------------------------------------------------------------
ARG ARTIFACTORY_HOSTNAME_ARG="artifactory.sdlc.ctl.gcp.db.com"
ARG CUDA_IMAGE="${ARTIFACTORY_HOSTNAME_ARG}/com/db/awp/cuda:12.2.2-cudnn-runtime-ubuntu22.04"
ARG BEAM_SDK_IMAGE="${ARTIFACTORY_HOSTNAME_ARG}/dkr-io/apache/beam_python3.11_sdk:2.71.0"
ARG LAUNCHER_IMAGE="gcr.io/dataflow-templates-base/python311-template-launcher-base"

# Stage 1 — Beam SDK worker boot (version-matched to apache-beam==2.71.0).
FROM ${BEAM_SDK_IMAGE} AS beam_sdk

# Stage 2 — Python Flex Template launcher source layer.
FROM ${LAUNCHER_IMAGE} AS flex_launcher

# ---------------------------------------------------------------------------
# Stage 3 — CUDA + cuDNN runtime base + Python 3.11 + uv-installed workspace.
# The NVIDIA driver is NOT baked in: Dataflow's `install-nvidia-driver`
# installs it on the host VM and mounts it at /usr/local/nvidia/. The model
# warm-pull uses the google-cloud-storage Python client (no gcloud CLI).
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE}

ARG ENV_NAME_ARG="unset"
ARG ARTIFACTORY_HOSTNAME_ARG="unset"
ARG ARTIFACTORY_RELEASER_USERNAME_ARG="unset"
ARG ARTIFACTORY_RELEASER_PWD_ARG="unset"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    BEAM_PYTHON_SDK_OPTIONS="--experiments=use_runner_v2" \
    ENV=$ENV_NAME_ARG \
    ARTIFACTORY_HOSTNAME=$ARTIFACTORY_HOSTNAME_ARG \
    ARTIFACTORY_RELEASER_USERNAME=$ARTIFACTORY_RELEASER_USERNAME_ARG \
    ARTIFACTORY_RELEASER_PASSWORD=$ARTIFACTORY_RELEASER_PWD_ARG

# System deps from the INTERNAL apt mirror only — no PPA, no gcr.io, no
# packages.cloud.google.com. If the mirror lacks python3.11 (Ubuntu 22.04
# ships 3.10), fall back to uv-managed Python via UV_PYTHON_INSTALL_MIRROR
# pointed at a JFrog generic repo. See ADR 0012.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Overlay Beam SDK worker bits + Flex Template launcher.
COPY --from=beam_sdk      /opt/apache/beam                                /opt/apache/beam
COPY --from=flex_launcher /opt/google/dataflow/python_template_launcher   /opt/google/dataflow/python_template_launcher

# Bootstrap uv from the JFrog pypi mirror (pyproject's [[tool.uv.index]]
# resolves the rest of the dependency tree from the same mirror).
RUN pip3 install --no-cache-dir \
        --index-url "https://${ARTIFACTORY_HOSTNAME_ARG}/artifactory/api/pypi/pypi-all/simple" \
        uv

WORKDIR /workspace

# Layer-cache strategy: pyprojects + lockfile + requirements first.
COPY pyproject.toml uv.lock README.md ./
COPY packages/sdfb-core/pyproject.toml packages/sdfb-core/pyproject.toml
COPY packages/sdfb-beam/pyproject.toml packages/sdfb-beam/pyproject.toml
COPY packages/sdfb-tests/pyproject.toml packages/sdfb-tests/pyproject.toml

# `uv sync --frozen --no-dev` installs into /workspace/.venv. The Flex
# Template launcher does NOT re-install requirements at job start: we
# deliberately leave FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE *unset* so the
# launcher skips its built-in `pip install -r` step and uses the venv we
# baked in. This avoids both startup latency and double-install conflicts.
RUN uv sync --frozen --no-dev --all-packages \
        --extra gpu --extra embedding --extra library

# Source last (changes here don't invalidate the deps layer).
COPY packages/sdfb-core/src packages/sdfb-core/src
COPY packages/sdfb-beam/src packages/sdfb-beam/src

# The .venv site-packages already has sdfb-core / sdfb-beam installed via uv;
# adding the source roots is belt + suspenders for `save_main_session=False`
# pickling that the worker performs.
ENV PYTHONPATH="/workspace/.venv/lib/python3.11/site-packages:/workspace/packages/sdfb-core/src:/workspace/packages/sdfb-beam/src"

# Flex Template launcher contract.
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/workspace/packages/sdfb-beam/src/sdfb_beam/cli/run_pipeline.py"
# FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE INTENTIONALLY UNSET — see uv sync comment above.

ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]
