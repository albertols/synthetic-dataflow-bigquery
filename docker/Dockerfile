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
# Stage 1 — Beam SDK source layer (Python 3.11). Pinned to match the laptop
# venv (apache-beam 2.73.0).
# ---------------------------------------------------------------------------
FROM apache/beam_python3.11_sdk:2.73.0 AS beam_sdk

# ---------------------------------------------------------------------------
# Stage 2 — Python Flex Template launcher source layer.
# ---------------------------------------------------------------------------
FROM gcr.io/dataflow-templates-base/python311-template-launcher-base AS flex_launcher

# ---------------------------------------------------------------------------
# Stage 3 — CUDA + cuDNN runtime base, with Python 3.11, gcloud (for the
# per-worker model warm-pull from GCS), uv-installed workspace.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# System dependencies (Python 3.11, gcloud CLI, CA certs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ca-certificates curl gnupg apt-transport-https \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Overlay Beam SDK worker bits + Flex Template launcher.
COPY --from=beam_sdk      /opt/apache/beam                                /opt/apache/beam
COPY --from=flex_launcher /opt/google/dataflow/python_template_launcher   /opt/google/dataflow/python_template_launcher

RUN pip install --no-cache-dir uv

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
