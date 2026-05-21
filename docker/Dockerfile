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
# ADR-0009 (single image, two entrypoint paths), ADR-0008 (CI-driven build),
# ADR-0012 (enterprise network constraints) carry the locked decisions.
#
# Architecture strategy:
#   The JFrog CUDA images (com/db/awp/cuda/*) are arm64-only → unusable for
#   amd64 Dataflow workers. Instead we use the Beam SDK image (proven amd64)
#   as the base. vLLM/torch wheels bundle their own CUDA runtime libs
#   (libcudart, libnccl, etc.); the NVIDIA *driver* is provided at runtime
#   by Dataflow's `install-nvidia-driver` experiment (mounted at
#   /usr/local/nvidia/). No system CUDA install needed in the image.
#
# Base images:
#   Beam SDK: dkr-public-local/dkr-io/apache/beam_python3.11_sdk:2.71.0
#     → amd64, Debian-based, Python 3.11, /opt/apache/beam/boot included
#     → apache-beam pinned ==2.71.0 to match (see ADR 0012)
#   Launcher: gcr.io/dataflow-templates-base/python311-template-launcher-base
#     → only /opt/google/dataflow/python_template_launcher is copied from it
# ---------------------------------------------------------------------------

ARG ARTIFACTORY_HOSTNAME_ARG="artifactory.sdlc.ctl.gcp.db.com"
ARG BEAM_SDK_IMAGE="${ARTIFACTORY_HOSTNAME_ARG}/dkr-public-local/dkr-io/apache/beam_python3.11_sdk:2.71.0"
ARG LAUNCHER_IMAGE="gcr.io/dataflow-templates-base/python311-template-launcher-base"

# ---------------------------------------------------------------------------
# Stage 1 — Flex Template launcher (just need the binary).
# ---------------------------------------------------------------------------
FROM ${LAUNCHER_IMAGE} AS flex_launcher

# ---------------------------------------------------------------------------
# Stage 2 — Final image: Beam SDK base (amd64, Python 3.11, has /opt/apache/beam).
# ---------------------------------------------------------------------------
FROM ${BEAM_SDK_IMAGE}

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

# Overlay Flex Template launcher from GCR image.
# /opt/apache/beam is already in the Beam SDK base — no need to copy it.
COPY --from=flex_launcher /opt/google/dataflow/python_template_launcher /opt/google/dataflow/python_template_launcher

# Bootstrap uv from the JFrog pypi mirror.
RUN pip install --no-cache-dir \
        --index-url "https://${ARTIFACTORY_HOSTNAME_ARG}/artifactory/api/pypi/pypi-all/simple" \
        uv

WORKDIR /workspace

# Layer-cache strategy: pyprojects + lockfile first.
COPY pyproject.toml uv.lock README.md ./
COPY packages/sdfb-core/pyproject.toml packages/sdfb-core/pyproject.toml
COPY packages/sdfb-beam/pyproject.toml packages/sdfb-beam/pyproject.toml
COPY packages/sdfb-tests/pyproject.toml packages/sdfb-tests/pyproject.toml

# Install all deps. vLLM/torch wheels include bundled CUDA runtime libs —
# no system CUDA needed. The NVIDIA driver is mounted by Dataflow at runtime.
# FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE intentionally UNSET so the launcher
# skips re-install and uses this pre-baked venv.
RUN uv sync --frozen --no-dev --all-packages \
        --extra gpu --extra embedding --extra library

# Source last (changes here don't invalidate the deps layer).
COPY packages/sdfb-core/src packages/sdfb-core/src
COPY packages/sdfb-beam/src packages/sdfb-beam/src

# Runtime config read at launch from WORKDIR /workspace (run_pipeline defaults
# --thresholds_uri to config/thresholds.yml). Also ships the bq_schemas.
COPY config config

ENV PYTHONPATH="/workspace/.venv/lib/python3.11/site-packages:/workspace/packages/sdfb-core/src:/workspace/packages/sdfb-beam/src"

# NVIDIA driver mount point (populated by Dataflow's install-nvidia-driver).
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/nvidia/bin:${PATH}"

# Flex Template launcher contract.
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/workspace/packages/sdfb-beam/src/sdfb_beam/cli/run_pipeline.py"

ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]
