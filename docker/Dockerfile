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

# Install into the base image's GLOBAL Python env (/usr/local), NOT a venv.
# Beam >= 2.44 *workers* build their own virtualenv from the global env when the
# custom container starts and DISCARD any venv the image created — so a
# /workspace/.venv + PYTHONPATH bridge is invisible to the worker SDK harness and
# DoFn unpickling dies with `ModuleNotFoundError: No module named 'sdfb_core'`.
# Google: "install dependencies into the default (global) Python environment
# instead of creating a virtual environment" (Build custom container images for
# Dataflow). `--inexact` (below) keeps the base image's pre-installed
# apache-beam + /opt/apache/beam/boot deps — uv installs/updates only what the
# lockfile manages and never prunes the rest.
ENV UV_PROJECT_ENVIRONMENT=/usr/local

# Layer-cache strategy: pyprojects + lockfile first.
COPY pyproject.toml uv.lock README.md ./
COPY packages/sdfb-core/pyproject.toml packages/sdfb-core/pyproject.toml
COPY packages/sdfb-beam/pyproject.toml packages/sdfb-beam/pyproject.toml
COPY packages/sdfb-tests/pyproject.toml packages/sdfb-tests/pyproject.toml

# vLLM/torch wheels bundle their own CUDA runtime libs — no system CUDA needed;
# the NVIDIA driver is mounted by Dataflow at runtime. The workspace members are
# installed editable here (finders written now, src COPY'd below) so the launcher
# and the worker's system-site-packages venv both resolve them from /usr/local.
# FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE stays UNSET so the launcher reuses this
# pre-baked env.
RUN uv sync --frozen --no-dev --all-packages --inexact \
        --extra gpu --extra embedding --extra library

# Source last (changes here don't invalidate the deps layer).
COPY packages/sdfb-core/src packages/sdfb-core/src
COPY packages/sdfb-beam/src packages/sdfb-beam/src

# Runtime config read at launch from WORKDIR /workspace (run_pipeline defaults
# --thresholds_uri to config/thresholds.yml). Also ships the bq_schemas.
COPY config config

# No PYTHONPATH bridge: sdfb-core/sdfb-beam are editable-installed into the
# global env (/usr/local) above, so /usr/local/lib/python3.11/site-packages —
# always on sys.path for both the launcher and the worker's boot venv — resolves
# them. A PYTHONPATH/venv bridge is explicitly unreliable on workers (see the
# `uv sync` note above), which is what caused the worker ModuleNotFoundError.

# NVIDIA driver mount point (populated by Dataflow's install-nvidia-driver).
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/nvidia/bin:${PATH}"

# Flex Template launcher contract.
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/workspace/packages/sdfb-beam/src/sdfb_beam/cli/run_pipeline.py"

# Worker harness image for Runner v2. The Flex Template's `--image` sets only
# the LAUNCHER; the worker harness image must be set separately and a Flex
# Template can't bake a default (Google "Configure Flex Templates"). So the
# image self-describes its own pushed coordinate and run_pipeline.py applies it
# (WorkerOptions.sdk_container_image) — keeping the sha-free template + DAG
# untouched. Without it, workers boot the stock Beam SDK container (no
# sdfb_core) and DoFn unpickling fails: ModuleNotFoundError: 'sdfb_core'.
# Workflow 1 passes `$IMG:$RELEASE_VERSION`. Declared late so this per-commit
# value never invalidates the cached `uv sync` layer above.
ARG SDFB_SDK_CONTAINER_IMAGE_ARG=""
ENV SDFB_SDK_CONTAINER_IMAGE=$SDFB_SDK_CONTAINER_IMAGE_ARG

ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]
