#!/usr/bin/env bash
# Build the synthetic-dataflow-bigquery GPU worker image.
#
# Tagged with both the short git SHA (immutable) and `latest` (rolling).
# Always use SHA in production launches; `latest` is for local iteration.
#
# Required env vars (see .envrc.example or docs/GPU_CONTAINER.md):
#   ARTIFACTORY_HOSTNAME   — e.g. artifactory.intranet.db.com
#   ARTIFACTORY_NAMESPACE  — e.g. pwcclakees
#
# Usage:
#   ./scripts/build_gpu_image.sh                   # tag = git short SHA
#   ./scripts/build_gpu_image.sh v0.1.0            # tag = v0.1.0
set -euo pipefail

: "${ARTIFACTORY_HOSTNAME:?Set ARTIFACTORY_HOSTNAME in .envrc}"
: "${ARTIFACTORY_NAMESPACE:?Set ARTIFACTORY_NAMESPACE in .envrc}"

VERSION="${1:-$(git rev-parse --short HEAD)}"
IMAGE_NAME="${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu"

echo "==> Building ${IMAGE_NAME}:${VERSION}"
DOCKER_BUILDKIT=1 docker build \
    --platform=linux/amd64 \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    -f docker/Dockerfile.gpu \
    .

echo "==> Built. Images:"
docker images "${IMAGE_NAME}" | head -4

# Quick sanity: confirm boot binary + Python + uv all landed.
echo "==> In-image smoke check"
docker run --rm --entrypoint /bin/sh "${IMAGE_NAME}:${VERSION}" -c "\
    test -x /opt/apache/beam/boot && echo '  beam boot: OK' && \
    python --version && \
    uv --version && \
    python -c 'import sdfb_beam; print(\"  sdfb_beam: OK\")'"

echo
echo "==> Next: ./scripts/push_gpu_image.sh ${VERSION}"
