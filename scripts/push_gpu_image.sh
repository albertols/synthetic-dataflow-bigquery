#!/usr/bin/env bash
# Push the GPU worker image to corporate JFrog.
#
# Required env vars:
#   ARTIFACTORY_HOSTNAME           — e.g. artifactory.intranet.db.com
#   ARTIFACTORY_NAMESPACE          — e.g. pwcclakees
#   ARTIFACTORY_RELEASER_USER      — docker login username
#   ARTIFACTORY_RELEASER_PS        — docker login password
#
# Tip: store the two RELEASER_* vars in Secret Manager and export them in
# .envrc only when you actually need to push (avoid leaving creds in env).
set -euo pipefail

: "${ARTIFACTORY_HOSTNAME:?Set ARTIFACTORY_HOSTNAME}"
: "${ARTIFACTORY_NAMESPACE:?Set ARTIFACTORY_NAMESPACE}"
: "${ARTIFACTORY_RELEASER_USER:?Set ARTIFACTORY_RELEASER_USER}"
: "${ARTIFACTORY_RELEASER_PS:?Set ARTIFACTORY_RELEASER_PS}"

VERSION="${1:-$(git rev-parse --short HEAD)}"
IMAGE_NAME="${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu"

echo "==> docker login ${ARTIFACTORY_HOSTNAME}"
echo "${ARTIFACTORY_RELEASER_PS}" | docker login --password-stdin \
    --username "${ARTIFACTORY_RELEASER_USER}" \
    "${ARTIFACTORY_HOSTNAME}"

echo "==> Pushing ${IMAGE_NAME}:${VERSION}"
docker push "${IMAGE_NAME}:${VERSION}"

echo "==> Pushing ${IMAGE_NAME}:latest"
docker push "${IMAGE_NAME}:latest"

echo
echo "==> Pushed. Image URI for Dataflow:"
echo "    --sdk_container_image=${IMAGE_NAME}:${VERSION}"
