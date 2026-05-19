#!/usr/bin/env bash
# 1-row Dataflow GPU probe — confirms the L4 worker boots from the image we
# pushed and a single Beam bundle runs to completion.
#
# Acceptance gate for M1 §10 (see .claude/agents/gpu-image-builder.md):
#   1. Image pulls without auth errors.
#   2. NVIDIA driver installs (nvidia-smi succeeds in worker startup logs).
#   3. GPU metric is non-zero in Cloud Monitoring.
#   4. Cold start < 3 min.
#
# Required env (typically from .envrc):
#   ARTIFACTORY_HOSTNAME, ARTIFACTORY_NAMESPACE,  GCP_REGION, GCP_ZONE
#
# Usage:
#   ./scripts/probe_gpu_dataflow.sh                # tag = current short SHA
#   ./scripts/probe_gpu_dataflow.sh v0.1.0         # tag = v0.1.0
set -euo pipefail

: "${ARTIFACTORY_HOSTNAME:?Set ARTIFACTORY_HOSTNAME in .envrc}"
: "${ARTIFACTORY_NAMESPACE:?Set ARTIFACTORY_NAMESPACE in .envrc}"
: "${GCP_REGION:?Set GCP_REGION in .envrc (e.g. europe-west3)}"
: "${GCP_ZONE:?Set GCP_ZONE in .envrc (e.g. europe-west3-b)}"

VERSION="${1:-$(git rev-parse --short HEAD)}"
PROJECT="$(gcloud config get-value project)"
JOB_NAME="sdfb-gpu-probe-$(date +%s)"
IMAGE="${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu:${VERSION}"

echo "==> Probe target"
echo "    Project: ${PROJECT}"
echo "    Region:  ${GCP_REGION}"
echo "    Zone:    ${GCP_ZONE}"
echo "    Image:   ${IMAGE}"
echo "    Job:     ${JOB_NAME}"

# Verify both Secret Manager entries exist; either is fatal for image pull.
for secret in ARTIFACTORY_RELEASER_USER ARTIFACTORY_RELEASER_PS; do
    if ! gcloud secrets describe "${secret}" --project="${PROJECT}" >/dev/null 2>&1; then
        echo "ERROR: Secret ${secret} missing in project ${PROJECT}." >&2
        echo "       Create with:  gcloud secrets create ${secret} --data-file=-" >&2
        exit 1
    fi
done

# Ensure temp/staging buckets exist (idempotent).
gsutil ls "gs://${PROJECT}-dataflow" >/dev/null 2>&1 \
    || gsutil mb -l "${GCP_REGION}" "gs://${PROJECT}-dataflow"

echo "==> Submitting probe pipeline"
uv run python - <<PY
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

options = PipelineOptions([
    "--runner=DataflowRunner",
    "--project=${PROJECT}",
    "--region=${GCP_REGION}",
    "--worker_zone=${GCP_ZONE}",
    "--worker_machine_type=g2-standard-8",
    "--sdk_container_image=${IMAGE}",
    "--experiments=use_runner_v2",
    "--dataflow_service_options=worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver",
    "--worker_disk_type=compute.googleapis.com/projects/${PROJECT}/regions/${GCP_REGION}/diskTypes/pd-ssd",
    "--worker_disk_size_gb=200",
    "--temp_location=gs://${PROJECT}-dataflow/temp",
    "--staging_location=gs://${PROJECT}-dataflow/staging",
    "--job_name=${JOB_NAME}",
    "--num_workers=1",
    "--max_num_workers=1",
    "--image-repository-username-secret-id=projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_USER",
    "--image-repository-password-secret-id=projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_PS",
])

with beam.Pipeline(options=options) as p:
    (
        p
        | "OneRow" >> beam.Create(["hello-gpu"])
        | "LogIt" >> beam.Map(lambda x: print(f"probe got: {x}"))
    )
PY

echo
echo "==> Probe submitted. Follow at:"
echo "    https://console.cloud.google.com/dataflow/jobs?project=${PROJECT}&region=${GCP_REGION}"
echo "    Job name: ${JOB_NAME}"
