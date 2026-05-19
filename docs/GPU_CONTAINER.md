# GPU container build & deploy — M1 §10 runbook

End-to-end recipe for building the L4 GPU worker image, pushing to corporate JFrog, and verifying with a 1-row Dataflow probe. M4-only — the image is `linux/amd64` and ships CUDA bits; building on a Mac M4 requires Docker Desktop with Rosetta-emulated amd64 (slow but works) or buildx with a remote builder.

Companion to `MODEL_LAYOUT.md` (weights) and `M4_SETUP.md` (env).

## TL;DR

```bash
# one-time per machine
brew install --cask docker
# JFrog creds + region into .envrc (see "Step 0" below)
direnv allow

# every build
./scripts/build_gpu_image.sh
./scripts/push_gpu_image.sh
# (optional probe — see "Step 4")
```

## Locked decisions

| Variable | Value | Source |
|---|---|---|
| **Image registry** | Corporate JFrog | user decision 2026-05-19 |
| **Image path** | `${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu:<tag>` | mirrors prior SCIO build |
| **Dataflow region** | `europe-west3` (Frankfurt) | user decision 2026-05-19 |
| **GPU type** | `nvidia-l4`, count=1 | `MODEL_LAYOUT.md` |
| **Machine type** | `g2-standard-8` (1×L4, 8 vCPU, 32 GB) | `.claude/skills/gpu-dockerfile.md` |
| **Beam SDK** | pinned to `2.73.0` in Dockerfile | matches laptop venv |
| **CUDA** | `12.4.1-cudnn-runtime-ubuntu22.04` | vLLM 0.6+ requirement |
| **Python** | 3.11 (matches `apache/beam_python3.11_sdk`) | workspace `requires-python` |

If any locked value needs to change, edit it once in `docker/Dockerfile.gpu` and the launch flags below — don't sprinkle.

## Step 0 — env vars in `.envrc`

Add these to your real `.envrc` (gitignored) on the M4. Get the values from your platform team or from the prior SCIO repo's `build.yaml`:

```bash
# Corporate JFrog (matches the SCIO pipeline pattern)
export ARTIFACTORY_HOSTNAME="artifactory.intranet.db.com"   # <-- check exact host
export ARTIFACTORY_NAMESPACE="pwcclakees"                    # <-- check exact namespace

# Docker push creds (when you're actually pushing, otherwise leave unset)
export ARTIFACTORY_RELEASER_USER="<service-account-name>"
export ARTIFACTORY_RELEASER_PS="<token>"

# Convenience for Dataflow launch flags
export GCP_REGION="europe-west3"
export GCP_ZONE="europe-west3-b"
```

Then: `direnv allow`.

## Step 1 — preflight checks

Run these on the M4 first; each takes <30s.

### 1a. uv.lock is committed

```bash
test -f uv.lock && echo OK || echo "MISSING — run: uv lock && git add uv.lock && git commit -m 'add uv.lock'"
```

The Dockerfile uses `uv sync --frozen` which fails fast if `uv.lock` is absent. This avoids a build-time resolution that would silently produce a different deps tree than the laptop venv.

### 1b. Docker daemon is reachable

```bash
docker version | head -8
docker buildx ls   # confirm amd64 / linux platform support
```

If `docker version` errors, start Docker Desktop. If `linux/amd64` isn't in `buildx ls`, enable Rosetta in Docker Desktop → Settings → General → "Use Rosetta for x86_64/amd64 emulation".

### 1c. L4 GPU quota in `europe-west3`

```bash
gcloud compute project-info describe \
    --project="$(gcloud config get-value project)" \
    --flatten='quotas[]' \
    --format='table(quotas.metric,quotas.limit,quotas.usage)' \
  | grep -iE 'NVIDIA_L4_GPUS|GPUS_ALL_REGIONS' \
  | head -5
```

You need at least `NVIDIA_L4_GPUS = 1` in `europe-west3`. If 0, file a quota request via Cloud Console → IAM & Admin → Quotas — typically approved within a business day.

### 1d. Dataflow API + Compute API enabled

```bash
gcloud services list --enabled --project="$(gcloud config get-value project)" \
    --filter='config.name~dataflow OR config.name~compute' \
    --format='value(config.name)'
```

Should list both `dataflow.googleapis.com` and `compute.googleapis.com`.

## Step 2 — build the image

```bash
./scripts/build_gpu_image.sh
```

What happens:
1. Tags the image `${IMAGE_NAME}:<short-git-sha>` and `${IMAGE_NAME}:latest`.
2. Builds linux/amd64 (slow on M4 — first build ~15–25 min as it pulls CUDA, Beam SDK, vLLM wheels).
3. Runs an in-image smoke check: `beam/boot` exists, `python` works, `uv` works, `sdfb_beam` imports.

**Expected output tail**:
```
==> In-image smoke check
  beam boot: OK
Python 3.11.x
uv 0.4.x
  sdfb_beam: OK
==> Next: ./scripts/push_gpu_image.sh <sha>
```

If the build fails:
- **`vllm` wheel resolution error** — vLLM doesn't publish amd64 wheels for every version; if your `uv.lock` pins a missing version, run `uv lock --upgrade-package vllm` on the M4 and re-commit.
- **`apache/beam_python3.11_sdk:2.73.0 not found`** — Beam may have shipped a newer patch; bump the version in `Dockerfile.gpu` line 14 to whatever the laptop venv reports (`uv run python -c 'import apache_beam; print(apache_beam.__version__)'`).
- **OOM during `uv sync`** — Docker Desktop default RAM is 4 GB. Raise to 8 GB in Settings → Resources.

## Step 3 — push to JFrog

```bash
./scripts/push_gpu_image.sh
```

Pushes both the SHA tag and `latest`. Prints the image URI you'll pass to Dataflow.

**Sanity check the push worked** — pull from a fresh-ish view:

```bash
docker rmi "${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu:<sha>"
docker pull "${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu:<sha>"
```

## Step 4 — 1-row Dataflow probe

The probe is the acceptance gate for #10 — it confirms a GPU worker can pull the image, boot Beam, claim an L4, and run an empty bundle.

A real probe script lives in #11; for now you can submit a minimal pipeline ad-hoc. **Suggested probe** (paste into a `scripts/probe_gpu_dataflow.sh` if you want to keep it — I'll formalize this in #11):

```bash
PROJECT="$(gcloud config get-value project)"
JOB_NAME="sdfb-gpu-probe-$(date +%s)"
IMAGE="${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/sdfb-gpu:$(git rev-parse --short HEAD)"

uv run python -c "
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

options = PipelineOptions([
    '--runner=DataflowRunner',
    '--project=${PROJECT}',
    '--region=${GCP_REGION}',
    '--worker_zone=${GCP_ZONE}',
    '--worker_machine_type=g2-standard-8',
    '--sdk_container_image=${IMAGE}',
    '--experiments=use_runner_v2',
    '--dataflow_service_options=worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver',
    '--worker_disk_type=compute.googleapis.com/projects/${PROJECT}/regions/${GCP_REGION}/diskTypes/pd-ssd',
    '--worker_disk_size_gb=200',
    '--temp_location=gs://${PROJECT}-dataflow/temp',
    '--staging_location=gs://${PROJECT}-dataflow/staging',
    '--job_name=${JOB_NAME}',
    '--num_workers=1',
    '--max_num_workers=1',
    '--image-repository-username-secret-id=projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_USER',
    '--image-repository-password-secret-id=projects/${PROJECT}/secrets/ARTIFACTORY_RELEASER_PS',
])

with beam.Pipeline(options=options) as p:
    (p | 'OneRow' >> beam.Create(['hello-gpu'])
       | 'LogIt' >> beam.Map(lambda x: print(f'probe got: {x}')))
"
```

**Prerequisites for the probe**:
- Two secrets exist in Secret Manager:
  ```bash
  echo -n "${ARTIFACTORY_RELEASER_USER}" | gcloud secrets create ARTIFACTORY_RELEASER_USER --data-file=-
  echo -n "${ARTIFACTORY_RELEASER_PS}"   | gcloud secrets create ARTIFACTORY_RELEASER_PS   --data-file=-
  ```
  (Run once. If they already exist, use `versions add` instead of `create`.)
- The Dataflow service account has `roles/secretmanager.secretAccessor` on both secrets.
- The temp/staging buckets exist: `gsutil mb -l ${GCP_REGION} gs://${PROJECT}-dataflow` (run once).

**Watch the job**:
```bash
gcloud dataflow jobs list --region="${GCP_REGION}" --limit=1
# follow the Cloud Console URL it prints; expect status FAILED at the
# Beam shutdown step (no sink) but with all worker startup phases green
```

**Acceptance** (matches `.claude/agents/gpu-image-builder.md`):
1. Image pulls without auth errors (no `unauthorized: BAD_CREDENTIAL` in worker logs).
2. NVIDIA driver installs (look for `nvidia-smi` succeeded in worker startup logs).
3. GPU utilization metric is non-zero in Cloud Monitoring during the run (even idle vLLM warmup shows ≥10%).
4. Cold-start budget: worker ready < 3 min from job submit.

## What to send back

Paste these once you've worked through the steps:

1. **After Step 1c**: the `NVIDIA_L4_GPUS` quota row.
2. **After Step 2**: tail of `build_gpu_image.sh` output (the 5-line smoke check).
3. **After Step 3**: the `==> Pushed.` line with the image URI.
4. **After Step 4**: Dataflow job ID + the worker-startup tail (last ~50 lines from the Cloud Console).

Once #10 is green, #9 (vLLM `ModelHandler` + `ModelClient`) becomes the immediate next task — that's where the real LLM calls finally happen.

## Gotchas worth knowing now

- **M4 amd64 build is slow.** First clean build can take 25 min because of CUDA/cuDNN layer downloads (~3 GB). Rebuilds after touching only `packages/*/src` are ~30s thanks to layer caching.
- **`uv.lock` drift between laptop and M4.** If the M4 generates `uv.lock` with different wheel hashes (different arch/Python patch), `uv sync --frozen` in the container can fail. Resolution: always run `uv lock` on the M4, commit, and let the laptop pull.
- **JFrog pull on workers needs both Secret Manager secrets** — if you push but the probe fails with `unauthorized`, you forgot to wire the secrets or grant `secretAccessor`.
- **`--num_workers=1` for the probe.** Don't let Dataflow autoscale to multiple GPU workers for a no-op probe — that wastes quota and the L4 cost is non-trivial.
- **Region drift.** If your BigQuery dataset is in a different region than `europe-west3`, the Dataflow job will fail at the BQ read step (M1 §11). For the probe (no BQ I/O), region only needs to match where you have L4 quota.

## File map

```
docker/
├── Dockerfile.gpu            # the image definition (this PR)
└── .dockerignore             # build-context exclusions
scripts/
├── build_gpu_image.sh        # tag + build + in-image smoke check
└── push_gpu_image.sh         # docker login + push
docs/
└── GPU_CONTAINER.md          # THIS FILE
```
