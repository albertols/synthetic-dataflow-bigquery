"""Airflow DAG — submit the synthetic-dataflow-bigquery Flex Template.

This file is the *template*. Workflow `3_db_import_dag.yaml` runs `sed` over
it at deploy time to substitute build-time values. Runtime values
(table_fqn, num_rows, run_id) come from Airflow DAG params and Composer
Variables — operators don't need to re-import the DAG to change them.

Substitution markers:
  {{PROJECT_VERSION}}     project version of the sdfb-beam package being deployed
  {{DAG_VERSION}}         {{PROJECT_VERSION}}_<ISO timestamp> (unique DAG id)
  {{ENV}}                 dev | uat | prd
  {{EXECUTION_DATES}}     (reserved, unused for batch synthetic)

Network tag literals (devnetproxy / netsegcloudegress / artifactory / gke /
dataflow) are copied verbatim from the pe-btr-producer-scala-beam mediation
DAG — bank standard.
"""

from __future__ import annotations

from airflow import models
from airflow.models import Variable
from airflow.models.param import Param
from airflow.providers.google.cloud.operators.dataflow import (
    DataflowStartFlexTemplateOperator,
)
from airflow.utils.dates import days_ago

# -----------------------------------------------------------------------------
# Environment-resolved values — populated from Composer Variables.
# Set these once per Composer env via `gcloud composer environments update
# … --update-airflow-configs` or the Composer UI.
# -----------------------------------------------------------------------------
bucket_path = Variable.get("GCS_SYNTHETIC_DATAFLOW_STAGING")     # …-dataflow-staging
templates_path = Variable.get("GCS_SYNTHETIC_DATAFLOW_TEMPLATES")  # …-dataflow-templates
project_id = Variable.get("PROJECT_ID")
region = Variable.get("REGION")
subnetwork = Variable.get("DATAFLOW_SUBNET")
service_account = Variable.get("SA_DATAFLOW")
model_uri = Variable.get("SDFB_MODEL_URI")  # e.g. gs://<bucket>/synthetic/models/gemma4/e4b-it/v1/ (set per-env)
default_table_fqn = Variable.get("SDFB_DEFAULT_TABLE_FQN")

# -----------------------------------------------------------------------------
# Build-time substituted constants (replaced by sed in workflow 3).
# -----------------------------------------------------------------------------
app_domain = "synthetic"
app_name = "sdfb"
project_version = "{{PROJECT_VERSION}}"
dag_version = "{{DAG_VERSION}}"
env_name = "{{ENV}}"

job_name = f"{app_domain}-{app_name}-v{project_version.replace('.', '-').lower()}"
flex_template = f"sdfb-{project_version}-template.json"
dag_id = f"{app_domain}_{app_name}_{dag_version}"

# Network tags — five-tag chain, bank standard (see ebm_mediation.py).
devnetproxy_tag = "baseline-int-{{ENV}}-152100-1-devnetproxy"
netseg_network_tag = "baseline-int-{{ENV}}-152100-1-netsegcloudegress"
artifactory_network_tag = "baseline-int-{{ENV}}-97434-1-artifactory"
gke_network_tag = "int-{{ENV}}-125479-2-pwcc-es-gke"
dataflow_network_tag = "int-{{ENV}}-125479-2-pwcc-es-dataflow"
network_tags_chain = (
    f"{dataflow_network_tag};{netseg_network_tag};{artifactory_network_tag};"
    f"{gke_network_tag};{devnetproxy_tag}"
)
# -----------------------------------------------------------------------------
# DAG params — runtime-overridable on every trigger.
# Using Param objects so {{ params.X }} resolves to the VALUE, not the definition.
# All values are strings because Dataflow Flex Template parameters are strings.
# -----------------------------------------------------------------------------
default_dag_params = {
    "table_fqn": Param(
        default=default_table_fqn,
        type="string",
        description="FQN of the source BigQuery table to clone.",
    ),
    "num_rows": Param(
        default="1000",
        type="string",
        description="Number of synthetic rows to generate.",
    ),
    "engine": Param(
        default="b1_rag",
        type="string",
        enum=["b1_rag", "b2_library"],
        description="Generation engine.",
    ),
    "batch_size": Param(
        default="16",
        type="string",
        description="Records per LLM batch.",
    ),
    "similarity": Param(
        default="0.5",
        type="string",
        description="Similarity to reference (0.0 random → 1.0 mimic).",
    ),
}

with models.DAG(
        dag_id=dag_id,
        start_date=days_ago(1),
        schedule_interval="@once",
        catchup=False,
        max_active_runs=1,
        tags=["SYNTHETIC", "Dataflow", env_name.upper()],
        params=default_dag_params,
) as dag:
    DataflowStartFlexTemplateOperator(
        task_id=f"start_{app_name}",
        project_id=project_id,
        location=region,
        body={
            "launchParameter": {
                "containerSpecGcsPath": f"gs://{templates_path}/synthetic/{flex_template}",
                "jobName": job_name,
                "environment": {
                    "tempLocation": f"gs://{bucket_path}/temp/",
                    "stagingLocation": f"gs://{bucket_path}/staging",
                    "subnetwork": subnetwork,
                    "ipConfiguration": "WORKER_IP_PRIVATE",
                    "serviceAccountEmail": service_account,
                    "additionalExperiments": [
                        "use_runner_v2",
                        "upload_graph",
                        "enable_secure_boot",
                        # L4 GPU worker — see docs/GPU_CONTAINER.md.
                        "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver",
                        f"use_network_tags={network_tags_chain}",
                        f"use_network_tags_for_flex_templates={network_tags_chain}",
                    ],
                    "additionalUserLabels": {
                        "app": app_name,
                        "env": env_name,
                        "dag": dag_id,
                    },
                    "machineType": "g2-standard-8",
                    "maxWorkers": 4,
                    "diskSizeGb": 200,
                    "workerRegion": region,
                },
                "parameters": {
                    "ddl_uri": "{{ var.value.SDFB_DDL_URI }}",
                    "reference_table": "{{ params.table_fqn }}",
                    "reference_rows_limit": "10000",
                    "landing_table": "{{ var.value.SDFB_LANDING_TABLE }}",
                    "dlq_table": "{{ var.value.SDFB_DLQ_TABLE }}",
                    "num_rows": "{{ params.num_rows }}",
                    "batch_size": "{{ params.batch_size }}",
                    "similarity": "{{ params.similarity }}",
                    "run_id": "{{ dag_run.run_id }}",
                    "engine": "{{ params.engine }}",
                    "model_uri": model_uri,
                    "client_type": "vllm",
                },
            }
        },
        do_xcom_push=True,
        wait_until_finished=False,
    )