"""Airflow DAG — submit the synthetic-dataflow-bigquery Flex Template.

This file is the *template*. Workflow `3_import_dag.yaml` runs `sed` over it
at deploy time to substitute build-time values (env, project version, DAG
version, network tags). Runtime values (table_fqn, num_rows, run_id) come
from Airflow DAG params — operators don't need to re-import the DAG to
change them.

Substitution markers:
  {{PROJECT_VERSION}}     project version of the sdfb-beam package being deployed
  {{DAG_VERSION}}         {{PROJECT_VERSION}}_<ISO timestamp> (unique DAG id)
  {{ENV}}                 dev | uat | prd
"""

from __future__ import annotations

from airflow import models
from airflow.models import Variable
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
model_uri = Variable.get("SDFB_MODEL_URI")  # gs://<project>-models/gemma4/26b-a4b-awq/v1/

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

# Network tags — env-specific, set via sed substitution. Bank pattern.
dataflow_tag = f"int-{env_name}-<TEAM_NETWORK_TAG>-dataflow"

# -----------------------------------------------------------------------------
# DAG params — runtime-overridable on every trigger.
# -----------------------------------------------------------------------------
default_dag_params = {
    "table_fqn": {
        "type": "string",
        "default": "<DEFAULT_TABLE_FQN>",
        "description": "FQN of the source BigQuery table to clone.",
    },
    "num_rows": {
        "type": "integer",
        "default": 1000,
        "description": "Number of synthetic rows to generate.",
    },
    "engine": {
        "type": "string",
        "default": "b1_rag",
        "enum": ["b1_rag", "b2_library"],
        "description": "Generation engine.",
    },
    "batch_size": {
        "type": "integer",
        "default": 16,
        "description": "Records per LLM batch.",
    },
    "similarity": {
        "type": "number",
        "default": 0.5,
        "description": "Similarity to reference (0.0 random → 1.0 mimic).",
    },
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
                        f"use_network_tags={dataflow_tag}",
                        f"use_network_tags_for_flex_templates={dataflow_tag}",
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
