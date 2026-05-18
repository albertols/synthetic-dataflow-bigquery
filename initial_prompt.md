# ROLE

ACT as Senior AI GCP Technical Lead + Senior GCP Cloud Engineer + most experience Apache Beam committer. We are gonna
make something really extreme top tier quality!

# STATEMENT

I need to create a Synthetic Apache Beam on Dataflow, synthetic-dataflow-bigquery. This is not generating just random
mock data, but generating synthetic data that is realistic and follows the schema and constraints of the original
BigQuery table.

The goal is to have a scalable solution that can generate large volumes of synthetic data based on the DDL metadata of a
BigQuery table (from gcs:// _ddl.json or directly with parts of bigquery_ddl_metadata.py ), using a LLM for the
generation process.

A plain prompt statement and LLM is easier, but structured/typed data consumable in a DWH is a different story, and we
need to design a robust system to achieve this goal. We also need to consider the cost implications of using LLMs for
data generation, and optimize the pipeline for performance and reliability."

This will be used for every single company in the world that uses DataFlow and BigQuery (so it must fulfill OSS
world-class quality deliveries), and needs to generate synthetic data for testing, development, or any other purpose; as
well as validating and setting up a full foundational RAG.

* IMPORTANT: at the beginning we would use the DEV environment (when validated we will create a PRD parallel universe),
data MUST be fictitious, but at the same time FAKE, and sometimes within ranges following similarity but not identical
to the original data, so it can be used for testing and development purposes. We should control with input args the
degree of similarity (or randomness) of the generated data, and also the volume (number of rows) to be generated. We can
start with a small volume (like 1000 rows) for testing, and then scale up to larger volumes (like 1 million rows) once
we have a working solution.

* HINT: please deep dive them and explore them EVERY SINGLE attached resource (and find other related ones, deep search
with the related literature and workable solutions), assessing pros and cons and potential challenges but making sure we
MUST END UP with fully functional end-to-end solutions.

* MODELS: I want to try different OSS models, and also compare with Gemma3, 2.5, GPT5 (if available) or any other well
known for this purpose (Llama, Claude, Deepseek, Grok, etc), to see which one is more effective for the synthetic data
generation use case. (avoid hugging face for the time being), they can be loaded from GCS (pre downloaded here in the
format we need) or any GCP OSS model registry

* MODES: single-table (considering P.K, if exists), multi-table (considering P.K if exists in other tables). The priority
is single-table from the time being, relevant info: https://docs.cloud.google.com/bigquery/docs/primary-foreign-keys

# 1. HLA

At very high level, the approach is to create synthetic data in BigQuery (input the DDL.json, considering sample data or
RAG approach), then with Apache Beam on DataFlow (with a LLM), write in BigQuery (output). Hereby, some components (or
context) of the system are (can be):

A. bigquery_ddl_metadata.py DDL.json (potential input), please
check: https://docs.cloud.google.com/bigquery/docs/schemas#creating_a_JSON_schema_file, we can infer the datatypes and
bigquery metadata with a big query client, I have this local script already (it can be useful at
pre_processing/data_preparation stage):

```bigquery_ddl_metadata.py
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from google.cloud import bigquery
from google.api_core.retry import Retry
import logging
import json
import os
import sys
import time
import certifi
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ddl_metadata.log')
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from third-party loggers but keep warnings
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('apache_beam').setLevel(logging.WARNING)

# Default timeout in seconds (override via --timeout or env BIGQUERY_TIMEOUT)
DEFAULT_TIMEOUT = float(os.environ.get('BIGQUERY_TIMEOUT', '50'))


def configure_ssl():
    """Configure SSL certificates for environments with custom/corporate CAs.

    Reads from environment variables:
        - REQUESTS_CA_BUNDLE: path to custom CA bundle
        - SSL_CERT_FILE: alternative path to CA bundle

    These should be set in the .run.xml or shell environment, not hardcoded here.
    """
    ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
    if ca_bundle:
        logger.info(f"Using custom CA bundle: {ca_bundle}")
        return

    certifi_path = certifi.where()
    logger.info(f"No custom CA bundle set. Using certifi default: {certifi_path}")


def log_proxy_config():
    """Log current proxy configuration (set via environment variables in .run.xml)."""
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    no_proxy = os.environ.get('NO_PROXY') or os.environ.get('no_proxy')

    if http_proxy or https_proxy:
        logger.info(f"Proxy config (from environment):")
        logger.info(f"  HTTP_PROXY:  {http_proxy or 'NOT SET'}")
        logger.info(f"  HTTPS_PROXY: {https_proxy or 'NOT SET'}")
        logger.info(f"  NO_PROXY:    {no_proxy or 'NOT SET'}")
    else:
        logger.info("No proxy configured (HTTP_PROXY/HTTPS_PROXY not set)")


def test_bigquery_connection(project: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Test BigQuery connectivity before running the pipeline."""
    logger.info(f"Testing BigQuery connection to project '{project}' (timeout={timeout}s)...")
    start = time.time()
    try:
        client = bigquery.Client(project=project)
        query = "SELECT 1"
        client.query(query, timeout=timeout).result(timeout=timeout)
        elapsed = time.time() - start
        logger.info(f"✓ BigQuery connection successful ({elapsed:.1f}s)")
        return True
    except KeyboardInterrupt:
        logger.error("Connection test interrupted by user.")
        return False
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ BigQuery connection FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        logger.error("Possible causes:")
        logger.error("  1. Proxy not configured or unreachable")
        logger.error(f"     → HTTP_PROXY:  {os.environ.get('HTTP_PROXY', 'NOT SET')}")
        logger.error(f"     → HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'NOT SET')}")
        logger.error("  2. SSL certificate issue (self-signed cert in proxy chain)")
        logger.error(f"     → REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}")
        logger.error("  3. No VPN/network connectivity to GCP")
        logger.error("  4. Invalid credentials (run: gcloud auth application-default login)")
        logger.error("  5. Project does not exist or insufficient permissions")
        return False


class ExtractDDLMetadata(beam.DoFn):
    """Extracts full DDL metadata from a BigQuery table."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def setup(self):
        """Called once per worker initialization."""
        logger.info("ExtractDDLMetadata DoFn initialized.")

    def process(self, table_ref: Dict[str, str]):
        project = table_ref['project']
        dataset = table_ref['dataset']
        table_name = table_ref['table']
        full_table_id = f"{project}.{dataset}.{table_name}"

        logger.info(f"[ExtractDDLMetadata] Starting metadata extraction for {full_table_id}")
        start_time = time.time()

        client = bigquery.Client(project=project)

        logger.info(f"[ExtractDDLMetadata] Fetching table metadata (timeout={self.timeout}s)...")
        retry_config = Retry(deadline=self.timeout, maximum=3)

        try:
            table = client.get_table(full_table_id, retry=retry_config, timeout=self.timeout)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[ExtractDDLMetadata] FAILED to get table '{full_table_id}' after {elapsed:.1f}s: "
                f"{type(e).__name__}: {e}"
            )
            raise

        elapsed = time.time() - start_time
        logger.info(
            f"[ExtractDDLMetadata] Got table metadata in {elapsed:.1f}s — "
            f"{len(table.schema)} fields found"
        )

        # Build schema as plain dicts (avoids serialization issues with dataclass + Beam)
        schema = []
        for field in table.schema:
            schema.append({
                "name": field.name,
                "field_type": field.field_type,
                "mode": field.mode,
                "description": field.description if field.description else ""
            })

        # Build comprehensive table info
        table_info = {
            "table_id": full_table_id,
            "created": table.created.isoformat() if table.created else None,
            "last_modified": table.modified.isoformat() if table.modified else None,
            "table_expiry": table.expires.isoformat() if table.expires else "NEVER",
            "data_location": table.location,
            "description": table.description or "",
            "labels": dict(table.labels) if table.labels else {},
            "table_type": table.table_type,
            "encryption_configuration": (
                table.encryption_configuration.kms_key_name
                if table.encryption_configuration else None
            ),
            "default_collation": getattr(table, 'default_collation_name', None),
            "case_insensitive": False,
        }

        # Partitioning info
        partitioning = None
        if table.time_partitioning:
            partitioning = {
                "type": table.time_partitioning.type_ or "DAY",
                "field": table.time_partitioning.field or None,
                "expiration_days": (
                    table.time_partitioning.expiration_ms / 86400000
                    if table.time_partitioning.expiration_ms else None
                ),
                "require_partition_filter": table.require_partition_filter or False,
            }
        elif table.range_partitioning:
            partitioning = {
                "type": "RANGE",
                "field": table.range_partitioning.field,
                "range": {
                    "start": table.range_partitioning.range_.start,
                    "end": table.range_partitioning.range_.end,
                    "interval": table.range_partitioning.range_.interval,
                }
            }

        # Clustering info
        clustering = None
        if table.clustering_fields:
            clustering = {"fields": list(table.clustering_fields)}

        # Storage info — fetch detailed bytes from INFORMATION_SCHEMA
        storage_info = self._get_storage_info(client, project, dataset, table_name, table)

        # Primary keys from table constraints
        primary_keys = self._get_primary_keys(table)

        logger.info("[ExtractDDLMetadata] Schema fields:")
        for field in schema[:10]:
            logger.info(f"  {field['name']}: {field['field_type']} ({field['mode']})")
        if len(schema) > 10:
            logger.info(f"  ... and {len(schema) - 10} more fields")

        logger.info(f"[ExtractDDLMetadata] Table info: location={table_info['data_location']}, "
                    f"type={table_info['table_type']}, rows={storage_info['num_rows']}")

        total_elapsed = time.time() - start_time
        logger.info(f"[ExtractDDLMetadata] Completed in {total_elapsed:.1f}s")

        yield {
            "table_ref": table_ref,
            "table_info": table_info,
            "schema": schema,
            "primary_keys": primary_keys,
            "partitioning": partitioning,
            "clustering": clustering,
            "storage_info": storage_info,
        }

    def _get_primary_keys(self, table) -> Optional[List[str]]:
        """Extract primary keys from table constraints or description."""
        # Try table_constraints (BigQuery API)
        constraints = getattr(table, 'table_constraints', None)
        if constraints and hasattr(constraints, 'primary_key'):
            pk = constraints.primary_key
            if pk and hasattr(pk, 'columns') and pk.columns:
                return list(pk.columns)

        # Fallback: parse from description
        if table.description and "PRIMARY KEY:" in table.description:
            return [pk.strip() for pk in
                    table.description.split("PRIMARY KEY:")[1].split("\n")[0].split(",")]
        return None

    @staticmethod
    def _human_bytes(num_bytes) -> Optional[str]:
        """Convert bytes to human-readable string."""
        if num_bytes is None:
            return None
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    def _get_storage_info(self, client, project: str, dataset: str, table_name: str, table) -> Dict:
        """Fetch detailed storage info from table properties and INFORMATION_SCHEMA."""
        # The BigQuery REST API returns storage stats in the table resource _properties
        props = table._properties

        def _int_prop(key):
            val = props.get(key)
            return int(val) if val is not None else None

        num_bytes = _int_prop('numBytes')
        num_long_term_bytes = _int_prop('numLongTermBytes')
        num_physical_bytes = _int_prop('numPhysicalBytes')
        num_active_logical_bytes = _int_prop('numActiveLogicalBytes')
        num_long_term_logical_bytes = _int_prop('numLongTermLogicalBytes')
        num_active_physical_bytes = _int_prop('numActivePhysicalBytes')
        num_long_term_physical_bytes = _int_prop('numLongTermPhysicalBytes')
        num_time_travel_physical_bytes = _int_prop('numTimeTravelPhysicalBytes')
        num_total_physical_bytes = _int_prop('numTotalPhysicalBytes')

        storage_info = {
            "num_rows": table.num_rows,
            "num_partitions": None,
            "total_logical_bytes": num_bytes,
            "total_logical_bytes_human": self._human_bytes(num_bytes),
            "active_logical_bytes": num_active_logical_bytes,
            "active_logical_bytes_human": self._human_bytes(num_active_logical_bytes),
            "long_term_logical_bytes": num_long_term_logical_bytes or num_long_term_bytes,
            "long_term_logical_bytes_human": self._human_bytes(num_long_term_logical_bytes or num_long_term_bytes),
            "total_physical_bytes": num_total_physical_bytes or num_physical_bytes,
            "total_physical_bytes_human": self._human_bytes(num_total_physical_bytes or num_physical_bytes),
            "active_physical_bytes": num_active_physical_bytes,
            "active_physical_bytes_human": self._human_bytes(num_active_physical_bytes),
            "long_term_physical_bytes": num_long_term_physical_bytes,
            "long_term_physical_bytes_human": self._human_bytes(num_long_term_physical_bytes),
            "time_travel_physical_bytes": num_time_travel_physical_bytes,
            "time_travel_physical_bytes_human": self._human_bytes(num_time_travel_physical_bytes),
        }

        # Log raw properties for debugging storage keys
        storage_keys = [k for k in props.keys() if 'byte' in k.lower() or 'physical' in k.lower() or 'logical' in k.lower()]
        logger.info(f"[ExtractDDLMetadata] Raw storage properties available: {storage_keys}")

        # Get partition count from INFORMATION_SCHEMA.PARTITIONS
        try:
            partition_query = f"""
                SELECT COUNT(*) as num_partitions
                FROM `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS`
                WHERE table_name = '{table_name}'
            """
            logger.info("[ExtractDDLMetadata] Querying INFORMATION_SCHEMA.PARTITIONS...")
            result = client.query(partition_query, timeout=self.timeout).result(timeout=self.timeout)
            row = next(result)
            storage_info["num_partitions"] = row.num_partitions
        except Exception as e:
            logger.warning(f"[ExtractDDLMetadata] Could not get partition count: {type(e).__name__}: {e}")

        # If raw properties didn't have detailed bytes, try INFORMATION_SCHEMA.TABLE_STORAGE as fallback
        if num_active_logical_bytes is None:
            try:
                storage_query = f"""
                    SELECT
                        total_rows,
                        total_logical_bytes,
                        active_logical_bytes,
                        long_term_logical_bytes,
                        total_physical_bytes,
                        active_physical_bytes,
                        long_term_physical_bytes,
                        time_travel_physical_bytes
                    FROM `{project}.region-{table.location}.INFORMATION_SCHEMA.TABLE_STORAGE`
                    WHERE table_schema = '{dataset}'
                      AND table_name = '{table_name}'
                """
                logger.info("[ExtractDDLMetadata] Querying region INFORMATION_SCHEMA.TABLE_STORAGE...")
                result = client.query(storage_query, timeout=self.timeout).result(timeout=self.timeout)
                row = next(result)

                storage_info["num_rows"] = row.total_rows or table.num_rows
                storage_info["total_logical_bytes"] = row.total_logical_bytes
                storage_info["total_logical_bytes_human"] = self._human_bytes(row.total_logical_bytes)
                storage_info["active_logical_bytes"] = row.active_logical_bytes
                storage_info["active_logical_bytes_human"] = self._human_bytes(row.active_logical_bytes)
                storage_info["long_term_logical_bytes"] = row.long_term_logical_bytes
                storage_info["long_term_logical_bytes_human"] = self._human_bytes(row.long_term_logical_bytes)
                storage_info["total_physical_bytes"] = row.total_physical_bytes
                storage_info["total_physical_bytes_human"] = self._human_bytes(row.total_physical_bytes)
                storage_info["active_physical_bytes"] = row.active_physical_bytes
                storage_info["active_physical_bytes_human"] = self._human_bytes(row.active_physical_bytes)
                storage_info["long_term_physical_bytes"] = row.long_term_physical_bytes
                storage_info["long_term_physical_bytes_human"] = self._human_bytes(row.long_term_physical_bytes)
                storage_info["time_travel_physical_bytes"] = row.time_travel_physical_bytes
                storage_info["time_travel_physical_bytes_human"] = self._human_bytes(row.time_travel_physical_bytes)

            except Exception as e:
                logger.warning(f"[ExtractDDLMetadata] Could not get TABLE_STORAGE info: {type(e).__name__}: {e}")

        return storage_info


def get_output_path(pipeline_options, base_path: str, dataset: str, table: str) -> str:
    """Determine output path based on runner type.

    Output structure: {base_path}/{dataset}/ddl_metadata_{dataset}_{table}.json

    - DirectRunner: local filesystem path
    - DataflowRunner: GCS bucket path (gs://)
    """
    runner = pipeline_options.view_as(StandardOptions).runner
    filename = f"ddl_metadata_{dataset}_{table}.json"

    if runner == 'DirectRunner':
        output_dir = os.path.join(base_path, dataset)
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    else:
        if not base_path.startswith('gs://'):
            logger.warning(
                f"Output base '{base_path}' does not start with 'gs://'. "
                f"For DataflowRunner, output should be a GCS path."
            )
        return f"{base_path}/{dataset}/{filename}"


class WriteMetadataToJSON(beam.DoFn):
    """Writes extracted DDL metadata to a JSON file."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def process(self, element: Dict):
        logger.info(f"[WriteMetadataToJSON] Writing metadata to: {self.output_path}")
        start = time.time()

        output = {
            "generated_at": datetime.now().isoformat(),
            "table_info": element["table_info"],
            "schema": element["schema"],
            "primary_keys": element["primary_keys"],
            "partitioning": element["partitioning"],
            "clustering": element["clustering"],
            "storage_info": element["storage_info"],
        }

        with beam.io.filesystems.FileSystems.create(self.output_path) as f:
            content = json.dumps(output, indent=2).encode('utf-8')
            f.write(content)

        elapsed = time.time() - start
        logger.info(
            f"[WriteMetadataToJSON] ✓ Written {len(content)} bytes to {self.output_path} ({elapsed:.1f}s)"
        )
        yield output


def run_pipeline(pipeline, project: str, dataset: str, table: str, output_path: str):
    """Run the DDL metadata extraction pipeline."""
    logger.info(f"Building pipeline graph for {project}.{dataset}.{table}")
    logger.info(f"Output will be written to: {output_path}")

    metadata = (
        pipeline
        | 'CreateTableRef' >> beam.Create([{
            'project': project,
            'dataset': dataset,
            'table': table
        }])
        | 'ExtractDDLMetadata' >> beam.ParDo(ExtractDDLMetadata(timeout=DEFAULT_TIMEOUT))
        | 'WriteMetadata' >> beam.ParDo(WriteMetadataToJSON(output_path))
    )

    return metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BigQuery DDL Metadata Extraction Pipeline')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--dataset', required=True, help='BigQuery dataset')
    parser.add_argument('--table', required=True, help='BigQuery table')
    parser.add_argument('--output_base', default=None,
                        help='Output base path. Local dir for DirectRunner, gs:// bucket for DataflowRunner')
    parser.add_argument('--runner', default='DirectRunner',
                        help='Pipeline runner (DirectRunner or DataflowRunner)')
    parser.add_argument('--skip_connection_test', action='store_true',
                        help='Skip the BigQuery connection pre-check')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Connection/read timeout in seconds (default: {DEFAULT_TIMEOUT})')

    args, pipeline_args = parser.parse_known_args()

    logger.info("=" * 60)
    logger.info("BigQuery DDL Metadata Extraction Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Project:  {args.project}")
    logger.info(f"  Dataset:  {args.dataset}")
    logger.info(f"  Table:    {args.table}")
    logger.info(f"  Runner:   {args.runner}")
    logger.info(f"  Timeout:  {args.timeout}s")
    logger.info("=" * 60)

    # Log proxy/SSL config (set externally via env vars in .run.xml or shell)
    log_proxy_config()
    configure_ssl()

    # Pre-check connectivity (fail fast instead of hanging)
    if not args.skip_connection_test:
        if not test_bigquery_connection(args.project, timeout=args.timeout):
            logger.critical("Aborting: Cannot connect to BigQuery. See errors above.")
            sys.exit(1)
    else:
        logger.info("Skipping connection pre-check (--skip_connection_test)")

    options = PipelineOptions(
        pipeline_args,
        runner=args.runner,
        project=args.project,
        temp_location='./temp' if args.runner == 'DirectRunner' else f'gs://{args.project}-temp/dataflow/temp',
        save_main_session=True,
    )

    # Determine output base path
    if args.output_base:
        output_base = args.output_base
    elif args.runner == 'DirectRunner':
        output_base = './output'
    else:
        output_base = f'gs://{args.project}-dataflow/ddl_metadata'

    output_path = get_output_path(options, output_base, args.dataset, args.table)
    logger.info(f"Resolved output path: {output_path}")

    try:
        logger.info("Launching pipeline...")
        with beam.Pipeline(options=options) as p:
            run_pipeline(p, project=args.project, dataset=args.dataset, table=args.table, output_path=output_path)
        logger.info("=" * 60)
        logger.info("✓ DDL metadata pipeline executed successfully")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Pipeline execution failed: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)

```

## FEATURE B. Apache Beam on DataFlow with a LLM (for synthetic data generation), the most important/relevant/ critical component! /ultrathink here, let's go!!!

I want to use a LLM (like Gemma3, 2.5, GPT5 or any open source model, well known for this purpose) to generate synthetic
data based on the DDL metadata extracted from BigQuery. The idea is to create a Beam pipeline that takes the DDL
metadata as input, prompts the LLM to generate synthetic data, and then writes the generated data back to BigQuery.

Experiments show me that direct prompting in local with models like models/gemma-2b-it or models/gemma-4b-it with sample
of 1000 data entries actually cannot be reliable enough (incomplete text, timestamps, etc, repetitive data, incomplete
fields, etc), and fine-tuning is expensive and an overkill for this. Therefore, we need a world-class GCP solution,
scalable, and with a good architecture design, to achieve this goal. I would love to explore and implement two (likely)
approaches, to compare and see which one is more effective/scalable/economical:

### ENGINE B.1. Leverage the solution with a RAG approach.

There are very interesting documentation, notebooks and examples following this approach, we can maybe adapt it for the
synthetic data generation use case. Hereby some resources for RAG, Embeddings, Beam-ML and LLMs on DataFlow:

- https://cloud.google.com/dataflow/docs/notebooks/bigquery_vector_ingestion_and_search -> https://github.com/apache/beam/blob/master/examples/notebooks/beam-ml/bigquery_vector_ingestion_and_search.ipynb
- https://docs.cloud.google.com/dataflow/docs/notebooks/bigquery_vector_ingestion_and_search
- https://beam.apache.org/releases/pydoc/current/apache_beam.ml.rag.html
- https://beam.apache.org/documentation/ml/large-language-modeling/
- https://github.com/apache/beam/tree/master/examples/notebooks/beam-ml (https://github.com/apache/beam/tree/master/examples/notebooks/beam-ml/rag_usecase)
- https://beam.apache.org/releases/pydoc/current/apache_beam.ml.inference.base.html#apache_beam.ml.inference.base.RunInference (
  such as https://docs.cloud.google.com/dataflow/docs/notebooks/run_inference_generative_ai or equivalent with OSS
  models)
- https://beam.apache.org/documentation/ml/about-ml/

I am aware of the Embedding Phase here. In any case, it is going to be useful (and worth the effort) to set up the
foundation of a RAG layer anyway (somehow) to leverage future RAG capabilities (for technical user, business users,
bots, agentic use cases, etc)

### ENGINE B.2. Leverage the solution without a RAG, but "attaching" reputational, well-tested, Python libraries into Beam DataFlow.

No need of RAG, but we can use the LLMs but making the most of the generation capabilities but adding the parallelism of
Beam and DataFlow into BigQuery. Hereby some projects or ideas:

- https://github.com/hitsz-ids/synthetic-data-generator
- https://github.com/meta-llama/synthetic-data-kit , although I see it is more focused on documents, not structured
  data.
- https://github.com/datadreamer-dev/DataDreamer/tree/main
- https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms (https://deepeval.com/guides/guides-rag-evaluation & https://github.com/confident-ai/deepeval/tree/main)
- https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data#21-techniques

The key here is to adapt a Python library into a efficient and scalable Beam pipeline, that can be run on DataFlow, and
that can generate synthetic data based on the DDL metadata extracted from BigQuery, and then write the generated data
back to BigQuery. We need to make sure that the generation process is robust, and that it can handle the constraints and
schema of the original BigQuery table, while also being cost-effective in terms of LLM usage.

### ENGINE B.2. OTHER RESEARCH IDEAS coming from pappers, articles, etc about synthetic data generation with LLMs, that we can maybe adapt or leverage for our use case if you haven found anything suitable maybe:

- https://github.com/pengr/LLM-Synthetic-Data - loads of papers and resources about synthetic data generation with LLMs.

### ENGINE B.3. OTHERS?

Feel free to suggest other approaches or variations of the above ones, as long as they are scalable, robust, and
cost-effective for generating synthetic data in BigQuery using LLMs on DataFlow.

# FEATURE C: THE VALIDATION MODULE/PROJECT

We would need a way to validate the generated synthetic data, to ensure that it follows the schema and constraints of
the original BigQuery table, and that it is realistic enough for testing and development purposes. We can use a
combination of automated tests (like schema validation, data quality checks, etc) or even use well known data validation
libraries (like Great Expectations, Deequ, etc) to validate the generated synthetic data before (or after, depending on
the config) writing it back to BigQuery.

Results should be sunk in a BigQuery table with the results of the validation, so we can track the quality of the
generated data over time and make improvements to the generation process as needed (include the engine type, LLM,
metrics by each column in each table, so we might a struct per validated_execution entry).

## FEATURE C: a potential VALIDATION SPEC DRIVEN DEVELOPMENT PROMPT (to be reviewed and improved if needed, but I think it is a good starting point for the validation module):

```
# SPEC: Production-Grade Synthetic Data Validation Framework
# Runtime: Apache Beam (Python SDK) on Google Cloud Dataflow → BigQuery
# Role: Senior Data Validation & ML Quality Engineer
# Modes: (A) Pre-write in-pipeline validation  |  (B) Post-write warehouse validation
# Both modes are MANDATORY and share contracts, checks, thresholds, and report schema.

## 1. CONTEXT & OBJECTIVE

Design and implement a production-grade, continuously evaluated data validation
framework for typed, structured synthetic tabular data generated by Apache Beam
pipelines running on Google Cloud Dataflow and persisted to BigQuery.

The framework MUST operate in two complementary modes, both backed by the same
contract and check catalog:

- MODE A — PRE-WRITE (in-pipeline from the previous synthetic data generation step):
  Validation executes INSIDE the Beam DAG, on each PCollection of generated
  synthetic records, BEFORE `WriteToBigQuery`. Invalid records are routed to
  a dead-letter sink via Beam tagged outputs; quality metrics are computed
  per-bundle and merged at the end of the job.

- MODE B — POST-WRITE (warehouse after data has landed in BigQuery):
  Validation executes AFTER the data has landed in BigQuery, against the
  physical table. Runs full-table assertions, fidelity vs. baseline, drift,
  and produces auditable HTML/JSON reports.

Both modes MUST be runnable independently or chained. CI/CD MUST gate
promotion on Mode B results; Dataflow job success MUST gate on Mode A.

## 2. GUIDING STANDARDS

- ISO/IEC 25012 — Data Quality Model (dimensional reference)
- ISO 8000 — Data quality and master data
- DAMA-DMBOK2 — Data management body of knowledge
- Open Data Contract Standard (ODCS) — Bitol / LF AI & Data
- OpenLineage spec — lineage event emission from Beam
- DCAM (EDM Council) — capability alignment

## 3. SIX DATA QUALITY DIMENSIONS

Every check MUST declare exactly one of:
1. Accuracy       — values reflect reference distribution
2. Completeness   — null rates / required-field presence
3. Uniqueness     — duplicate rates on declared keys
4. Consistency    — cross-table referential & cross-column logic
5. Timeliness     — generation/ingestion freshness
6. Validity       — type, enum, range, regex, schema conformance

## 4. CANONICAL TECHNOLOGY STACK (pinned for 2026 APIs)

### 4.1 Contracts & Typing (shared by both modes)
- pydantic >= 2.x                 — canonical record models
- apache-beam[gcp] >= 2.60         — typed PCollections, BigQueryIO
- google-cloud-bigquery >= 3.x     — INFORMATION_SCHEMA, DDL gen
- protobuf / avro (optional)       — cross-language contracts

### 4.2 Mode A — In-Pipeline (pre-write) validation
- pandera >= 0.20                  — DataFrame schema in DoFn micro-batches
- pydantic (per-record)            — at generator boundary
- whylogs >= 1.5                   — MERGEABLE column profiles across workers
- apache_beam.metrics.Metrics      — counters, distributions to Cloud Monitoring
- hypothesis                       — property-based tests of the generator

### 4.3 Mode B — Post-Write (warehouse) validation
- great_expectations >= 1.0 (GX Core)  — Suites, Checkpoints, Data Docs
  Uses: context.data_sources.add_or_update_sql(...) with
  connection_string="bigquery://<project>/<dataset>"
- soda-core-bigquery >= 3.x        — SodaCL declarative checks, CI gating
- dbt-expectations (optional)      — in-warehouse assertions if dbt is used
- Dataplex Data Quality (managed)  — org-wide rule layer (optional)

### 4.4 Synthetic-Data Fidelity (Mode B)
- sdmetrics >= 0.17                — QualityReport, DiagnosticReport
- synthcity (optional)             — research-grade fidelity/utility/privacy
- table-evaluator (optional)       — quick KS / Chi-squared overlap

### 4.5 Drift vs. Reference Baseline (Mode B, optionally Mode A on samples)
- evidently >= 0.4 (current stable API; do not use legacy <0.3 namespace)
  Reports: DataDriftPreset, DataSummaryPreset
- nannyml >= 0.10                  — drift with confidence intervals
- scipy.stats, statsmodels         — KS, Chi-squared, Wasserstein primitives

### 4.6 Lineage, Catalog, Orchestration
- openlineage-python + openlineage-airflow / dagster-openlineage
- marquez (reference backend) or DataHub / OpenMetadata
- Dagster (preferred) or Apache Airflow with GreatExpectationsOperator /
  SodaScanOperator for gating
- Cloud Composer 2 if Airflow is mandated by org

## 5. POTENTIAL VALIDATION ARCHITECTURE (to be validated)

### 5.1 Pipeline-wide flow

   Pydantic Contract (single source of truth)
        │
        ├──> codegen: Pandera schema (Mode A) + BigQuery DDL + GX Suite (Mode B)
        │
   ┌────┴─────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  MODE A — INSIDE THE BEAM PIPELINE (Dataflow)                    │
   │                                                                  │
   │  Generator DoFn ──> ValidateRecordDoFn (Pydantic, per-record)    │
   │       │                                                          │
   │       ├─main──> BatchToPandas ──> PanderaValidateDoFn ──┐        │
   │       │                                                 │        │
   │       │                                  WhylogsProfile │        │
   │       │                                  DoFn (merge)   │        │
   │       │                                                 │        │
   │       │                                                 ▼        │
   │       │                                       Combine.globally   │
   │       │                                       (merge profiles)   │
   │       │                                                 │        │
   │       │                                                 ▼        │
   │       │                                       Write profile JSON │
   │       │                                       to GCS + BQ row    │
   │       │                                                          │
   │       └─tagged(invalid)──> DeadLetter BigQuery Table              │
   │                                                                  │
   │  Main valid PCollection ──> WriteToBigQuery(landing_table)        │
   └──────────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  MODE B — POST-WRITE (triggered on Dataflow job SUCCESS)         │
   │                                                                  │
   │   GX Checkpoint ──> Expectation Suite over BigQuery landing tbl  │
   │   Soda Scan     ──> SodaCL checks, threshold gating              │
   │   INFORMATION_SCHEMA queries ──> row count, schema drift          │
   │   SDMetrics QualityReport ──> synthetic vs. reference            │
   │   Evidently DataDriftPreset ──> drift vs. baseline run           │
   │   Dataplex DQ scan (optional)                                    │
   │                                                                  │
   │   All results ──> synthetic_data_quality.* BigQuery dataset                │
   │   Reports HTML/JSON ──> GCS, signed URLs ──> Looker Studio       │
   └──────────────────────────────────────────────────────────────────┘

### 5.2 Mode A — Pre-Write Validation (inside Apache Beam)

REQUIRED Beam transforms (Python SDK):

a) `ValidateRecordDoFn` — applies the Pydantic model to each record;
   on `ValidationError` yields `beam.pvalue.TaggedOutput("invalid", ...)`
   with the raw record + structured error context (error_type, error_detail,
   pipeline_step, stage="pre_write", timestamp).

b) `BatchElements` (built-in) — groups validated records into micro-batches
   (default 1_000–5_000 elements) suitable for vectorized DataFrame checks.

c) `PanderaValidateBatchDoFn` — converts the batch to a pandas DataFrame,
   runs the Pandera schema with `lazy=True` to collect all failures, and
   emits any failing rows to the `"invalid"` tag with the failure cases
   attached. Valid rows are re-emitted to the main output.

d) `WhylogsProfileDoFn` + `CombineGlobally(MergeProfiles())` — each worker
   produces a mergeable whylogs profile; the combiner merges them into a
   single job-level profile. The merge MUST be commutative & associative.

e) Beam `Metrics.counter("validation", "valid"|"invalid"|"<rule_id>")`
   for every named rule, plus `Metrics.distribution(...)` for numeric
   columns. These surface in Cloud Monitoring without extra plumbing.

f) Side outputs:
   - main      → `WriteToBigQuery(landing_table, schema=SCHEMA_FROM_PYDANTIC,
                  create_disposition=CREATE_IF_NEEDED,
                  write_disposition=WRITE_APPEND,
                  method=FILE_LOADS)`  ← use FILE_LOADS for batch synthetic
   - invalid   → `WriteToBigQuery(dead_letter_table, ...)` partitioned by
                  DAY on `dlq_inserted_at`, clustered on (error_type, rule_id)
   - profile   → single-row write to `synthetic_data_quality.column_profiles`
   - BigQueryIO `FailedRows` side output → second-line DLQ for BQ-side
                  rejections (schema mismatches at load time)

g) Fail-fast policy: if `invalid_count / total_count > threshold_blocker`,
   the pipeline writes a BLOCKER row to `synthetic_data_quality.validation_runs` and
   raises a `BlockerThresholdExceeded` exception so Dataflow marks the job
   FAILED. Lower-severity breaches do NOT fail the job; they fail Mode B.

### 5.3 Mode B — Post-Write Validation (warehouse) -- (to be validated)

Triggered by Dataflow job SUCCESS (Airflow sensor / Dagster asset check /
Cloud Function on Dataflow completion event). Steps:

a) GX Core 1.x Checkpoint:
context     = gx.get_context(mode="file")
ds          = context.data_sources.add_or_update_sql(
name="bq_synth",
connection_string=f"bigquery://{PROJECT}/{DATASET}")
asset       = ds.add_table_asset(name="synth_table", table_name=TABLE)
batch_def   = asset.add_batch_definition_whole_table("whole_table")
suite       = context.suites.get(SUITE_NAME)
v_def       = context.validation_definitions.add(
gx.ValidationDefinition(
data=batch_def, suite=suite, name=VDEF_NAME))
checkpoint  = context.checkpoints.add(
gx.Checkpoint(name=CP_NAME,
validation_definitions=[v_def],
actions=[UpdateDataDocsAction(...)]))
result      = checkpoint.run()

b) Soda Core scan: declarative `checks.yml` per table; run via
   `soda scan -d bigquery_warehouse -c configuration.yml checks.yml`.
   Exit code drives CI gating.

c) INFORMATION_SCHEMA baselines (cheap, always run):
   - `INFORMATION_SCHEMA.TABLES`   — row_count, size_bytes
   - `INFORMATION_SCHEMA.COLUMNS`  — schema drift detection
   - `INFORMATION_SCHEMA.PARTITIONS` — partition row counts vs. prior run

d) SDMetrics fidelity (synthetic-specific, MUST run if reference dataset
   is configured):
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
diag    = DiagnosticReport(); diag.generate(real, synth, metadata)
quality = QualityReport();    quality.generate(real, synth, metadata)
   Persist scores (Column Shapes, Column Pair Trends) per column to
   `synthetic_data_quality.fidelity_metrics`.

e) Evidently drift report (current API):
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
snapshot = report.run(reference_data=baseline_df, current_data=synth_df)
snapshot.save_html(gcs_path); snapshot.save_json(...)


## 6. SHARED VALIDATION CHECK CATALOG

Every check declares: `id`, `dimension`, `severity`, `threshold`, `owner`,
`mode` (one of `A`, `B`, `BOTH`).

| ID            | Dimension     | Mode | Severity | Threshold (prod)        |
|---------------|---------------|------|----------|-------------------------|
| schema.types  | Validity      | BOTH | BLOCKER  | 100% type match         |
| schema.order  | Validity      | B    | CRITICAL | exact column order      |
| null.required | Completeness  | BOTH | BLOCKER  | 0% on required          |
| null.optional | Completeness  | B    | MAJOR    | ≤ 5% per column         |
| pk.duplicate  | Uniqueness    | BOTH | BLOCKER  | 0 duplicates            |
| range.numeric | Validity      | BOTH | CRITICAL | within [min,max]        |
| enum.member   | Validity      | BOTH | CRITICAL | 100% in declared set    |
| regex.format  | Validity      | BOTH | MAJOR    | ≥ 99.9% match           |
| fk.exists     | Consistency   | B    | CRITICAL | 100% parent existence   |
| logic.cross   | Consistency   | BOTH | CRITICAL | end_date >= start_date  |
| rowcount.sla  | Completeness  | B    | MAJOR    | ±5% of expected         |
| freshness     | Timeliness    | B    | MAJOR    | generated_at ≤ SLA      |
| drift.psi     | Accuracy      | B    | MAJOR    | < 0.25                  |
| drift.ks      | Accuracy      | B    | MAJOR    | p > 0.01                |
| fidelity.qr   | Accuracy      | B    | CRITICAL | SDMetrics ≥ 0.80        |
| privacy.dcr   | Accuracy      | B    | CRITICAL | DCR ≥ threshold         |

Thresholds MUST be environment-scoped (`dev` / `staging` / `prod`) and
loaded from a single `thresholds.yml`.

## 7. REUSABLE TEMPLATES (DELIVERABLES) -- (to be validated accordingly to the architecture and adapted to the project structure)

- `contracts/record_model.py`              — Pydantic v2 model
- `codegen/derive_pandera.py`              — Pydantic → Pandera schema
- `codegen/derive_bq_ddl.py`               — Pydantic → BQ DDL
- `codegen/derive_gx_suite.py`             — Pydantic → GX Expectation Suite
- `beam/dofns/validate_record.py`          — ValidateRecordDoFn (Mode A)
- `beam/dofns/pandera_batch.py`            — PanderaValidateBatchDoFn (Mode A)
- `beam/dofns/whylogs_profile.py`          — Profile + Merge combiner (Mode A)
- `beam/pipeline.py`                       — full Dataflow pipeline
- `gx/checkpoints/synth_table.py`          — GX 1.x Checkpoint (Mode B)
- `soda/checks/synth_table.yml`            — SodaCL declarative (Mode B)
- `evidently/drift_report.py`              — Evidently snapshot (Mode B)
- `sdmetrics/quality_report.py`            — Fidelity scorecard (Mode B)
- `sql/information_schema_baselines.sql`   — Metadata baselines (Mode B)
- `thresholds.yml`                         — env-scoped thresholds

## 8. FAILURE THRESHOLDS & GATING

| Severity   | Mode A behavior                | Mode B behavior              |
|------------|--------------------------------|------------------------------|
| BLOCKER    | Raise → Dataflow job FAILED    | Block CI promotion, page    |
| CRITICAL   | Tag DLQ; continue; emit metric | Fail CI; allow manual override |
| MAJOR      | Counter only                   | Warn + alert; continue       |
| MINOR      | Counter only                   | Trend only                   |

## 9. CI/CD STRATEGY

- pre-commit: ruff, mypy, pydantic model lint, pandera schema lint,
  `great_expectations suite list` smoke check, `soda scan --dry-run`.
- PR: pytest + Hypothesis on validators; run Beam pipeline on
  `DirectRunner` against fixture; execute Mode A checks; assert metrics.
- Pre-deploy (staging): launch Dataflow job against staging dataset;
  on success, run Mode B GX Checkpoint + Soda scan + SDMetrics +
  Evidently against staging baseline; block on BLOCKER/CRITICAL.
- Production: same as staging plus PagerDuty paging on BLOCKER and
  Slack on CRITICAL/MAJOR; Looker Studio dashboard reads from
  `synthetic_data_quality.*` tables.
- Schema evolution: ADR + Pydantic version bump; backward-compat checked
  via `buf` (if Protobuf) or contract diff job.

## 10. AUDIT & REPORT STRUCTURE (BigQuery `synthetic_data_quality` dataset)
Potential table schemas:

### Table: `synthetic_data_quality.validation_runs`
- run_id (STRING, PK), mode (A|B), pipeline_name, pipeline_version,
  dataflow_job_id, git_sha, dataset_ref, table_ref,
  started_at, finished_at, overall_status (PASS|WARN|FAIL),
  blocker_count, critical_count, major_count, minor_count

### Table: `synthetic_data_quality.check_results`
- run_id (FK), check_id, dimension, severity, mode, column_name,
  expected, actual, threshold, passed (BOOL),
  engine (pandera|gx|soda|evidently|sdmetrics|whylogs|info_schema)

### Table: `synthetic_data_quality.column_profiles`
- run_id, column_name, row_count, null_count, distinct_count,
  min, max, mean, stddev, p50, p95, p99, sum_total (← continuous
  column-total assessment), whylogs_profile_uri

### Table: `synthetic_data_quality.drift_metrics`
- run_id, column_name, psi, wasserstein, ks_stat, ks_pvalue,
  js_divergence, baseline_run_id

### Table: `synthetic_data_quality.fidelity_metrics`
- run_id, column_name, column_shape_score, column_pair_score,
  overall_quality_score, diagnostic_score

### Table: `synthetic_data_quality.dead_letter`
- dlq_inserted_at (PARTITION DAY), run_id, raw_record (JSON),
  error_type, error_detail, rule_id, pipeline_step
- CLUSTER BY (error_type, rule_id)

### Reports artefacts
- GX Data Docs → `gs://{bucket}/gx-docs/{run_id}/` (signed URL)
- Evidently HTML/JSON → `gs://{bucket}/evidently/{run_id}/`
- SDMetrics report .pkl + HTML viz → `gs://{bucket}/sdmetrics/{run_id}/`
- whylogs profile (.bin) → `gs://{bucket}/whylogs/{run_id}/`
- Looker Studio dashboard joins the five tables above on `run_id`.

## 11. DELIVERABLES

1. Architecture diagram (Mermaid) — both modes
2. Repository layout with module boundaries
3. Pydantic ↔ Pandera ↔ BQ DDL ↔ GX Suite code-gen utility
4. All templates listed in §7
5. CI workflow files (GitHub Actions or Cloud Build)
6. Terraform/IaC for `synthetic_data_quality` dataset, GCS buckets,
   Dataplex scans, Looker Studio data sources
7. Runbook per severity
8. ADR template for schema evolution

## 12. NON-FUNCTIONAL REQUIREMENTS

- Mode A validation overhead ≤ 10% of pipeline wall-clock
- whylogs profile merge MUST be commutative & associative
- All checks version-controlled; zero ad-hoc SQL in production
- Secrets via Secret Manager; no creds in YAML/configs
- Reports retained ≥ 13 months; DLQ partition expiry 90 days
- PII MUST NOT appear in profiles or reports (allow-list columns)
- BigQuery writes via FILE_LOADS for batch synthetic generation;
  streaming inserts only with explicit justification
- Pipeline MUST emit OpenLineage events at start/complete/fail

## 13. OUT OF SCOPE

- Unstructured data (text/image/audio) validation
- Sub-second streaming validation (batch + micro-batch only)
- Model-level monitoring (covered by separate ML monitoring spec)
```

## 2. BUILD the DOCKER IMAGE PYTHON and basic CICD github actions (to be filled out by the user)

We need to build the Docker image for the Apache Beam pipeline that will run on DataFlow. This image should include all
the necessary dependencies, including the LLM libraries and any other Python packages required for the synthetic data
generation process. Thus, checking this page in the finest detail is EXTREMELY CRITICAL!!!:

- https://docs.cloud.google.com/dataflow/docs/gpu
- https://docs.cloud.google.com/dataflow/docs/gpu/gpu-support
- https://docs.cloud.google.com/dataflow/docs/gpu/develop-with-gpus
- https://docs.cloud.google.com/dataflow/docs/gpu/use-gpus
- https://docs.cloud.google.com/dataflow/docs/gpu/gpu-metrics
- https://docs.cloud.google.com/dataflow/docs/gpu/use-l4-gpus
- https://docs.cloud.google.com/dataflow/docs/gpu/use-nvidia-mps
- https://docs.cloud.google.com/dataflow/docs/tutorials/satellite-images-gpus
- https://docs.cloud.google.com/dataflow/docs/gpu/troubleshoot-gpus

I have not deployed a custom Docker image in DataFlow (PYTHON) before (only in DataFlow SCIO), so I need to deep dive
into the documentation and understand the best practices for building and deploying custom images for Python DataFlow,
especially when it comes to GPU support and LLM libraries.

although it would not be necessary, this is my current one for SCIO including the github action

```Dockerfile
# Global Artifact Registry Dataflow: https://console.cloud.google.com/gcr/images/dataflow-templates-base/GLOBAL/java11-template-launcher-base
FROM gcr.io/dataflow-templates-base/java11-template-launcher-base-distroless:flex_templates_base_image_release_20250514_RC00

ARG FLEX_TEMPLATE_JAVA_MAIN_CLASS_ARG="ERROR: no java main class arg was provided to docker build"
ARG ENV_NAME_ARG="ERROR: no env name was provided to docker build"
ARG DATAFLOW_JOB_JAR_WITH_DEPENDENCIES_PATH_ARG="ERROR: no jar path was provided to docker build"
ARG JAR_NAME_ARG="ERROR: no jar name arg was provided to docker build"
#ARG WORKDIR=/dataflow/template

# for settings_workbench.xml (KieContainer)
ARG ARTIFACTORY_RELEASER_USER_ARG="ERROR: no env name was provided for ARTIFACTORY_RELEASER_USER_ARG"
ARG ARTIFACTORY_RELEASER_PWD_ARG="ERROR: no env name was provided for ARTIFACTORY_RELEASER_PWD_ARG"
ARG ARTIFACTORY_HOSTNAME_ARG="ERROR: no env name was provided for ARTIFACTORY_HOSTNAME_ARG"
ARG ARTIFACTORY_DEVELOPER_USER_ARG="ERROR: no env name was provided for ARTIFACTORY_DEVELOPER_USER_ARG"
ARG ARTIFACTORY_DEVELOPER_PWD_ARG="ERROR: no env name was provided for ARTIFACTORY_DEVELOPER_PWD_ARG"

#RUN mkdir -p ${WORKDIR}
#WORKDIR ${WORKDIR}

ADD $DATAFLOW_JOB_JAR_WITH_DEPENDENCIES_PATH_ARG ./job-with-dependencies
ADD .m2/settings_workbench.xml ./m2

#RUN echo "$( ls -all ./ )"
#RUN echo "$( ls -all ./job-with-dependencies/ )"

ENV FLEX_TEMPLATE_JAVA_MAIN_CLASS="${FLEX_TEMPLATE_JAVA_MAIN_CLASS_ARG}"
ENV FLEX_TEMPLATE_JAVA_CLASSPATH="./job-with-dependencies/*"
# below env var is not used at this point
ENV ENV=$ENV_NAME_ARG

# Environment variables used by settings_workbench.xml:
ENV ARTIFACTORY_RELEASER_USER=$ARTIFACTORY_RELEASER_USER_ARG
ENV ARTIFACTORY_RELEASER_PS=$ARTIFACTORY_RELEASER_PWD_ARG
ENV ARTIFACTORY_HOSTNAME=$ARTIFACTORY_HOSTNAME_ARG
ENV ARTIFACTORY_DEVELOPER_USER=$ARTIFACTORY_DEVELOPER_USER_ARG
ENV ARTIFACTORY_DEVELOPER_PS=$ARTIFACTORY_DEVELOPER_PWD_ARG

ENTRYPOINT ["./job-with-dependencies/$JAR_NAME_ARG.jar"]
```

```build.yaml
      - name: Setup SBT
        env:
          SBT_VERSION: 1.8.3
          ARTIFACTORY_USER: ${{ steps.get-secrets.outputs.artifactory_developer_USER }}
          ARTIFACTORY_PS: ${{ steps.get-secrets.outputs.artifactory_developer_PS }}
        run: |
          mkdir sbt
          mkdir -p ~/.sbt
          mkdir -p ~/.sbt/1.0/plugins
          curl -kL "https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz" -o "sbt-${SBT_VERSION}.tgz"
          tar xfz sbt-${SBT_VERSION}.tgz
          echo "$(pwd)/sbt/bin" >> $GITHUB_PATH
          cp .github/.sbt/* ~/.sbt/
          echo "realm=Artifactory Realm" >> ~/.sbt/.credentials
          echo "host=${{ env.ARTIFACTORY_HOSTNAME }}" >> ~/.sbt/.credentials
          echo "user=${{ env.ARTIFACTORY_USER }}" >> ~/.sbt/.credentials
          echo "password=${{ env.ARTIFACTORY_PS }}" >> ~/.sbt/.credentials
          echo "credentials += Credentials(Path.userHome / \".sbt\" / \".credentials\")" >> ~/.sbt/1.0/plugins/credentials.sbt
          export SBT_CREDENTIALS="~/.sbt/.credentials"

      - name: Run SBT PACK
        env:
          PROJECT_ID: ${{ steps.environment.outputs.GOOGLE_PROJECT_DBC }}
        run: |-
          export no_proxy="${no_proxy},the_artifactory.com"
          export NO_PROXY="${NO_PROXY},the_artifactory.com"
          export HTTPS_PROXY=""
          export https_proxy=""
          export HTTP_PROXY=""
          export http_proxy=""
          export SBT_OPTS="-Xss4m -XX:ReservedCodeCacheSize=512M -XX:MinRAMPercentage=20.0 -XX:MaxRAMPercentage=80.0 -XX:InitialRAMPercentage=40.0"
          sbt -v -Dsbt.insecureprotocol=true -Dsbt.override.build.repos=true -Dsbt.boot.credentials="~/.sbt/.credentials" -Dsbt.color=false -Dbigquery.project=${PROJECT_ID} -Dbigquery.types.debug=false clean ${{ env.APP_FOLDER }}/test ${{ env.APP_FOLDER }}/pack
          echo "all .jars for Dockers under ls -la ${{ env.APP_FOLDER }}/target/pack/lib"
          ls -la ${{ env.APP_FOLDER }}/target/pack/lib
          
            - name: 'Build ${{ env.BUILD_TYPE }} Image for env: ${{ env.ENV_NAME }}'
        env:
          ARTIFACTORY_RELEASER_USER: ${{ steps.get-secrets.outputs.artifactory_release_USER }}
          ARTIFACTORY_RELEASER_PS: ${{ steps.get-secrets.outputs.artifactory_release_PS }}
          ARTIFACTORY_DEVELOPER_USER: ${{ steps.get-secrets.outputs.artifactory_developer_USER }}
          ARTIFACTORY_DEVELOPER_PS: ${{ steps.get-secrets.outputs.artifactory_developer_PS }}
          DATAFLOW_JOB_JAR_WITH_DEPENDENCIES_PATH: '${{ env.APP_FOLDER }}/target/pack/lib'
          JAR_NAME: '${{ env.APP_FOLDER }}_2.12-${{ env.PROJECT_VERSION }}'
        run: |-
          echo "Logging into docker"
          echo ${ARTIFACTORY_RELEASER_PS} | docker login --password-stdin --username ${ARTIFACTORY_RELEASER_USER}  ${ARTIFACTORY_BASE_URL}/

          echo "Triggering docker build"
          DOCKER_BUILDKIT=1 \
          docker build -t ${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/${{ env.CONTAINER_IMAGE_NAME }}:${{ env.RELEASE_VERSION }} \
          --cache-from ${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/${{ env.CONTAINER_IMAGE_NAME }}:latest \
          --build-arg ENV_NAME_ARG=${{ env.ENV_NAME }} \
          --build-arg FLEX_TEMPLATE_JAVA_MAIN_CLASS_ARG=${{ env.FLEX_TEMPLATE_JAVA_MAIN_CLASS }} \
          --build-arg DATAFLOW_JOB_JAR_WITH_DEPENDENCIES_PATH_ARG=$DATAFLOW_JOB_JAR_WITH_DEPENDENCIES_PATH \
          --build-arg ARTIFACTORY_RELEASER_USER_ARG=$ARTIFACTORY_RELEASER_USER \
          --build-arg ARTIFACTORY_RELEASER_PWD_ARG=$ARTIFACTORY_RELEASER_PS \
          --build-arg ARTIFACTORY_HOSTNAME_ARG=$ARTIFACTORY_HOSTNAME \
          --build-arg ARTIFACTORY_DEVELOPER_USER_ARG=$ARTIFACTORY_DEVELOPER_USER \
          --build-arg ARTIFACTORY_DEVELOPER_PWD_ARG=$ARTIFACTORY_DEVELOPER_PS \
          --build-arg JAR_NAME_ARG=$JAR_NAME \
          .

          echo "Show docker images"
          docker images

          echo "Push docker image to: ${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/${{ env.CONTAINER_IMAGE_NAME }}:${{ env.RELEASE_VERSION }}"
          docker push ${ARTIFACTORY_HOSTNAME}/dkr-public-local/${ARTIFACTORY_NAMESPACE}/${{ env.CONTAINER_IMAGE_NAME }}:${{ env.RELEASE_VERSION }}
            - name: sbt assembly and curl fat.jar in JFrog
        env:
          PROJECT_ID: ${{ steps.environment.outputs.GOOGLE_PROJECT_DBC }}
          ARTIFACTORY_RELEASER_USER: ${{ steps.get-secrets.outputs.artifactory_release_USER }}
          ARTIFACTORY_RELEASER_PS: ${{ steps.get-secrets.outputs.artifactory_release_PS }}
          APP_FOLDER: '${{ env.APP_FOLDER }}'
          FAT_JAR_PATH: '${{ env.APP_FOLDER }}/target/scala-2.12'
          FAT_JAR_NAME: 'fat-${{ env.APP_FOLDER }}-${{ env.PROJECT_VERSION }}.jar'
        run: |
          if [[ "${{ env.BUILD_TYPE }}" == "Release" ]]; then
            echo "sbt assembly [ ${{ env.APP_FOLDER }} ]"
            export no_proxy="${no_proxy},the_artifactory"
            export NO_PROXY="${NO_PROXY},the_artifactory"
            export HTTPS_PROXY=""
            export https_proxy=""
            export HTTP_PROXY=""
            export http_proxy=""
            sbt -v -Dsbt.insecureprotocol=true -Dsbt.override.build.repos=true -Dsbt.boot.credentials="~/.sbt/.credentials" -Dsbt.color=false -Dbigquery.project=${PROJECT_ID} -Dbigquery.types.debug=false -DbuildType=${{ env.APP_FOLDER }} clean ${{ env.APP_FOLDER }}/assembly
          
            echo -e "\nls -la ${APP_FOLDER}/target"
            ls -la ${APP_FOLDER}/target
            echo -e "\nLooking for $FAT_JAR_NAME inside ${FAT_JAR_PATH}"
            ls -la ${FAT_JAR_PATH}
          
            echo "curl -u fat.jar as $FAT_JAR_NAME"
            curl -u $ARTIFACTORY_RELEASER_USER:$ARTIFACTORY_RELEASER_PS -T "$FAT_JAR_PATH/$FAT_JAR_NAME" "https://$ARTIFACTORY_HOSTNAME/artifactory/mvn-libs-release/com/db/pwcclakees/$APP_FOLDER/${{ env.PROJECT_VERSION }}/$FAT_JAR_NAME"
          else
            echo "[ ${{ env.APP_FOLDER }} ] no need of sbt assembly"
          fi
      
      - name: 'Build ${{ env.BUILD_TYPE }} flex-template for env: ${{ env.ENV_NAME }}'
        env:
          PROJECT_ID: ${{ steps.environment.outputs.GOOGLE_PROJECT_DBC }}
          CONTAINER_IMAGE_NAME: ${{ needs.init-vars.outputs.CONTAINER_IMAGE_NAME }}
        run: |-
          echo "Logging into docker"
          echo "Creating Data flow Template : ${{ env.CONTAINER_IMAGE_NAME }}"
          
          gcloud dataflow flex-template build ${{ env.DF_TEMPLATE_PATH }} \
                --image="${{ env.ARTIFACTORY_HOSTNAME }}/dkr-public-local/${{ env.ARTIFACTORY_NAMESPACE }}/${{ env.CONTAINER_IMAGE_NAME }}:${{ env.RELEASE_VERSION }}" \
                --image-repository-username-secret-id="projects/${PROJECT_ID}/secrets/ARTIFACTORY_RELEASER_USER" \
                --image-repository-password-secret-id="projects/${PROJECT_ID}/secrets/ARTIFACTORY_RELEASER_PS" \
                --sdk-language=JAVA \
                --staging-location="gs://db-${{ env.ENV_NAME }}-${{ env.PROJECT_BUCKET_SUFFIX }}-dataflow/staging" \
                --temp-location="gs://db-${{ env.ENV_NAME }}-${{ env.PROJECT_BUCKET_SUFFIX }}-dataflow/temp" \
                --dataflow-kms-key="pthe neys"

```

# 3. THE CLAUDE CODE SET UP in LOCAL COMPUTER (harness) and git pull in remote computer

We need to define an entire spec driven harness in claude code. I think we need to include skills.md and maybe
sub-agents (as a team of sub-agents, what do you think? would it make sense at some stages?), that at time can work in
parallel (if needed, with mocked datasets to validate the solution with lightweight Python, minimising the git pull from
other machines and advance as much as possible). As well the best practices in the industry. Please create it as needed.
including claude.md and skills.md, and any other file you think that would be relevant for the project.

As we have some many modules can be really relevant.

PLEASE NOTE (VERY IMPORTANT): This laptop we will run the basic stuff (and will implement the code), it is not powerful
enough to run the LLMs locally and I have no access to a GPU or the DEV GCP project. All the features will be "git
pulled" from another machine (M4 PRO 24GB of RAM) that can also access (deploy, run from local connecting into bq and
dataflow) the GCP project, and that will be used for the actual development and testing of the code. So we need to make
sure that the code is modular, applying the best practices in the industry.

Use the best practices in the industry for structuring the code, including modularization, separation of concerns, and
clear documentation. We should also include unit tests and integration tests to ensure the quality and reliability of
the code.

## THE ROADMAP

As this is a very massive project, we need to break it down into smaller tasks and milestones (focusing especially on
the FEATURE_B at the beginning) and keep track of all the progress and specs of the project in a clear and organized
way. Please progress when you can with the tasks (I will give you feedback when needed, unless it is blocker for the
next steps). do we need a sub-agent or skill or both? post hook? others? apply the best standard token efficient to keep
this up to date, so we can keep track among sessions even.

# WRAP UP

Please drill me with anything 