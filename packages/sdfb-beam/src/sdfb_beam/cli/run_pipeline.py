"""Flex Template entrypoint for the synthesis pipeline.

Set via `FLEX_TEMPLATE_PYTHON_PY_FILE` in `docker/Dockerfile`. The Python
launcher invokes this with argparse args populated from the Flex Template
parameters declared in `docker/flex_template_metadata.json`.

Runtime modes:
  - Production (Dataflow + L4 + Gemma 4 via vLLM):
      --runner=DataflowRunner --client_type=vllm --model_uri=gs://…
  - Local smoke on M4 (MLX backend, see docs/M4_LOCAL_SMOKE.md):
      --runner=DirectRunner --client_type=mlx --model_uri=./models/…
  - Deterministic CI integration test (no real LLM):
      --runner=DirectRunner --client_type=fake

REFs:
  - .claude/skills/beam-dofn.md
  - docs/CICD.md
  - docs/MODEL_LAYOUT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import TYPE_CHECKING

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.gcp.bigquery import BigQueryDisposition, WriteToBigQuery
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions,
    PipelineOptions,
    StandardOptions,
)

import sdfb_core.engines  # noqa: F401  populates ENGINE_REGISTRY at import time

from sdfb_beam.io.bq_sources import load_reference_rows
from sdfb_beam.pipeline import PipelineConfig, build_pipeline
from sdfb_core.contracts import TableSchema

if TYPE_CHECKING:
    from sdfb_core.engines import ModelClient

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Synthetic Dataflow BigQuery — pipeline launcher")
    p.add_argument("--ddl_uri", required=True,
                   help="gs:// or local path to _ddl.json")
    p.add_argument("--reference_table", required=True,
                   help="FQN of source table for live SELECT reference rows")
    p.add_argument("--reference_rows_limit", type=int, default=10_000)
    p.add_argument("--landing_table", required=True,
                   help="BQ table for synthetic rows (project.dataset.table)")
    p.add_argument("--dlq_table", required=True,
                   help="BQ DLQ table (project.dataset.table)")
    p.add_argument("--num_rows", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--similarity", type=float, default=0.5)
    p.add_argument("--run_id", required=True)
    p.add_argument("--engine", default="b1_rag",
                   help="Engine name registered in ENGINE_REGISTRY")
    p.add_argument("--model_uri", required=True,
                   help="gs://<bucket>/synthetic/models/<family>/<model>/<version>/")
    p.add_argument("--client_type", default="vllm",
                   choices=["vllm", "mlx", "fake"])
    return p.parse_known_args(argv)


def build_model_client(client_type: str, model_uri: str) -> "ModelClient":
    """Lazy factory — avoids importing vLLM / MLX on machines that don't have them."""
    if client_type == "fake":
        from sdfb_beam.handlers.fake_client import FakeModelClient
        # Empty pool — caller is expected to override for any real smoke test.
        return FakeModelClient(reference_pool=[{}])
    if client_type == "vllm":
        from sdfb_beam.handlers.vllm_client import VLLMModelClient
        return VLLMModelClient(model_uri=model_uri)
    if client_type == "mlx":
        from sdfb_beam.handlers.mlx_client import MLXModelClient
        return MLXModelClient(model_uri=model_uri)
    raise ValueError(f"Unknown client_type: {client_type}")


def load_ddl(ddl_uri: str) -> TableSchema:
    """Load `_ddl.json` from gs:// or local; transparent via Beam FileSystems."""
    with FileSystems.open(ddl_uri) as f:
        return TableSchema.model_validate(json.loads(f.read()))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        force=True,
    )
    args, beam_argv = parse_args(argv or sys.argv[1:])
    options = PipelineOptions(beam_argv)
    runner = options.view_as(StandardOptions).runner or "DataflowRunner"

    # save_main_session=True is needed only for DirectRunner ad-hoc; on
    # Dataflow the image bakes in deps + source so it adds startup cost
    # without value.
    options.view_as(GoogleCloudOptions).save_main_session = (runner == "DirectRunner")
    if runner == "DataflowRunner":
        options.view_as(GoogleCloudOptions).job_name = f"sdfb-{args.run_id}"

    logger.info("Loading DDL from %s", args.ddl_uri)
    table_schema = load_ddl(args.ddl_uri)
    logger.info("Loaded schema for %s (%d columns)",
                table_schema.fqn, len(table_schema.columns))

    logger.info("Building model client (client_type=%s)", args.client_type)
    model_client = build_model_client(args.client_type, args.model_uri)

    logger.info("Loading reference rows from %s (limit=%d)",
                args.reference_table, args.reference_rows_limit)
    reference_rows = load_reference_rows(
        table=args.reference_table,
        limit=args.reference_rows_limit,
    )

    config = PipelineConfig(
        table_schema=table_schema,
        engine_name=args.engine,
        model_client=model_client,
        num_rows=args.num_rows,
        batch_size=args.batch_size,
        similarity=args.similarity,
        run_id=args.run_id,
    )

    landing_sink = WriteToBigQuery(
        table=args.landing_table,
        method=WriteToBigQuery.Method.FILE_LOADS,
        write_disposition=BigQueryDisposition.WRITE_APPEND,
        create_disposition=BigQueryDisposition.CREATE_NEVER,
    )
    dlq_sink = WriteToBigQuery(
        table=args.dlq_table,
        method=WriteToBigQuery.Method.FILE_LOADS,
        write_disposition=BigQueryDisposition.WRITE_APPEND,
        create_disposition=BigQueryDisposition.CREATE_NEVER,
    )

    with beam.Pipeline(options=options) as p:
        result = build_pipeline(
            p,
            reference_rows=reference_rows,
            config=config,
            landing_sink=landing_sink,
            dlq_sink=dlq_sink,
        )
        logger.info(
            "Pipeline launched: run_id=%s reference_digest=%s",
            result["run_id"],
            result["reference_digest"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
