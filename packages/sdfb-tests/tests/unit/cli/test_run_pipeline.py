"""Unit tests for `sdfb_beam.cli.run_pipeline` — arg parsing and factory."""

from __future__ import annotations

import pytest

from sdfb_beam.cli.run_pipeline import build_model_client, parse_args


def _common_args() -> list[str]:
    return [
        "--ddl_uri", "gs://bucket/ddl.json",
        "--reference_table", "p.d.t",
        "--landing_table", "p.d.landing",
        "--dlq_table", "p.d.dlq",
        "--num_rows", "100",
        "--run_id", "abc-123",
        "--model_uri", "gs://bucket/models/gemma4/e4b/v1/",
    ]


def test_parse_args_minimal():
    args, beam_argv = parse_args(_common_args())
    assert args.ddl_uri == "gs://bucket/ddl.json"
    assert args.num_rows == 100
    assert args.engine == "b1_rag"        # default
    assert args.batch_size == 16          # default
    assert args.similarity == 0.5         # default
    assert args.client_type == "vllm"     # default
    assert beam_argv == []


def test_parse_args_overrides():
    argv = _common_args() + [
        "--engine", "b2_library",
        "--batch_size", "32",
        "--similarity", "0.9",
        "--client_type", "fake",
        "--reference_rows_limit", "500",
    ]
    args, _ = parse_args(argv)
    assert args.engine == "b2_library"
    assert args.batch_size == 32
    assert args.similarity == 0.9
    assert args.client_type == "fake"
    assert args.reference_rows_limit == 500


def test_parse_args_passes_unknown_to_beam():
    argv = _common_args() + ["--runner", "DirectRunner", "--project", "demo"]
    _, beam_argv = parse_args(argv)
    assert "--runner" in beam_argv
    assert "DirectRunner" in beam_argv


def test_parse_args_rejects_unknown_engine_value():
    """argparse-level rejection of bad client_type; engine is free-form."""
    argv = _common_args() + ["--client_type", "made-up"]
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_build_model_client_fake():
    """Fake client builds without touching vLLM / MLX imports."""
    c = build_model_client("fake", "ignored")
    assert c is not None


def test_build_model_client_vllm_stub_does_not_fail_on_import():
    """The vllm_client stub must be importable on M4 laptop (no vllm dep).
    Constructor succeeds; generate_json raises NotImplementedError but is
    not exercised here."""
    c = build_model_client("vllm", "gs://bucket/models/foo/")
    assert c.model_uri == "gs://bucket/models/foo/"


def test_build_model_client_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown client_type"):
        build_model_client("openai", "ignored")
