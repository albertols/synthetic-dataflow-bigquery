---
name: ddl-codegen
description: Recipe for the Pydantic ↔ Pandera ↔ BigQuery DDL codegen chain. One source of truth (Pydantic); Pandera and BQ DDL are derived. Load when working on `sdfb_core/contracts/`, `sdfb_core/codegen/`, or anything touching schema derivation.
---

# Skill — DDL codegen

The `_ddl.json` (produced by `bigquery_ddl_metadata.py`) feeds into a Pydantic model. Pandera schemas and BigQuery DDL are **derived** from the Pydantic model — never hand-written. This kills schema drift between validation layers.

```
_ddl.json (BQ INFORMATION_SCHEMA dump)
    │
    ▼
TableSchema (Pydantic — sdfb_core/contracts/schema.py)
    │
    ├──> dynamic GeneratedRecord model (pydantic.create_model) ─> ValidateRecordDoFn
    │
    ├──> Pandera DataFrameSchema (codegen/derive_pandera.py)   ─> PanderaValidateBatchDoFn
    │
    └──> BigQuery TableSchema dict (codegen/derive_bq_ddl.py)  ─> WriteToBigQuery(schema=...)
```

## BQ → Pydantic type map

| BQ type | Pydantic type | Notes |
|---|---|---|
| `STRING` | `str` | If `max_length` known, add `StringConstraints` |
| `INTEGER` / `INT64` | `int` | |
| `FLOAT` / `FLOAT64` | `float` | |
| `BOOLEAN` / `BOOL` | `bool` | |
| `NUMERIC` | `Decimal` | precision/scale enforced via constraints |
| `BIGNUMERIC` | `Decimal` | |
| `DATE` | `date` | |
| `DATETIME` | `datetime` (naive) | |
| `TIMESTAMP` | `datetime` (UTC) | |
| `TIME` | `time` | |
| `BYTES` | `bytes` | |
| `JSON` | `dict[str, Any]` | |
| `STRUCT<...>` | nested Pydantic model | recurse |
| `ARRAY<T>` | `list[T]` | preserve element mode |

Mode handling:
- `REQUIRED` → no default, `None` not allowed.
- `NULLABLE` → `T | None`, default `None`.
- `REPEATED` → `list[T]`, default `[]`.

REF: https://docs.cloud.google.com/bigquery/docs/schemas#creating_a_JSON_schema_file

## Primary keys

`_ddl.json["primary_keys"]` becomes:
1. A `model_validator(mode="after")` on `GeneratedRecord` that enforces the PK is set (non-null on all PK columns).
2. A `unique` check on the PK columns in the Pandera batch validator (enforces uniqueness across the batch).
3. The PK columns are NOT clustered by in the landing table (clustering is independent and column-cardinality-driven).

REF: https://docs.cloud.google.com/bigquery/docs/primary-foreign-keys

## Tests every codegen function must pass

In `packages/sdfb-tests/tests/unit/codegen/`:

1. **Round-trip** — `bq_ddl → Pydantic → bq_ddl` is identity (modulo description normalization).
2. **Pandera ⊇ Pydantic** — the Pandera-derived schema's failure set is a superset of the Pydantic model's failure set on a `hypothesis`-generated fuzz corpus.
3. **Nested structs** — a `STRUCT` of `ARRAY` of `STRUCT` round-trips correctly.
4. **All BQ types** — every type in the map above has a positive and negative test case.

## When the source-of-truth shifts

The Pydantic model is the source of truth at code-generation time. The `_ddl.json` is the source of truth at runtime for *which tables exist*, but the Pydantic model is regenerated from it. Never hand-edit the derived Pandera or BQ DDL.

## References

- BigQuery JSON schema file: https://docs.cloud.google.com/bigquery/docs/schemas#creating_a_JSON_schema_file
- BigQuery PK/FK: https://docs.cloud.google.com/bigquery/docs/primary-foreign-keys
- Pydantic v2 dynamic models: https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation
- Pandera DataFrameSchema: https://pandera.readthedocs.io/en/stable/
- `bigquery_ddl_metadata.py` (root of repo, to be refactored into `sdfb_core/ddl/`)
