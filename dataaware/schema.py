"""The run-summary schema.

Every experiment writes one JSON summary with this shape. Comparison and
reporting tools read only summaries, never logs, so the schema is the contract
between the parts of the tutorial.

Design rules that later phases depend on:

* metric names carry their unit (``_seconds``, ``_bytes``, ``mib_per_second``);
* the schema version is explicit, so comparison tools can refuse mixed versions;
* unknown fields are rejected, so a renamed metric fails loudly;
* correctness counters (failed, duplicate, missing) are always present, so a
  fast-looking run cannot hide incorrect sample assignment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

#: Field name -> accepted Python types. Present in every summary.
COMMON_FIELDS: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "run_name": (str,),
    "timestamp_utc": (str,),
    "git_commit": (str,),
    "config_path": (str,),
    "config_hash": (str,),
    "hostname": (str,),
    "slurm_job_id": (str,),
    "layout": (str,),
    "storage": (str,),
    "world_size": (int,),
    "num_workers": (int,),
    "warmup_batches": (int,),
    "measured_batches": (int,),
    "samples_measured": (int,),
    "bytes_read": (int,),
    "startup_seconds": (float, int),
    "staging_seconds": (float, int),
    "measured_seconds": (float, int),
    "samples_per_second": (float, int),
    "mib_per_second": (float, int),
    "mean_batch_wait_seconds": (float, int),
    "p95_batch_wait_seconds": (float, int),
    "mean_data_wait_fraction": (float, int),
    "peak_memory_bytes": (int,),
    "failed_samples": (int,),
    "duplicate_samples": (int,),
    "missing_samples": (int,),
}

#: Optional fields. Distributed runs add the rank block; layout experiments add
#: shard statistics; storage experiments add staging detail. Validation accepts a
#: summary that omits them but rejects one that misspells them.
OPTIONAL_FIELDS: dict[str, tuple[type, ...]] = {
    # Provenance and context.
    "manifest_path": (str,),
    "manifest_hash": (str,),
    "resolved_config": (dict,),
    "slurm_context": (dict,),
    "cpus_available": (int,),
    "batch_size": (int,),
    "prefetch_factor": (int,),
    "persistent_workers": (bool,),
    "shuffle": (bool,),
    "seed": (int,),
    "compute_steps": (int,),
    "notes": (str,),
    # Loader detail.
    "batches_measured": (int,),
    "files_opened": (int,),
    "batches_per_second": (float, int),
    "median_batch_wait_seconds": (float, int),
    "max_batch_wait_seconds": (float, int),
    "compute_seconds": (float, int),
    "warmup_seconds": (float, int),
    # Distributed block (Part VI).
    "rank": (int,),
    "rank_summaries": (list,),
    "min_rank_throughput": (float, int),
    "max_rank_throughput": (float, int),
    "rank_throughput_spread": (float, int),
    "min_rank_elapsed_seconds": (float, int),
    "max_rank_elapsed_seconds": (float, int),
    "rank_elapsed_spread": (float, int),
    "unique_samples": (int,),
    "max_data_wait_fraction": (float, int),
    # Layout block (Part III).
    "num_shards": (int,),
    "mean_shard_bytes": (float, int),
    "min_shard_bytes": (int,),
    "max_shard_bytes": (int,),
    "shard_bytes_cv": (float, int),
    "shard_work_cv": (float, int),
    "samples_per_shard": (int,),
    "max_samples_per_shard": (int,),
    "shard_opens": (int,),
    "shard_open_seconds": (float, int),
    "shuffle_buffer": (int,),
    "filesystem_objects": (int,),
    "image_bytes": (int,),
    "mount_seconds": (float, int),
    # Storage block (Part V).
    "validation_seconds": (float, int),
    "per_epoch_seconds": (float, int),
    "measured_epochs": (int,),
    "total_job_seconds": (float, int),
    "peak_tmp_bytes": (int,),
}

#: Printed as ``KEY=VALUE`` by the loader and by scripts/summarize_run.py. This is
#: the "expected output shape" the README shows for a baseline run.
KEYVALUE_ORDER = (
    "layout",
    "storage",
    "world_size",
    "num_workers",
    "samples_measured",
    "bytes_read",
    "samples_per_second",
    "mib_per_second",
    "startup_seconds",
    "mean_batch_wait_seconds",
    "p95_batch_wait_seconds",
    "mean_data_wait_fraction",
    "failed_samples",
    "duplicate_samples",
    "missing_samples",
)


class SummaryError(ValueError):
    """Raised when a run summary does not satisfy the schema."""


def new_run_summary(**values: Any) -> dict[str, Any]:
    """Build a summary with every common field present.

    Numeric fields default to zero rather than being omitted, so a comparison
    tool never has to distinguish "missing" from "not measured yet". Anything
    genuinely unknown should be recorded explicitly, not left out.
    """
    summary: dict[str, Any] = {}
    for name, types in COMMON_FIELDS.items():
        if str in types:
            summary[name] = ""
        elif types == (int,):
            summary[name] = 0
        else:
            summary[name] = 0.0
    summary["schema_version"] = SCHEMA_VERSION
    summary["world_size"] = 1

    unknown = set(values) - set(COMMON_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise SummaryError(
            f"unknown summary field(s) {sorted(unknown)}; add them to "
            "dataaware.schema.OPTIONAL_FIELDS if they are a real metric"
        )
    summary.update(values)
    return validate_run_summary(summary)


def validate_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate field presence and types. Returns the summary unchanged."""
    if not isinstance(summary, dict):
        raise SummaryError(f"summary must be a mapping, got {type(summary).__name__}")

    missing = [name for name in COMMON_FIELDS if name not in summary]
    if missing:
        raise SummaryError(f"summary is missing required field(s) {missing}")

    unknown = set(summary) - set(COMMON_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise SummaryError(f"summary has unknown field(s) {sorted(unknown)}")

    for name, value in summary.items():
        types = COMMON_FIELDS.get(name) or OPTIONAL_FIELDS[name]
        # bool is a subclass of int; only accept it where it is declared.
        if isinstance(value, bool) and bool not in types:
            raise SummaryError(f"{name} must be {_names(types)}, got a bool")
        if not isinstance(value, types):
            raise SummaryError(
                f"{name} must be {_names(types)}, got {type(value).__name__}"
            )

    if summary["schema_version"] != SCHEMA_VERSION:
        raise SummaryError(
            f"schema_version {summary['schema_version']!r} is not supported by "
            f"this release (expected {SCHEMA_VERSION!r})"
        )
    if summary["world_size"] < 1:
        raise SummaryError(f"world_size must be >= 1, got {summary['world_size']}")
    # measured_batches is the requested window, which the configuration loader
    # already constrains; it is not re-checked here. The counters below are
    # results, so an impossible value means the measurement itself is wrong.
    for name in ("samples_measured", "bytes_read", "failed_samples", "peak_memory_bytes"):
        if summary[name] < 0:
            raise SummaryError(f"{name} must be >= 0, got {summary[name]}")
    fraction = summary["mean_data_wait_fraction"]
    if not 0.0 <= fraction <= 1.0:
        raise SummaryError(f"mean_data_wait_fraction must be in [0, 1], got {fraction}")
    return summary


def write_run_summary(path: str | os.PathLike[str], summary: dict[str, Any]) -> Path:
    """Validate then write. Invalid summaries are never persisted."""
    validate_run_summary(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def read_run_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    try:
        summary = json.loads(path.read_text())
    except FileNotFoundError:
        raise SummaryError(f"run summary not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SummaryError(f"{path}: invalid JSON: {exc}") from None
    return validate_run_summary(summary)


def format_keyvalue(summary: dict[str, Any], keys: tuple[str, ...] = KEYVALUE_ORDER) -> str:
    """Render selected metrics as uppercase ``KEY=VALUE`` lines."""
    lines = []
    for key in keys:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            value = f"{value:.4g}"
        lines.append(f"{key.upper()}={value}")
    return "\n".join(lines)


def _names(types: tuple[type, ...]) -> str:
    return " or ".join(t.__name__ for t in types)
