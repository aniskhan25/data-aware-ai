"""The run-summary schema.

Every experiment writes one JSON summary with this shape. The comparison and reporting
tools read only summaries, never logs, so this is the contract between the parts.

Two design points carry the lesson:

* **Metric names carry their unit** (`_seconds`, `_bytes`, `mib_per_second`), so a
  reader never has to guess.
* **Correctness counters are always present.** A fast-looking run cannot hide incorrect
  sample assignment by omitting the field that would show it.

Unknown field names are rejected, so a renamed metric fails loudly instead of silently
disappearing from a report. Field *types* are not checked: a wrong type breaks visibly
at the point of use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import SummaryError

SCHEMA_VERSION = "1.0"

#: Present in every summary. This literal *is* the schema: the whole shape of a run
#: summary, with its defaults, in one readable place.
COMMON = {
    "schema_version": SCHEMA_VERSION,
    "run_name": "",
    "timestamp_utc": "",
    "git_commit": "",
    "config_path": "",
    "config_hash": "",
    "hostname": "",
    "slurm_job_id": "",
    "layout": "",
    "storage": "",
    "world_size": 1,
    "rank": 0,
    "num_workers": 0,
    "warmup_batches": 0,
    "measured_batches": 0,
    "samples_measured": 0,
    "bytes_read": 0,
    "startup_seconds": 0.0,
    "staging_seconds": 0.0,
    "measured_seconds": 0.0,
    "samples_per_second": 0.0,
    "mib_per_second": 0.0,
    "mean_batch_wait_seconds": 0.0,
    "p95_batch_wait_seconds": 0.0,
    "mean_data_wait_fraction": 0.0,
    "peak_memory_bytes": 0,
    "failed_samples": 0,
    "duplicate_samples": 0,
    "missing_samples": 0,
}

#: Fields a run may add. Grouped by the part that produces them.
OPTIONAL = frozenset(
    {
        # Provenance and configuration echo.
        "manifest_path", "manifest_hash", "resolved_config", "slurm_context",
        "cpus_available", "allocated_cores", "threads_per_core", "batch_size",
        "prefetch_factor", "persistent_workers", "shuffle", "shuffle_buffer", "seed",
        "compute_steps", "notes", "total_samples", "adapter",
        # Loader detail.
        "batches_measured", "files_opened", "batches_per_second",
        "median_batch_wait_seconds", "max_batch_wait_seconds", "compute_seconds",
        "warmup_seconds", "unique_samples", "measured_epochs",
        # CPU and scheduling (Part IV).
        "user_cpu_seconds", "system_cpu_seconds", "cpu_utilization",
        "voluntary_context_switches", "involuntary_context_switches",
        "involuntary_switches_per_second", "child_processes",
        "oversubscription_ratio", "processes_per_physical_core",
        # Layouts and artifacts (Part III, optional tracks).
        "num_shards", "mean_shard_bytes", "min_shard_bytes", "max_shard_bytes",
        "shard_bytes_cv", "shard_work_cv", "samples_per_shard",
        "max_samples_per_shard", "shard_opens", "shard_open_seconds",
        "filesystem_objects", "image_bytes", "mount_seconds", "artifact_bytes",
        "row_groups", "chunk_size",
        # Storage and staging (Part V).
        "validation_seconds", "estimated_epoch_seconds", "per_epoch_seconds",
        "total_job_seconds", "peak_tmp_bytes", "staged_bytes", "staged_files",
        "staging_source", "staging_destination", "memory_allocated_bytes",
        "memory_source", "dataset_fraction_of_memory", "safety_fraction",
        # Distributed (Part VI).
        "rank_summaries", "min_rank_throughput", "max_rank_throughput",
        "rank_throughput_spread", "min_rank_elapsed_seconds",
        "max_rank_elapsed_seconds", "rank_elapsed_spread", "max_data_wait_fraction",
        "coverage_fraction", "idle_ranks", "partitioning_valid",
    }
)

#: Printed as KEY=VALUE, so job logs stay greppable.
KEYVALUE_ORDER = (
    "layout", "storage", "world_size", "num_workers", "samples_measured",
    "bytes_read", "samples_per_second", "mib_per_second", "startup_seconds",
    "mean_batch_wait_seconds", "p95_batch_wait_seconds", "mean_data_wait_fraction",
    "failed_samples", "duplicate_samples", "missing_samples",
)


def new_run_summary(**values: Any) -> dict[str, Any]:
    """Build a summary with every common field present."""
    unknown = sorted(set(values) - set(COMMON) - OPTIONAL)
    if unknown:
        raise SummaryError(
            f"unknown summary field(s) {unknown}; add them to OPTIONAL in "
            "dataaware/schema.py if they are a real metric"
        )
    return {**COMMON, **values}


def validate_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Check that a summary has the right field names and schema version."""
    missing = sorted(set(COMMON) - set(summary))
    if missing:
        raise SummaryError(f"summary is missing required field(s) {missing}")
    unknown = sorted(set(summary) - set(COMMON) - OPTIONAL)
    if unknown:
        raise SummaryError(f"summary has unknown field(s) {unknown}")
    if summary["schema_version"] != SCHEMA_VERSION:
        raise SummaryError(
            f"schema_version {summary['schema_version']!r} is not supported "
            f"(expected {SCHEMA_VERSION!r})"
        )
    return summary


def write_run_summary(path: str | os.PathLike[str], summary: dict[str, Any]) -> Path:
    """Validate, then write. An invalid summary is never stored."""
    validate_run_summary(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
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
    """Render selected metrics as uppercase KEY=VALUE lines."""
    lines = []
    for key in keys:
        if key in summary:
            value = summary[key]
            lines.append(f"{key.upper()}={value:.4g}" if isinstance(value, float) else f"{key.upper()}={value}")
    return "\n".join(lines)
