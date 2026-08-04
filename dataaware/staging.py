"""Staging a dataset to node-local storage, with the cost included.

Compute-node ``/tmp`` on LUMI lives in memory and is charged against the job's
allocation. Two consequences drive everything here:

* **Staging can fail the job.** A dataset larger than the allocation leaves no room
  for the workload. This module refuses to stage in that case rather than letting the
  job die partway through a copy.
* **Staging is never free.** The copy has to be paid for before the first sample is
  read, so a comparison that reports only steady-state throughput will recommend
  staging for workloads that never recover the cost. Staging and validation time are
  measured and returned, not hidden.

Data in node-local storage disappears when the job ends, so results must be written
back to shared storage before the job exits. Cleanup runs in a ``finally`` block.
"""

from __future__ import annotations

import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .inspection import detect_allocated_memory
from .manifest import Sample

#: Largest share of allocated memory a staged dataset may occupy. Node-local /tmp is
#: memory, so the rest of the allocation still has to hold the workload: the model,
#: the batches in flight, the worker processes, and the framework itself.
DEFAULT_SAFETY_FRACTION = 0.5


class StagingRefused(RuntimeError):
    """Raised when staging would be unsafe, before anything is copied."""


class StagingFailed(RuntimeError):
    """Raised when a staged copy is incomplete or does not match the source."""


def resolve_tmp_dir(configured: str = "") -> Path:
    """Choose the node-local directory to stage into.

    Prefers an explicit setting, then Slurm's per-job temporary directory, then a
    job-scoped path under ``/tmp``. The job ID is included so that two jobs sharing a
    node cannot collide or delete each other's data.
    """
    if configured:
        return Path(configured)
    for name in ("SLURM_TMPDIR", "TMPDIR"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return Path(f"/tmp/daai-{job}")


def artifact_bytes(path: Path) -> tuple[int, int]:
    """Size of what would be staged, as ``(bytes, files)``."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    files = 0
    for directory, _, names in os.walk(path):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def check_safety(
    dataset_bytes: int,
    memory_bytes: int | None,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    memory_source: str = "",
) -> dict[str, Any]:
    """Decide whether staging this dataset is safe. Raises if it is not.

    An unknown memory allocation is treated as unsafe. Node-local ``/tmp`` is memory,
    and staging a dataset of unknown relative size is exactly how a job dies halfway
    through a copy with a confusing out-of-memory error.
    """
    if not 0.0 < safety_fraction <= 1.0:
        raise StagingRefused(f"safety_fraction must be in (0, 1], got {safety_fraction}")

    if memory_bytes is None or memory_bytes <= 0:
        raise StagingRefused(
            "Refusing to stage: the job's memory allocation could not be determined "
            f"({memory_source or 'unknown'}).\n"
            "Compute-node /tmp is memory and is charged against the allocation, so "
            "staging a dataset of unknown relative size risks failing the job. Run "
            "inside a Slurm allocation, or set storage.memory_bytes explicitly."
        )

    fraction = dataset_bytes / memory_bytes
    budget = int(memory_bytes * safety_fraction)
    if dataset_bytes > budget:
        raise StagingRefused(
            f"Refusing to stage: the dataset is {dataset_bytes / 1024**3:.2f} GiB, "
            f"{fraction:.0%} of the {memory_bytes / 1024**3:.2f} GiB allocation, "
            f"above the {safety_fraction:.0%} safety margin.\n"
            "Compute-node /tmp is memory: staging this would leave too little for the "
            "workload itself. Either request more memory, stage a packaged form of the "
            "dataset, or read from shared storage (which is the right answer for a "
            "one-pass workload anyway)."
        )
    return {
        "memory_allocated_bytes": memory_bytes,
        "memory_source": memory_source,
        "dataset_fraction_of_memory": fraction,
        "safety_fraction": safety_fraction,
    }


@contextmanager
def staged_artifact(
    source: str | os.PathLike[str],
    tmp_dir: str | os.PathLike[str] | None = None,
    samples: Sequence[Sample] | None = None,
    validate: bool = True,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    memory_bytes: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Copy an artifact to node-local storage, then always remove it.

    Yields a dict with the staged path and the full cost of getting it there:
    ``staging_seconds``, ``validation_seconds``, ``staged_bytes``, ``staged_files``,
    plus the memory-safety numbers. The caller records those in its run summary, which
    is what stops a staged result from looking free.

    The copy is checked before it is used. A truncated or partial copy that gets
    measured anyway would produce a fast, wrong result.
    """
    source = Path(source)
    if not source.exists():
        raise StagingFailed(f"nothing to stage: {source} does not exist")

    dataset_bytes, dataset_files = artifact_bytes(source)
    if memory_bytes is None:
        memory_bytes, memory_source = detect_allocated_memory()
    else:
        memory_source = "explicit"
    safety = check_safety(dataset_bytes, memory_bytes, safety_fraction, memory_source)

    destination_root = resolve_tmp_dir(str(tmp_dir) if tmp_dir else "")
    destination = destination_root / source.name
    created_root = not destination_root.exists()
    destination_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "staging_source": str(source),
        "staging_destination": str(destination),
        "staged_bytes": dataset_bytes,
        "staged_files": dataset_files,
        **safety,
    }

    try:
        started = time.perf_counter()
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            # dirs_exist_ok=False: a leftover directory from a previous job would
            # otherwise be measured instead of a fresh copy.
            shutil.copytree(source, destination)
        result["staging_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        if validate:
            validate_staged(source, destination, samples)
        result["validation_seconds"] = time.perf_counter() - started

        staged_bytes, staged_files = artifact_bytes(destination)
        result["peak_tmp_bytes"] = staged_bytes
        result["staged_path"] = str(destination)
        if staged_files != dataset_files:
            raise StagingFailed(
                f"staged copy has {staged_files} files, source has {dataset_files}"
            )
        yield result
    finally:
        # Node-local data must not outlive the job: the next job on this node would
        # inherit it, and it is occupying memory until removed.
        _remove(destination)
        if created_root:
            _remove(destination_root)


def validate_staged(
    source: Path,
    destination: Path,
    samples: Sequence[Sample] | None = None,
) -> None:
    """Check that a staged copy matches its source.

    Compares file count and every file's size, which catches truncation and missing
    files — the realistic failure modes of an interrupted copy. Sizes rather than
    checksums by design: a checksum pass would read the whole dataset a second time,
    which on a large dataset costs as much as the staging it is verifying. When a
    manifest is supplied, the staged tree is also checked against it, so a copy that
    is internally consistent but missing manifest entries is still caught.
    """
    if destination.is_file():
        if not destination.exists():
            raise StagingFailed(f"staged file missing: {destination}")
        if destination.stat().st_size != source.stat().st_size:
            raise StagingFailed(
                f"staged file is {destination.stat().st_size} bytes, source is "
                f"{source.stat().st_size}: the copy is incomplete"
            )
        return

    for directory, _, names in os.walk(source):
        relative = Path(directory).relative_to(source)
        for name in names:
            original = Path(directory) / name
            copied = destination / relative / name
            if not copied.exists():
                raise StagingFailed(f"staged copy is missing {relative / name}")
            if copied.stat().st_size != original.stat().st_size:
                raise StagingFailed(
                    f"staged {relative / name} is {copied.stat().st_size} bytes, "
                    f"source is {original.stat().st_size}: the copy is incomplete"
                )

    if samples:
        missing = [
            sample.sample_id
            for sample in samples
            if not (destination / sample.relative_path).exists()
        ]
        if missing:
            raise StagingFailed(
                f"{len(missing)} manifest sample(s) are absent from the staged copy, "
                f"starting with {missing[0]}"
            )


def _remove(path: Path) -> None:
    """Remove a staged path, reporting but not raising on failure.

    A cleanup problem should not mask whatever the job was actually doing, but it must
    not pass silently either: leftover node-local data occupies memory.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        print(f"WARNING could not remove staged data at {path}: {exc}", flush=True)
