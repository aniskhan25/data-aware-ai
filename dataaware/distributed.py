"""Validating that many readers share a dataset correctly.

The question this answers is not "how fast" but "is each rank reading the right
samples". A distributed pipeline can report excellent aggregate throughput while every
rank reads the same data — eight times the work, none of it useful. Throughput is
meaningless until assignment is known to be correct.

Process-group handling lives here, but the arithmetic that decides correctness is a
pure function (:func:`aggregate`) so it can be tested without launching ranks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import metrics


@dataclass
class RankReport:
    """What one rank contributes to the correctness check.

    ``unique_indices`` is the set of manifest positions this rank read during the
    measured window. Sending the indices rather than a count is what makes cross-rank
    duplicate detection possible: two ranks reporting 100 samples each could be
    covering 200 samples or the same 100 twice, and only the identities distinguish
    those cases.
    """

    rank: int
    samples_observed: int
    unique_indices: list[int] = field(default_factory=list)
    samples_per_second: float = 0.0
    elapsed_seconds: float = 0.0
    data_wait_fraction: float = 0.0
    duplicate_samples_within_rank: int = 0
    batches_measured: int = 0
    shard_opens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, without the index list.

        The indices can be tens of thousands of integers per rank; they are needed for
        the aggregate check but would dominate a stored summary.
        """
        return {
            "rank": self.rank,
            "samples_observed": self.samples_observed,
            "unique_samples": len(set(self.unique_indices)),
            "samples_per_second": self.samples_per_second,
            "elapsed_seconds": self.elapsed_seconds,
            "data_wait_fraction": self.data_wait_fraction,
            "duplicate_samples_within_rank": self.duplicate_samples_within_rank,
            "batches_measured": self.batches_measured,
            "shard_opens": self.shard_opens,
        }


def aggregate(
    reports: Sequence[RankReport],
    total_samples: int,
    expect_full_coverage: bool | None = None,
) -> dict[str, Any]:
    """Combine per-rank reports into a correctness and balance verdict.

    ``expect_full_coverage`` says whether the ranks between them read enough samples to
    have covered the dataset once. When they did not, ``missing_samples`` is left at
    zero and the note says so: a window that only touched a third of the dataset says
    nothing about coverage, and reporting the remainder as missing would be wrong.
    """
    if not reports:
        raise ValueError("no rank reports to aggregate")

    ordered = sorted(reports, key=lambda report: report.rank)
    observed = sum(report.samples_observed for report in ordered)
    union: set[int] = set()
    for report in ordered:
        union.update(report.unique_indices)
    unique = len(union)

    if expect_full_coverage is None:
        expect_full_coverage = observed >= total_samples

    throughputs = [report.samples_per_second for report in ordered]
    elapsed = [report.elapsed_seconds for report in ordered]
    waits = [report.data_wait_fraction for report in ordered]

    idle = [report.rank for report in ordered if report.samples_observed == 0]

    result = {
        "world_size": len(ordered),
        "total_samples": total_samples,
        "samples_measured": observed,
        "unique_samples": unique,
        # Every read beyond the first of a given sample, across all ranks. This is the
        # number that exposes ranks sharing a stream.
        "duplicate_samples": observed - unique,
        "missing_samples": max(0, total_samples - unique) if expect_full_coverage else 0,
        "coverage_fraction": unique / total_samples if total_samples else 0.0,
        "total_samples_per_second": sum(throughputs),
        "min_rank_throughput": min(throughputs),
        "max_rank_throughput": max(throughputs),
        "rank_throughput_spread": metrics.spread(throughputs),
        "min_rank_elapsed_seconds": min(elapsed),
        "max_rank_elapsed_seconds": max(elapsed),
        "rank_elapsed_spread": metrics.spread(elapsed),
        "mean_data_wait_fraction": metrics.mean(waits),
        "max_data_wait_fraction": max(waits),
        "idle_ranks": idle,
        "rank_summaries": [report.to_dict() for report in ordered],
    }
    result["partitioning_valid"] = (
        result["duplicate_samples"] == 0 and result["missing_samples"] == 0 and not idle
    )
    result["notes"] = _note(result, expect_full_coverage)
    return result


def _note(result: dict[str, Any], expect_full_coverage: bool) -> str:
    parts = []
    if not expect_full_coverage:
        parts.append(
            f"The ranks read {result['samples_measured']} samples between them, fewer "
            f"than the {result['total_samples']} in the dataset, so missing_samples is "
            "not evaluated. Duplicates are still counted over the measured window."
        )
    else:
        parts.append(
            f"The ranks read {result['samples_measured']} samples between them, enough "
            "to have covered the dataset once, so coverage is evaluated."
        )
    if result["idle_ranks"]:
        parts.append(
            f"Rank(s) {result['idle_ranks']} read nothing at all. Their capacity was "
            "wasted, and aggregate throughput is being produced by fewer readers than "
            "were allocated."
        )
    return " ".join(parts)


def diagnose(result: dict[str, Any]) -> list[str]:
    """Turn a verdict into findings a reader can act on, worst first."""
    findings = []

    if result["idle_ranks"]:
        findings.append(
            f"IDLE READERS: rank(s) {result['idle_ranks']} received no data. There are "
            "fewer shards than readers, so some readers have nothing assigned. Rebuild "
            "with at least as many shards as ranks x workers."
        )
    if result["duplicate_samples"]:
        share = result["duplicate_samples"] / max(1, result["samples_measured"])
        findings.append(
            f"DUPLICATE READS: {result['duplicate_samples']} of "
            f"{result['samples_measured']} sample reads ({share:.0%}) were repeats. "
            f"Only {result['unique_samples']} distinct samples were touched. The ranks "
            "are not partitioning the dataset: aggregate throughput here is measuring "
            "redundant work, not progress."
        )
    if result["missing_samples"]:
        findings.append(
            f"MISSING SAMPLES: {result['missing_samples']} sample(s) were never read "
            f"although the ranks read enough to cover the dataset "
            f"({result['coverage_fraction']:.1%} covered). Some samples are assigned to "
            "nobody."
        )
    if result["rank_elapsed_spread"] > 0.2:
        findings.append(
            f"IMBALANCE: rank elapsed times differ by "
            f"{result['rank_elapsed_spread']:.0%} "
            f"({result['min_rank_elapsed_seconds']:.2f}s to "
            f"{result['max_rank_elapsed_seconds']:.2f}s). With synchronised ranks the "
            "slowest sets the pace, so the fastest ranks are idling. Equal sample "
            "counts per shard do not mean equal work per shard."
        )
    if not findings:
        findings.append(
            f"HEALTHY: {result['world_size']} ranks read {result['unique_samples']} "
            "distinct samples with no duplicates, no missing samples, and no idle "
            f"readers. Rank throughput spread {result['rank_throughput_spread']:.1%}."
        )
    return findings


# --- process group -----------------------------------------------------------


def rank_and_world_size() -> tuple[int, int]:
    """Resolve this process's rank and the world size.

    Reads torchrun's variables first, then Slurm's. Supporting both means the same
    script works under ``srun`` directly, which is how LUMI jobs usually launch, and
    under ``torchrun`` for local testing.
    """
    for rank_name, size_name in (
        ("RANK", "WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
    ):
        rank, size = os.environ.get(rank_name), os.environ.get(size_name)
        if rank is not None and size is not None:
            return int(rank), int(size)
    return 0, 1


def local_rank() -> int:
    for name in ("LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def init_process_group(backend: str = "gloo") -> tuple[int, int]:
    """Initialise the process group and return ``(rank, world_size)``.

    ``gloo`` by default, deliberately. This benchmark validates the *data path*: it
    does no collective computation on tensors and touches no GPU, so a CPU backend is
    both sufficient and less to go wrong. Only the correctness gather uses the group.
    """
    import torch.distributed as dist

    rank, world_size = rank_and_world_size()
    if world_size == 1:
        return rank, world_size

    os.environ.setdefault("MASTER_ADDR", _master_address())
    os.environ.setdefault("MASTER_PORT", os.environ.get("DAAI_MASTER_PORT", "29500"))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def _master_address() -> str:
    """First host of the allocation, which every rank agrees on."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    import subprocess

    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    # Fall back to a crude parse of the first name in the list.
    return nodelist.split(",")[0].split("[")[0]


def gather_reports(report: RankReport, world_size: int) -> list[RankReport]:
    """Collect every rank's report onto every rank.

    All ranks receive the full set so that any of them could produce the verdict;
    only rank 0 writes it. Ranks that finished early wait here, which is also what a
    real synchronised training step would do — so an idle or slow rank is visible
    rather than silently tolerated.
    """
    if world_size == 1:
        return [report]

    import torch.distributed as dist

    bucket: list[Any] = [None] * world_size
    dist.all_gather_object(bucket, report)
    return [item for item in bucket if isinstance(item, RankReport)]


def shutdown() -> None:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask a real failure
        print(f"WARNING error shutting down the process group: {exc}", flush=True)
