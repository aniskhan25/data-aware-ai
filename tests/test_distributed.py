"""Distributed correctness arithmetic.

Each of Part VI's three broken cases is reproduced here as rank reports, so the
detector is pinned against the exact symptom it exists to catch. No process group is
launched: the aggregation is a pure function precisely so that it can be tested this
way. Needs no PyTorch.
"""

from __future__ import annotations

import pytest

from dataaware.distributed import (
    RankReport,
    aggregate,
    diagnose,
    rank_and_world_size,
)


def rank(index, indices, throughput=1000.0, elapsed=10.0, waiting=0.3, observed=None):
    return RankReport(
        rank=index,
        samples_observed=len(indices) if observed is None else observed,
        unique_indices=list(indices),
        samples_per_second=throughput,
        elapsed_seconds=elapsed,
        data_wait_fraction=waiting,
    )


def healthy_reports(world_size=8, total=800):
    """Each rank reads a disjoint slice covering the dataset exactly once."""
    per_rank = total // world_size
    return [
        rank(index, range(index * per_rank, (index + 1) * per_rank))
        for index in range(world_size)
    ]


# --- the healthy case --------------------------------------------------------


def test_a_healthy_run_is_valid():
    result = aggregate(healthy_reports(), total_samples=800)
    assert result["duplicate_samples"] == 0
    assert result["missing_samples"] == 0
    assert result["unique_samples"] == 800
    assert result["coverage_fraction"] == 1.0
    assert result["idle_ranks"] == []
    assert result["partitioning_valid"] is True
    assert "HEALTHY" in diagnose(result)[0]


# --- Challenge C: too few shards --------------------------------------------


def test_idle_ranks_are_detected():
    reports = [
        rank(0, range(0, 400)),
        rank(1, range(400, 800)),
        *[rank(index, [], throughput=0.0, elapsed=0.0) for index in range(2, 8)],
    ]
    result = aggregate(reports, total_samples=800)

    assert result["idle_ranks"] == [2, 3, 4, 5, 6, 7]
    assert result["partitioning_valid"] is False
    assert result["min_rank_throughput"] == 0.0
    assert result["rank_throughput_spread"] == 1.0
    finding = diagnose(result)[0]
    assert "IDLE READERS" in finding
    assert "fewer shards than readers" in finding


# --- Challenge D: duplicate samples -----------------------------------------


def test_every_rank_reading_the_same_stream_is_detected():
    """The failure mode a throughput number cannot see."""
    reports = [rank(index, range(100)) for index in range(8)]
    result = aggregate(reports, total_samples=800)

    assert result["samples_measured"] == 800
    assert result["unique_samples"] == 100
    assert result["duplicate_samples"] == 700
    assert result["partitioning_valid"] is False
    # Aggregate throughput looks excellent while seven eighths of it is waste.
    assert result["total_samples_per_second"] == pytest.approx(8000.0)

    finding = next(f for f in diagnose(result) if "DUPLICATE READS" in f)
    assert "88%" in finding or "87%" in finding
    assert "redundant work" in finding


def test_partial_overlap_between_ranks_is_counted():
    reports = [rank(0, range(0, 100)), rank(1, range(50, 150))]
    result = aggregate(reports, total_samples=150)
    assert result["samples_measured"] == 200
    assert result["unique_samples"] == 150
    assert result["duplicate_samples"] == 50


# --- Challenge E: imbalanced shards -----------------------------------------


def test_imbalance_is_detected_even_when_partitioning_is_correct():
    """Correct assignment is necessary but not sufficient."""
    reports = [
        rank(0, range(0, 100), elapsed=10.0, throughput=10.0),
        rank(1, range(100, 200), elapsed=10.0, throughput=10.0),
        rank(2, range(200, 300), elapsed=40.0, throughput=2.5),
        rank(3, range(300, 400), elapsed=10.0, throughput=10.0),
    ]
    result = aggregate(reports, total_samples=400)

    assert result["duplicate_samples"] == 0
    assert result["missing_samples"] == 0
    assert result["partitioning_valid"] is True
    assert result["rank_elapsed_spread"] == pytest.approx(0.75)

    finding = next(f for f in diagnose(result) if "IMBALANCE" in f)
    assert "slowest sets the pace" in finding
    assert "Equal sample counts per shard do not mean equal work" in finding


# --- coverage semantics ------------------------------------------------------


def test_missing_samples_are_reported_when_the_dataset_was_traversed():
    reports = [rank(0, range(0, 400)), rank(1, range(400, 700))]
    result = aggregate(reports, total_samples=800, expect_full_coverage=True)
    assert result["missing_samples"] == 100
    assert any("MISSING SAMPLES" in f for f in diagnose(result))


def test_a_partial_pass_does_not_report_missing_samples():
    """A window that touched a third of the dataset says nothing about coverage."""
    reports = [rank(0, range(0, 100)), rank(1, range(100, 200))]
    result = aggregate(reports, total_samples=800)
    assert result["missing_samples"] == 0
    assert result["coverage_fraction"] == pytest.approx(0.25)
    assert "not evaluated" in result["notes"]


# --- rank resolution ---------------------------------------------------------


def test_torchrun_variables_are_preferred(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert rank_and_world_size() == (3, 8)


