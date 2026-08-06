"""Reading a worker ladder.

The analysis has to distinguish four shapes - still improving, plateau, regression,
flat - and name a limiting resource. Getting that wrong sends a user to tune the wrong
thing, so each shape is pinned here. Needs no PyTorch.
"""

from __future__ import annotations


from dataaware.schema import new_run_summary
from dataaware.workers import (
    CPU_SATURATED,
    PLATEAU_GAIN,
    analyse,
    ladder_rows,
)


def rung(workers, throughput, cpu=0.4, waiting=0.3, memory=1 << 30, switches=100.0,
         cpus=14, cores=7, **extra):
    values = {
        "run_name": f"workers-{workers}",
        "num_workers": workers,
        "manifest_hash": "abc123",
        "batch_size": 64,
        "measured_batches": 200,
        "seed": 1234,
        "compute_steps": 1,
        "samples_per_second": throughput,
        "cpu_utilization": cpu,
        "mean_data_wait_fraction": waiting,
        "peak_memory_bytes": memory,
        "involuntary_switches_per_second": switches,
        "cpus_available": cpus,
        "allocated_cores": cores,
        "threads_per_core": (cpus / cores) if cores else 0.0,
        "processes_per_physical_core": ((workers + 1) / cores) if cores else 0.0,
        "child_processes": workers + 1,
        "oversubscription_ratio": (workers + 1) / cpus,
    }
    values.update(extra)
    return new_run_summary(**values)


# --- rows --------------------------------------------------------------------


def test_repeats_are_aggregated_on_the_median():
    rows = ladder_rows([rung(2, 100), rung(2, 200), rung(2, 5000)])
    assert len(rows) == 1
    assert rows[0]["runs"] == 3
    assert rows[0]["samples_per_second"] == 200
    assert rows[0]["samples_per_second_cv"] > 0


# --- ladder shapes -----------------------------------------------------------


def test_still_improving_is_not_treated_as_a_recommendation():
    """A ladder that never turned over has not found its limit."""
    analysis = analyse([rung(0, 100), rung(2, 400), rung(7, 900)])
    assert analysis["pattern"] == "still-improving"
    assert analysis["best_workers"] == 7
    assert analysis["limiting_factor"] == "not-yet-saturated"
    assert "Add a higher rung" in analysis["explanation"]


def test_plateau_recommends_the_cheapest_adequate_rung():
    """Workers cost memory, so the knee is the answer, not the peak."""
    analysis = analyse([rung(0, 100), rung(2, 900), rung(7, 920)])
    assert analysis["pattern"] == "plateau"
    assert analysis["best_workers"] == 7
    assert analysis["recommended_workers"] == 2
    assert "fewer processes" in analysis["explanation"]


def test_regression_is_detected_and_named():
    analysis = analyse([rung(0, 100), rung(2, 900), rung(7, 950), rung(28, 400)])
    assert analysis["pattern"] == "regression"
    assert analysis["limiting_factor"] == "oversubscribed"
    assert "not a free knob" in analysis["explanation"]


# --- limiting factor ---------------------------------------------------------


def test_saturated_cpu_is_named_as_the_limit():
    analysis = analyse(
        [rung(0, 100, cpu=0.2), rung(2, 900, cpu=CPU_SATURATED + 0.05), rung(7, 910, cpu=0.95)]
    )
    assert analysis["limiting_factor"] == "cpu-decode"
    assert "CPU-bound" in analysis["explanation"]


def test_idle_cpu_with_high_waits_points_at_storage():
    analysis = analyse(
        [
            rung(0, 100, cpu=0.1, waiting=0.99),
            rung(2, 900, cpu=0.2, waiting=0.9),
            rung(7, 905, cpu=0.2, waiting=0.9),
        ]
    )
    assert analysis["limiting_factor"] == "storage-or-synchronisation"
    assert "Idle CPUs with high waits" in analysis["explanation"]


# --- cautions ----------------------------------------------------------------


def test_correctness_failures_come_before_tuning():
    analysis = analyse([rung(0, 100), rung(2, 900, failed_samples=5)])
    assert any("Fix correctness before tuning" in c for c in analysis["cautions"])


def test_oversubscribed_rungs_are_flagged_as_not_adoptable():
    analysis = analyse([rung(0, 100), rung(2, 900), rung(28, 300)])
    assert any("more processes than the 14 logical CPUs" in c for c in analysis["cautions"])


# --- rendering ---------------------------------------------------------------


def test_plateau_threshold_is_the_documented_one():
    """A gain just under the threshold must not justify more workers."""
    just_under = 900 * (1.0 + PLATEAU_GAIN * 0.5)
    analysis = analyse([rung(0, 100), rung(2, 900), rung(7, just_under)])
    assert analysis["recommended_workers"] == 2


# --- the allocation is a hard ceiling on advice ------------------------------


def test_an_oversubscribed_rung_is_never_recommended():
    """Measured on LUMI: 28 workers on 14 logical CPUs was the fastest rung.

    Recommending it would contradict the caution printed about that same rung, and it
    borrows CPU from whatever else shares the node.
    """
    analysis = analyse([rung(0, 100), rung(2, 900), rung(7, 1200), rung(28, 2000)])
    assert analysis["best_workers"] == 28
    assert analysis["best_affordable_workers"] == 7
    assert analysis["recommended_workers"] == 7
    assert "not recommended" in analysis["explanation"]
    assert "fastest count that fits" in analysis["explanation"]


def test_all_rungs_oversubscribed_still_yields_a_recommendation():
    """With nothing inside the allocation there is still a least-bad answer."""
    analysis = analyse([rung(20, 500, cpus=4, cores=2), rung(40, 900, cpus=4, cores=2)])
    assert analysis["recommended_workers"] in (20, 40)


# --- physical cores are not logical threads ----------------------------------


def test_smt_saturation_is_reported_separately_from_oversubscription():
    """13 workers + parent fits 14 logical CPUs but puts 2 processes on each of 7 cores.

    Calling that "fits the allocation" without qualification is what made the first
    version of this tutorial's Part IV misleading.
    """
    analysis = analyse([rung(2, 900), rung(6, 12000), rung(13, 13760)])
    joined = " ".join(analysis["cautions"])
    assert "SMT-saturated" in joined
    assert "physical cores" in joined
    # A rung at one process per physical core must not be flagged.
    assert "[6]" not in joined


def test_one_process_per_physical_core_is_not_flagged():
    analysis = analyse([rung(2, 900), rung(6, 12000)])
    assert not any("SMT-saturated" in caution for caution in analysis["cautions"])


