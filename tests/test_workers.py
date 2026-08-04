"""Reading a worker ladder.

The analysis has to distinguish four shapes — still improving, plateau, regression,
flat — and name a limiting resource. Getting that wrong sends a user to tune the wrong
thing, so each shape is pinned here. Needs no PyTorch.
"""

from __future__ import annotations

import pytest

from dataaware.schema import new_run_summary
from dataaware.workers import (
    CPU_SATURATED,
    PLATEAU_GAIN,
    analyse,
    format_table,
    ladder_rows,
)


def rung(workers, throughput, cpu=0.4, waiting=0.3, memory=1 << 30, switches=100.0,
         cpus=14, **extra):
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
        "child_processes": workers + 1,
        "oversubscription_ratio": (workers + 1) / cpus,
    }
    values.update(extra)
    return new_run_summary(**values)


# --- rows --------------------------------------------------------------------


def test_rows_are_ordered_by_worker_count():
    rows = ladder_rows([rung(7, 900), rung(0, 100), rung(2, 500)])
    assert [row["num_workers"] for row in rows] == [0, 2, 7]


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


def test_flat_ladder_points_elsewhere():
    analysis = analyse([rung(0, 500), rung(2, 505), rung(7, 498)])
    assert analysis["pattern"] == "flat"
    assert analysis["limiting_factor"] == "not-worker-bound"
    assert "Part III" in analysis["explanation"]


def test_a_single_rung_is_inconclusive():
    analysis = analyse([rung(4, 900)])
    assert analysis["pattern"] == "inconclusive"
    assert "at least two rungs" in analysis["explanation"]


def test_no_summaries_is_an_error():
    with pytest.raises(ValueError, match="no run summaries"):
        analyse([])


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


def test_a_healthy_plateau_is_called_balanced():
    analysis = analyse(
        [
            rung(0, 100, cpu=0.1, waiting=0.9),
            rung(2, 900, cpu=0.5, waiting=0.2),
            rung(7, 910, cpu=0.6, waiting=0.2),
        ]
    )
    assert analysis["limiting_factor"] == "balanced"
    assert "keeping up" in analysis["explanation"]


# --- cautions ----------------------------------------------------------------


def test_correctness_failures_come_before_tuning():
    analysis = analyse([rung(0, 100), rung(2, 900, failed_samples=5)])
    assert any("Fix correctness before tuning" in c for c in analysis["cautions"])


def test_single_runs_per_rung_are_flagged():
    analysis = analyse([rung(0, 100), rung(2, 900)])
    assert any("measured once" in c for c in analysis["cautions"])


def test_noisy_rungs_are_flagged():
    analysis = analyse(
        [rung(0, 100), rung(0, 100), rung(2, 500), rung(2, 2000)]
    )
    assert any("varied by more than 10" in c for c in analysis["cautions"])


def test_oversubscribed_rungs_are_flagged_as_not_adoptable():
    analysis = analyse([rung(0, 100), rung(2, 900), rung(28, 300)])
    assert any("more processes than the 14 allocated" in c for c in analysis["cautions"])


def test_a_worker_count_that_did_not_materialise_is_flagged():
    """The configured count is not evidence that many processes actually ran."""
    analysis = analyse([rung(0, 100), rung(7, 900, child_processes=2)])
    assert any("child process(es)" in c for c in analysis["cautions"])


def test_missing_process_count_is_not_flagged():
    """-1 means the platform could not tell, which is not a fault."""
    analysis = analyse([rung(0, 100), rung(7, 900, child_processes=-1)])
    assert not any("child process(es)" in c for c in analysis["cautions"])


# --- rendering ---------------------------------------------------------------


def test_table_marks_the_recommendation_and_reports_keys():
    text = format_table(analyse([rung(0, 100), rung(2, 900), rung(7, 920)]))
    assert "2 *" in text
    assert "LADDER_PATTERN=plateau" in text
    assert "RECOMMENDED_WORKERS=2" in text
    assert "BEST_WORKERS=7" in text
    assert "MAIN_LIMITING_FACTOR=" in text


def test_analysis_is_json_serialisable():
    import json

    analysis = analyse([rung(0, 100), rung(2, 900)])
    assert json.loads(json.dumps(analysis)) == analysis


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
    analysis = analyse([rung(20, 500, cpus=4), rung(40, 900, cpus=4)])
    assert analysis["recommended_workers"] in (20, 40)


def test_runaway_context_switches_are_reported():
    analysis = analyse(
        [rung(2, 900, switches=5.0), rung(7, 950, switches=15.0), rung(28, 960, switches=3000.0)]
    )
    assert "involuntary context switches rise" in analysis["explanation"]
    assert "costs the rest of the node" in analysis["explanation"]


def test_modest_context_switch_growth_is_not_reported():
    analysis = analyse([rung(2, 900, switches=10.0), rung(7, 950, switches=20.0)])
    assert "involuntary context switches rise" not in analysis["explanation"]
