"""The data-readiness decision.

The dangerous output here is a cheerful READY on evidence nobody gathered, or on a
pipeline that reads the wrong data quickly. Both are pinned below, along with the rule
that a missing input is never treated as a passing one. Needs no PyTorch.
"""

from __future__ import annotations



from dataaware.decision import decide, render_markdown


def layout_comparison(best="webdataset", failures=None, controlled=True, runs=3):
    groups = {
        "loose-files": {
            "runs": runs,
            "samples_per_second": {"median": 405.0},
            "correctness": {"failed_samples": 0, "duplicate_samples": 0, "missing_samples": 0},
        },
        best: {
            "runs": runs,
            "samples_per_second": {"median": 6926.0},
            "correctness": failures
            or {"failed_samples": 0, "duplicate_samples": 0, "missing_samples": 0},
        },
    }
    return {
        "baseline": "loose-files",
        "groups": groups,
        "changes": {best: {"samples_per_second": {"percent": 1610.0}}},
        "controlled": controlled,
        "comparable": True,
        "cautions": [],
        "notes": [],
    }


def worker_analysis(pattern="plateau", factor="storage-or-synchronisation", recommended=13):
    return {
        "rows": [],
        "pattern": pattern,
        "recommended_workers": recommended,
        "best_workers": recommended,
        "best_affordable_workers": recommended,
        "limiting_factor": factor,
        "explanation": "Throughput plateaued at 13 workers with CPU utilisation at 5%.",
        "cautions": [],
    }


def storage_comparison(staging_break_even=75.0, tmp_setup=4.76, cautions=None):
    placements = {
        "scratch": {
            "runs": 2,
            "estimated_epoch_seconds": 3.712,
            "staging_seconds": 0.0,
            "validation_seconds": 0.0,
            "samples_per_second": 13480.0,
        },
        "tmp": {
            "runs": 2,
            "estimated_epoch_seconds": 3.648,
            "staging_seconds": tmp_setup,
            "validation_seconds": 0.0,
            "samples_per_second": 13720.0,
        },
    }
    return {
        "baseline": "scratch",
        "placements": placements,
        "comparisons": {"tmp": {"break_even_epochs": staging_break_even}},
        "cautions": cautions or [],
        "horizons": [1, 3, 10, 50],
        "cheapest_at_epochs": {},
    }


def distributed_verdict(duplicates=0, missing=0, idle=None, spread=0.11, wait=0.3):
    return {
        "world_size": 8,
        "total_samples": 50000,
        "samples_measured": 50000,
        "unique_samples": 50000,
        "duplicate_samples": duplicates,
        "missing_samples": missing,
        "idle_ranks": idle or [],
        "rank_elapsed_spread": spread,
        "max_data_wait_fraction": wait,
        "coverage_fraction": 1.0,
    }


def complete(**overrides):
    inputs = {
        "inspection": {
            "tree": {"total_files": 50002, "total_gib": 0.134},
            "file_sizes": {"median_bytes": 2673.0},
            "small_files": {"fraction": 1.0, "threshold_bytes": 65536},
        },
        "layouts": layout_comparison(),
        "workers": worker_analysis(),
        "storage": storage_comparison(),
        "distributed": distributed_verdict(),
        "planned_epochs": 3,
    }
    inputs.update(overrides)
    return inputs


# --- readiness states --------------------------------------------------------


def test_missing_required_inputs_yield_inconclusive():
    """An absent measurement is not a passing one."""
    report = decide(**complete(distributed=None))
    assert report["data_readiness"] == "INCONCLUSIVE"
    assert "distributed" in report["inputs_missing"]
    assert report["next_experiment"] == "complete-the-missing-experiments"


def test_duplicate_reads_make_it_not_ready():
    report = decide(**complete(distributed=distributed_verdict(duplicates=350000)))
    assert report["data_readiness"] == "NOT_READY"
    assert any("not partitioning" in issue for issue in report["blocking_issues"])
    assert report["next_experiment"] == "fix-blocking-issues-then-revalidate"


def test_idle_ranks_make_it_not_ready():
    report = decide(**complete(distributed=distributed_verdict(idle=[2, 3, 4, 5, 6, 7])))
    assert report["data_readiness"] == "NOT_READY"
    assert any("received no data" in issue for issue in report["blocking_issues"])


def test_a_layout_that_failed_to_read_is_not_ready():
    failures = {"failed_samples": 12, "duplicate_samples": 0, "missing_samples": 0}
    report = decide(**complete(layouts=layout_comparison(failures=failures)))
    assert report["data_readiness"] == "NOT_READY"
    assert any("does not read its data correctly" in i for i in report["blocking_issues"])


def test_correctness_failures_outrank_missing_inputs():
    """NOT_READY is the more actionable finding, so it wins over INCONCLUSIVE."""
    report = decide(
        layouts=layout_comparison(),
        distributed=distributed_verdict(duplicates=100),
    )
    assert report["data_readiness"] == "NOT_READY"


def test_rank_imbalance_is_a_caution_not_a_blocker():
    """Correct partitioning that wastes capacity is usable but flawed."""
    report = decide(**complete(distributed=distributed_verdict(spread=0.34)))
    assert report["data_readiness"] == "READY_WITH_CAUTION"
    assert report["distributed_partitioning"] == "valid"
    assert any("differ by 34%" in caution for caution in report["cautions"])


# --- no single number decides readiness -------------------------------------


def test_throughput_alone_does_not_make_a_run_ready():
    """The Phase 7 rule: readiness is correctness and completeness, not speed."""
    fast_but_wrong = decide(
        layouts=layout_comparison(),
        workers=worker_analysis(),
        storage=storage_comparison(),
        distributed=distributed_verdict(duplicates=350000),
    )
    assert fast_but_wrong["data_readiness"] == "NOT_READY"

    slow_but_correct = decide(
        **complete(
            layouts={
                **layout_comparison(),
                "groups": {
                    "loose-files": {
                        "runs": 3,
                        "samples_per_second": {"median": 12.0},
                        "correctness": {
                            "failed_samples": 0,
                            "duplicate_samples": 0,
                            "missing_samples": 0,
                        },
                    }
                },
            }
        )
    )
    assert slow_but_correct["data_readiness"] in ("READY", "READY_WITH_CAUTION")


# --- staging depends on the planned horizon ---------------------------------


def test_staging_is_rejected_when_break_even_exceeds_the_plan():
    report = decide(**complete(planned_epochs=3))
    assert report["node_local_staging"] == "not-recommended"
    assert "75 epochs" in report["staging_evidence"]
    assert "only 3 are planned" in report["staging_evidence"]


def test_staging_that_saves_nothing_is_never_recommended():
    report = decide(
        **complete(storage=storage_comparison(staging_break_even=None))
    )
    assert report["node_local_staging"] == "not-recommended"
    assert "never recovered" in report["staging_evidence"]


def test_the_same_measurements_give_opposite_staging_advice():
    """Nothing in the data says how many epochs you intend to run."""
    short = decide(**complete(planned_epochs=1))
    long_run = decide(**complete(planned_epochs=500))
    assert short["node_local_staging"] == "not-recommended"
    assert long_run["node_local_staging"] in ("recommended", "viable")


# --- evidence and limitations -----------------------------------------------


def test_limitations_are_always_stated():
    report = decide(**complete())
    joined = " ".join(report["limitations"])
    assert "not a benchmark of LUMI storage" in joined
    assert "cannot account for requirements that were never measured" in joined
    assert "says nothing about whether the model" in joined


# --- rendering ---------------------------------------------------------------


def test_markdown_leads_with_blocking_issues_when_not_ready():
    markdown = render_markdown(
        decide(**complete(distributed=distributed_verdict(duplicates=100)))
    )
    assert "NOT_READY" in markdown
    assert "## Blocking issues" in markdown
    assert "multiplies the waste" in markdown


# --- near-ties favour the documented default --------------------------------


def test_a_noise_level_advantage_does_not_displace_scratch():
    """Measured on LUMI: flash beat scratch by 4% with 5% run-to-run spread.

    Recommending flash on that margin would contradict the caution printed alongside it,
    and flash is a smaller, scarcer resource than scratch.
    """
    storage = storage_comparison()
    storage["placements"]["flash"] = {
        "runs": 2,
        "estimated_epoch_seconds": 3.556,
        "staging_seconds": 0.0,
        "validation_seconds": 0.0,
        "samples_per_second": 14070.0,
        "samples_per_second_cv": 0.043,
    }
    storage["placements"]["scratch"]["samples_per_second_cv"] = 0.024
    report = decide(**complete(storage=storage))

    assert report["recommended_storage"] == "scratch"
    assert "documented default for job I/O" in report["storage_evidence"]


def test_a_real_advantage_does_displace_scratch():
    storage = storage_comparison()
    storage["placements"]["flash"] = {
        "runs": 2,
        "estimated_epoch_seconds": 0.5,
        "staging_seconds": 0.0,
        "validation_seconds": 0.0,
        "samples_per_second": 100000.0,
        "samples_per_second_cv": 0.01,
    }
    storage["placements"]["scratch"]["samples_per_second_cv"] = 0.01
    report = decide(**complete(storage=storage))
    assert report["recommended_storage"] == "flash"


def test_without_repeats_the_cheaper_placement_stands():
    """No measured variability means nothing can be declared a tie."""
    storage = storage_comparison()
    storage["placements"]["flash"] = {
        "runs": 1,
        "estimated_epoch_seconds": 3.0,
        "staging_seconds": 0.0,
        "validation_seconds": 0.0,
        "samples_per_second": 15000.0,
    }
    report = decide(**complete(storage=storage))
    assert report["recommended_storage"] == "flash"
