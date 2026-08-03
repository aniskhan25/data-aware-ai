"""Comparing runs, and refusing to compare incomparable ones.

The refusal is the point. A comparison tool that always produces a tidy table is
how a configuration mistake becomes a published conclusion. Needs no PyTorch.
"""

from __future__ import annotations

import pytest

from dataaware.compare import (
    aggregate,
    blocking,
    compare,
    find_mismatches,
    format_table,
    group_summaries,
)
from dataaware.schema import new_run_summary


def summary(layout="loose-files", **overrides):
    values = {
        "run_name": f"{layout}-run",
        "layout": layout,
        "manifest_hash": "abc123",
        "batch_size": 64,
        "num_workers": 4,
        "measured_batches": 200,
        "seed": 1234,
        "compute_steps": 1,
        "shuffle": True,
        "samples_per_second": 1000.0,
        "mib_per_second": 100.0,
        "mean_batch_wait_seconds": 0.01,
        "p95_batch_wait_seconds": 0.02,
        "mean_data_wait_fraction": 0.5,
        "startup_seconds": 1.0,
        "files_opened": 12800,
        "filesystem_objects": 50000,
    }
    values.update(overrides)
    return new_run_summary(**values)


# --- compatibility -----------------------------------------------------------


def test_identical_configurations_have_no_mismatches():
    assert find_mismatches([summary(), summary(layout="squashfs")]) == []


def test_a_different_manifest_blocks_the_comparison():
    """Different data means no table of numbers is meaningful."""
    mismatches = find_mismatches([summary(), summary(manifest_hash="different")])
    assert [m.field for m in mismatches] == ["manifest_hash"]
    assert blocking(mismatches)


def test_a_different_schema_version_blocks_the_comparison():
    first = summary()
    second = summary()
    second["schema_version"] = "9.9"
    assert blocking(find_mismatches([first, second]))


@pytest.mark.parametrize(
    "field", ["batch_size", "num_workers", "measured_batches", "seed", "compute_steps"]
)
def test_uncontrolled_differences_are_reported_but_not_blocking(field):
    mismatches = find_mismatches([summary(), summary(**{field: 999})])
    assert [m.field for m in mismatches] == [field]
    assert not blocking(mismatches)


def test_shuffle_differing_across_layouts_is_not_flagged_as_uncontrolled():
    """A streaming layout cannot shuffle an index; demanding it match would make
    every layout comparison in the tutorial look broken."""
    report = compare([summary(), summary(layout="webdataset", shuffle=False)])
    assert report["controlled"] is True
    assert any("expected" in note or "property of the layouts" in note for note in report["notes"])


def test_shuffle_differing_within_one_layout_is_flagged():
    report = compare([summary(), summary(shuffle=False)])
    assert any("shuffle differs between runs of the same group" in c for c in report["cautions"])


def test_a_single_summary_has_nothing_to_mismatch():
    assert find_mismatches([summary()]) == []


# --- aggregation -------------------------------------------------------------


def test_aggregate_reports_median_and_spread():
    result = aggregate([10.0, 20.0, 30.0])
    assert result["runs"] == 3
    assert result["median"] == 20.0
    assert result["min"] == 10.0
    assert result["max"] == 30.0
    assert result["cv"] > 0.0


def test_aggregate_uses_the_median_not_the_mean():
    """One slow run on a shared filesystem must not move the headline number."""
    assert aggregate([10.0, 10.0, 10.0, 1000.0])["median"] == 10.0


def test_aggregate_handles_no_values():
    assert aggregate([])["runs"] == 0


def test_grouping_preserves_first_seen_order():
    groups = group_summaries(
        [summary("webdataset"), summary("loose-files"), summary("webdataset")], "layout"
    )
    assert list(groups) == ["webdataset", "loose-files"]
    assert len(groups["webdataset"]) == 2


# --- the report --------------------------------------------------------------


def test_baseline_defaults_to_the_first_group():
    report = compare([summary("loose-files"), summary("squashfs")])
    assert report["baseline"] == "loose-files"


def test_baseline_can_be_chosen():
    report = compare([summary("loose-files"), summary("squashfs")], baseline="squashfs")
    assert report["baseline"] == "squashfs"
    assert "loose-files" in report["changes"]


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="not among the compared groups"):
        compare([summary()], baseline="nonexistent")


def test_changes_are_percentages_against_the_baseline():
    report = compare(
        [
            summary("loose-files", samples_per_second=1000.0),
            summary("squashfs", samples_per_second=1500.0),
        ]
    )
    assert report["changes"]["squashfs"]["samples_per_second"]["percent"] == pytest.approx(50.0)


def test_a_zero_baseline_yields_no_percentage_rather_than_infinity():
    report = compare(
        [summary("a", samples_per_second=0.0), summary("b", samples_per_second=10.0)]
    )
    assert report["changes"]["b"]["samples_per_second"]["percent"] is None


def test_object_reduction_appears_in_the_table():
    text = format_table(
        compare(
            [
                summary("loose-files", filesystem_objects=50000),
                summary("squashfs", filesystem_objects=1),
            ]
        )
    )
    assert "FILESYSTEM_OBJECT_REDUCTION=5e+04x" in text


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no run summaries"):
        compare([])


# --- cautions ----------------------------------------------------------------


def test_failed_samples_invalidate_a_group():
    report = compare([summary(), summary("squashfs", failed_samples=3)])
    assert any("did not read its data" in caution for caution in report["cautions"])


def test_duplicate_samples_are_called_out_as_inflated_throughput():
    report = compare([summary(), summary("squashfs", duplicate_samples=100)])
    assert any("redundant work" in caution for caution in report["cautions"])


def test_missing_samples_are_called_out_as_incomplete_coverage():
    report = compare([summary(), summary("squashfs", missing_samples=7)])
    assert any("not covering the dataset" in caution for caution in report["cautions"])


def test_single_runs_are_flagged_as_unknown_variance():
    report = compare([summary(), summary("squashfs")])
    assert any("Single run per group" in caution for caution in report["cautions"])


def test_noisy_repeats_are_flagged_instead():
    report = compare(
        [
            summary("loose-files", samples_per_second=1000.0),
            summary("loose-files", samples_per_second=3000.0),
            summary("squashfs", samples_per_second=1000.0),
            summary("squashfs", samples_per_second=1010.0),
        ]
    )
    cautions = " ".join(report["cautions"])
    assert "varied by more than 10" in cautions
    assert "loose-files" in cautions
    assert "Single run per group" not in cautions


def test_correctness_counters_are_summed_per_group():
    report = compare(
        [
            summary("loose-files"),
            summary("squashfs", duplicate_samples=2),
            summary("squashfs", duplicate_samples=3),
        ]
    )
    assert report["groups"]["squashfs"]["correctness"]["duplicate_samples"] == 5


def test_table_marks_the_baseline():
    text = format_table(compare([summary("loose-files"), summary("squashfs")]))
    assert "loose-files *" in text
    assert "baseline: loose-files" in text


def test_report_is_json_serialisable():
    import json

    report = compare([summary(), summary("squashfs")])
    assert json.loads(json.dumps(report)) == report
