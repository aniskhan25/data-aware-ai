"""Staging to node-local storage, and the break-even arithmetic that judges it.

Two things here can do real damage if wrong: a safety check that lets a job stage more
data than it has memory for, and a comparison that recommends staging for a workload
that never recovers the copy. Both are pinned. Needs no PyTorch.
"""

from __future__ import annotations

import os

import pytest

from dataaware.staging import (
    DEFAULT_SAFETY_FRACTION,
    StagingFailed,
    StagingRefused,
    artifact_bytes,
    check_safety,
    resolve_tmp_dir,
    staged_artifact,
    validate_staged,
)
from dataaware.storage import (
    break_even,
    compare,
    format_report,
    placement_rows,
    setup_cost,
    total_cost,
)
from dataaware.schema import new_run_summary

GIB = 1024**3


def build_tree(root, files):
    for relative, size in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)


# --- safety ------------------------------------------------------------------


def test_a_small_dataset_is_safe_to_stage():
    result = check_safety(1 * GIB, 32 * GIB, 0.5, "SLURM_MEM_PER_NODE")
    assert result["dataset_fraction_of_memory"] == pytest.approx(1 / 32)
    assert result["memory_allocated_bytes"] == 32 * GIB


def test_a_dataset_over_the_margin_is_refused():
    with pytest.raises(StagingRefused, match="above the 50% safety margin"):
        check_safety(20 * GIB, 32 * GIB, 0.5)


def test_the_refusal_explains_that_tmp_is_memory():
    with pytest.raises(StagingRefused, match="/tmp is memory"):
        check_safety(20 * GIB, 32 * GIB, 0.5)


def test_unknown_memory_is_treated_as_unsafe():
    """Staging a dataset of unknown relative size is how a job dies mid-copy."""
    with pytest.raises(StagingRefused, match="could not be determined"):
        check_safety(1 * GIB, None, 0.5, "unknown (not running under Slurm)")


def test_non_positive_memory_is_treated_as_unsafe():
    with pytest.raises(StagingRefused, match="could not be determined"):
        check_safety(1 * GIB, 0, 0.5)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_an_invalid_safety_fraction_is_rejected(fraction):
    with pytest.raises(StagingRefused, match="safety_fraction"):
        check_safety(1 * GIB, 32 * GIB, fraction)


def test_exactly_at_the_margin_is_allowed():
    assert check_safety(16 * GIB, 32 * GIB, 0.5)["dataset_fraction_of_memory"] == 0.5


def test_the_default_margin_leaves_room_for_the_workload():
    assert DEFAULT_SAFETY_FRACTION <= 0.5


# --- destination -------------------------------------------------------------


def test_an_explicit_tmp_dir_wins(monkeypatch):
    monkeypatch.setenv("SLURM_TMPDIR", "/somewhere/else")
    assert str(resolve_tmp_dir("/explicit")) == "/explicit"


def test_slurm_tmpdir_is_preferred(monkeypatch):
    """Slurm already makes SLURM_TMPDIR per-job, so it is used as given."""
    monkeypatch.setenv("SLURM_TMPDIR", "/scratch-local/job")
    assert str(resolve_tmp_dir()) == "/scratch-local/job"


def test_tmpdir_still_gets_a_job_scoped_subdirectory(monkeypatch):
    """TMPDIR inside a container is usually plain /tmp.

    Using it directly would place every job's staged data at the same path, so two
    jobs on one node would overwrite each other and delete it on cleanup.
    """
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    assert str(resolve_tmp_dir()) == "/tmp/daai-999"


def test_the_fallback_path_is_job_scoped(monkeypatch):
    """Two jobs on one node must not share or delete each other's staged data."""
    for name in ("SLURM_TMPDIR", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert str(resolve_tmp_dir()) == "/tmp/daai-12345"


# --- staging -----------------------------------------------------------------


def test_staging_copies_validates_and_reports_cost(tmp_path):
    source = tmp_path / "shards"
    build_tree(source, {"shard-0.tar": 1000, "shard-1.tar": 1000})

    with staged_artifact(
        source, tmp_dir=tmp_path / "node-local", memory_bytes=1 << 30
    ) as staged:
        path = tmp_path / "node-local" / "shards"
        assert path.is_dir()
        assert (path / "shard-0.tar").read_bytes() == b"x" * 1000
        assert staged["staging_seconds"] >= 0.0
        assert staged["validation_seconds"] >= 0.0
        assert staged["staged_bytes"] == 2000
        assert staged["staged_files"] == 2
        assert staged["peak_tmp_bytes"] == 2000

    # Node-local data must not outlive the job.
    assert not path.exists()


def test_a_single_file_artifact_can_be_staged(tmp_path):
    """A SquashFS image is one file; staging it moves one object, not fifty thousand."""
    image = tmp_path / "dataset.squashfs"
    image.write_bytes(b"y" * 4096)

    with staged_artifact(image, tmp_dir=tmp_path / "local", memory_bytes=1 << 30) as staged:
        assert staged["staged_files"] == 1
        assert (tmp_path / "local" / "dataset.squashfs").read_bytes() == b"y" * 4096
    assert not (tmp_path / "local" / "dataset.squashfs").exists()


def test_staged_data_is_removed_even_when_the_run_fails(tmp_path):
    source = tmp_path / "shards"
    build_tree(source, {"a.tar": 100})

    with pytest.raises(RuntimeError, match="measurement blew up"):
        with staged_artifact(source, tmp_dir=tmp_path / "local", memory_bytes=1 << 30):
            raise RuntimeError("measurement blew up")

    assert not (tmp_path / "local" / "shards").exists()


def test_refusal_happens_before_anything_is_copied(tmp_path):
    source = tmp_path / "big"
    build_tree(source, {"a.bin": 10_000})
    with pytest.raises(StagingRefused):
        with staged_artifact(
            source, tmp_dir=tmp_path / "local", memory_bytes=1000, safety_fraction=0.5
        ):
            pass
    assert not (tmp_path / "local" / "big").exists()


def test_staging_a_missing_source_is_reported(tmp_path):
    with pytest.raises(StagingFailed, match="does not exist"):
        with staged_artifact(tmp_path / "absent", memory_bytes=1 << 30):
            pass


def test_validation_can_be_skipped(tmp_path):
    source = tmp_path / "shards"
    build_tree(source, {"a.tar": 100})
    with staged_artifact(
        source, tmp_dir=tmp_path / "local", validate=False, memory_bytes=1 << 30
    ) as staged:
        assert staged["validation_seconds"] >= 0.0


# --- validation --------------------------------------------------------------


def test_a_truncated_copy_is_detected(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    build_tree(source, {"a.bin": 1000})
    build_tree(destination, {"a.bin": 500})
    with pytest.raises(StagingFailed, match="the copy is incomplete"):
        validate_staged(source, destination)


def test_a_missing_file_is_detected(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    build_tree(source, {"a.bin": 100, "b.bin": 100})
    build_tree(destination, {"a.bin": 100})
    with pytest.raises(StagingFailed, match="missing"):
        validate_staged(source, destination)


def test_a_truncated_single_file_is_detected(tmp_path):
    source = tmp_path / "a.img"
    destination = tmp_path / "b.img"
    source.write_bytes(b"x" * 1000)
    destination.write_bytes(b"x" * 10)
    with pytest.raises(StagingFailed, match="the copy is incomplete"):
        validate_staged(source, destination)


def test_manifest_samples_absent_from_the_copy_are_detected(tmp_path):
    from dataaware.manifest import Sample

    source = tmp_path / "src"
    destination = tmp_path / "dst"
    build_tree(source, {"images/a.jpg": 100})
    build_tree(destination, {"images/a.jpg": 100})
    samples = [
        Sample("s0", "images/a.jpg", 0, 100, 1, 1, "0" * 16, 1),
        Sample("s1", "images/absent.jpg", 0, 100, 1, 1, "0" * 16, 1),
    ]
    with pytest.raises(StagingFailed, match="manifest sample"):
        validate_staged(source, destination, samples)


def test_artifact_bytes_counts_a_tree_and_a_file(tmp_path):
    build_tree(tmp_path / "tree", {"a": 10, "sub/b": 20})
    assert artifact_bytes(tmp_path / "tree") == (30, 2)
    (tmp_path / "solo").write_bytes(b"x" * 7)
    assert artifact_bytes(tmp_path / "solo") == (7, 1)


# --- break-even --------------------------------------------------------------


def placement(storage, throughput, epoch_seconds, staging=0.0, validation=0.0, **extra):
    values = {
        "run_name": f"storage-{storage}",
        "storage": storage,
        "manifest_hash": "abc123",
        "batch_size": 64,
        "num_workers": 13,
        "measured_batches": 1000,
        "seed": 1234,
        "compute_steps": 1,
        "samples_per_second": throughput,
        "estimated_epoch_seconds": epoch_seconds,
        "staging_seconds": staging,
        "validation_seconds": validation,
        "total_job_seconds": staging + validation + epoch_seconds,
    }
    values.update(extra)
    return new_run_summary(**values)


def test_setup_cost_includes_validation():
    """Validation is work the job does because it staged; excluding it flatters staging."""
    row = placement_rows([placement("tmp", 100, 10, staging=20, validation=5)])["tmp"]
    assert setup_cost(row) == 25


def test_break_even_epochs_from_the_documented_formula():
    rows = placement_rows(
        [placement("scratch", 100, 100.0), placement("tmp", 200, 50.0, staging=100.0)]
    )
    result = break_even(rows["scratch"], rows["tmp"])
    assert result["per_epoch_time_saved"] == 50.0
    assert result["setup_cost_seconds"] == 100.0
    assert result["break_even_epochs"] == 2.0


def test_a_placement_that_saves_nothing_never_breaks_even():
    """Reporting a large number would imply it eventually pays off. It does not."""
    rows = placement_rows(
        [placement("scratch", 100, 50.0), placement("tmp", 100, 50.0, staging=100.0)]
    )
    assert break_even(rows["scratch"], rows["tmp"])["break_even_epochs"] is None


def test_a_slower_placement_with_setup_cost_never_breaks_even():
    rows = placement_rows(
        [placement("scratch", 200, 50.0), placement("tmp", 100, 100.0, staging=30.0)]
    )
    result = break_even(rows["scratch"], rows["tmp"])
    assert result["per_epoch_time_saved"] < 0
    assert result["break_even_epochs"] is None


def test_a_free_faster_placement_breaks_even_immediately():
    """Flash costs no staging, so any per-epoch gain is immediate."""
    rows = placement_rows([placement("scratch", 100, 100.0), placement("flash", 200, 50.0)])
    assert break_even(rows["scratch"], rows["flash"])["break_even_epochs"] == 0.0


def test_total_cost_includes_setup():
    row = placement_rows([placement("tmp", 100, 10.0, staging=100.0)])["tmp"]
    assert total_cost(row, 1) == 110.0
    assert total_cost(row, 10) == 200.0


def test_the_cheapest_placement_depends_on_the_horizon():
    """A one-pass workload and a long campaign reach opposite conclusions."""
    report = compare(
        [
            placement("scratch", 100, 100.0),
            placement("tmp", 400, 25.0, staging=200.0),
        ],
        horizons=(1, 50),
    )
    assert report["cheapest_at_epochs"]["1"] == "scratch"
    assert report["cheapest_at_epochs"]["50"] == "tmp"


def test_never_recovering_is_reported_as_never():
    report = compare(
        [placement("scratch", 100, 50.0), placement("tmp", 100, 50.0, staging=100.0)]
    )
    assert "BREAK_EVEN_EPOCHS=never" in format_report(report)
    assert "never recovered" in format_report(report)


def test_baseline_falls_back_when_scratch_is_absent():
    report = compare([placement("flash", 100, 50.0), placement("tmp", 200, 25.0)])
    assert report["baseline"] in ("flash", "tmp")


def test_staged_memory_share_is_reported_as_a_caution():
    report = compare(
        [
            placement("scratch", 100, 100.0),
            placement("tmp", 200, 50.0, staging=10.0, dataset_fraction_of_memory=0.4),
        ]
    )
    assert any("40% of the job's memory" in c for c in report["cautions"])


def test_correctness_failures_are_cautioned():
    report = compare(
        [placement("scratch", 100, 100.0), placement("tmp", 200, 50.0, failed_samples=3)]
    )
    assert any("did not read its data" in c for c in report["cautions"])


def test_report_is_json_serialisable():
    import json

    report = compare([placement("scratch", 100, 100.0), placement("tmp", 200, 50.0)])
    assert json.loads(json.dumps(report)) == report


def test_no_summaries_is_an_error():
    with pytest.raises(ValueError, match="no run summaries"):
        compare([])


def test_report_states_the_headline_rule():
    report = compare([placement("scratch", 100, 100.0), placement("tmp", 200, 50.0)])
    assert "do not automatically mean a faster end-to-end job" in format_report(report)


def test_a_difference_within_noise_is_flagged():
    """Measured on LUMI: flash beat scratch by 4.4% with 3-6% run-to-run spread.

    Naming a cheapest placement on a margin thinner than its own noise is exactly the
    mistake the rest of the tutorial warns against.
    """
    report = compare(
        [
            placement("scratch", 13243, 3.78),
            placement("scratch", 13708, 3.65),
            placement("flash", 13648, 3.66),
            placement("flash", 14500, 3.45),
        ]
    )
    assert any("indistinguishable on" in c for c in report["cautions"])


def test_a_difference_well_beyond_noise_is_not_flagged():
    report = compare(
        [
            placement("scratch", 1000, 50.0),
            placement("scratch", 1010, 49.5),
            placement("flash", 5000, 10.0),
            placement("flash", 5050, 9.9),
        ]
    )
    assert not any("indistinguishable on" in c for c in report["cautions"])
