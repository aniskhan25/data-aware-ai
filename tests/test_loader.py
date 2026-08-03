"""The loose-file loader and its sample accounting.

Skipped when PyTorch is absent, so the rest of the suite still runs in a minimal
environment such as a login node.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the loader benchmark requires PyTorch")

from dataaware.config import config_from_dict  # noqa: E402
from dataaware.loaders import (  # noqa: E402
    LooseFileDataset,
    SampleAccounting,
    build_dataset,
    run_loader_benchmark,
    synthetic_compute,
)
from dataaware.manifest import read_manifest  # noqa: E402
from dataaware.schema import validate_run_summary  # noqa: E402


# --- sample accounting -------------------------------------------------------


def accounting(dataset_size=10, batch_size=5, drop_last=True) -> SampleAccounting:
    return SampleAccounting(dataset_size, batch_size, drop_last)


def test_a_correct_epoch_reports_no_duplicates_or_missing():
    book = accounting()
    book.observe(range(10))
    book.end_epoch(complete=True)
    assert book.duplicate_samples == 0
    assert book.missing_samples == 0
    assert book.unique_samples == 10


def test_repeats_within_an_epoch_are_duplicates():
    book = accounting()
    book.observe([0, 1, 2, 2, 3])
    book.end_epoch(complete=True)
    assert book.duplicate_samples == 1


def test_every_rank_reading_the_same_stream_shows_up_as_duplicates():
    """The Part VI failure mode, in miniature."""
    book = accounting(dataset_size=4, batch_size=2)
    for _ in range(8):  # eight readers, same four samples
        book.observe([0, 1, 2, 3])
    book.end_epoch(complete=True)
    assert book.total_observed == 32
    assert book.unique_samples == 4
    assert book.duplicate_samples == 28


def test_uncovered_samples_in_a_complete_epoch_are_missing():
    book = accounting(dataset_size=10, batch_size=5, drop_last=False)
    book.observe(range(7))
    book.end_epoch(complete=True)
    assert book.missing_samples == 3


def test_drop_last_remainder_is_not_reported_as_missing():
    book = accounting(dataset_size=10, batch_size=4, drop_last=True)
    book.observe(range(8))  # two batches of four; two samples correctly dropped
    book.end_epoch(complete=True)
    assert book.missing_samples == 0


def test_a_partial_epoch_never_reports_missing_samples():
    book = accounting()
    book.observe(range(3))
    book.end_epoch(complete=False)
    assert book.missing_samples == 0
    assert book.partial_epoch is True


def test_duplicates_are_counted_per_epoch_not_across_epochs():
    book = accounting()
    for _ in range(2):
        book.observe(range(10))
        book.end_epoch(complete=True)
    # Visiting every sample once per epoch is correct, not duplication.
    assert book.duplicate_samples == 0
    assert book.total_observed == 20
    assert book.unique_samples == 10


# --- dataset -----------------------------------------------------------------


def test_dataset_returns_decoded_samples(tiny_dataset):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    dataset = LooseFileDataset(root, samples, verify_checksums=True)

    assert len(dataset) == len(samples)
    item = dataset[0]
    assert item["image"].shape == (3, samples[0].height, samples[0].width)
    assert item["image"].dtype == torch.uint8
    assert item["byte_size"] == samples[0].byte_size
    assert item["class_id"] == samples[0].class_id
    assert item["failed"] == 0


def test_a_missing_file_is_counted_not_raised(tiny_dataset, tmp_path):
    """One unreadable sample must not abort a long measurement."""
    _, manifest = tiny_dataset
    samples = read_manifest(manifest)
    dataset = LooseFileDataset(tmp_path / "does-not-exist", samples)
    item = dataset[0]
    assert item["failed"] == 1
    assert item["byte_size"] == 0


def test_checksum_mismatch_is_detected(tiny_dataset, tmp_path):
    _, manifest = tiny_dataset
    samples = read_manifest(manifest)
    corrupted = tmp_path / "corrupted"
    (corrupted / samples[0].relative_path).parent.mkdir(parents=True)
    (corrupted / samples[0].relative_path).write_bytes(b"not an image")

    dataset = LooseFileDataset(corrupted, samples[:1], verify_checksums=True)
    assert dataset[0]["failed"] == 1


def test_unimplemented_layouts_fail_with_a_pointer(base_config_dict):
    base_config_dict["dataset"]["layout"] = "webdataset"
    config = config_from_dict(base_config_dict)
    with pytest.raises(NotImplementedError, match="Part III"):
        build_dataset(config, [])


# --- benchmark ---------------------------------------------------------------


def test_benchmark_produces_a_valid_summary(base_config_dict):
    config = config_from_dict(base_config_dict)
    summary = validate_run_summary(run_loader_benchmark(config))

    expected_samples = config.run.measured_batches * config.loader.batch_size
    assert summary["samples_measured"] == expected_samples
    assert summary["batches_measured"] == config.run.measured_batches
    assert summary["failed_samples"] == 0
    assert summary["duplicate_samples"] == 0
    assert summary["samples_per_second"] > 0
    assert summary["bytes_read"] > 0
    assert summary["layout"] == "loose-files"
    assert summary["manifest_hash"]
    assert summary["config_hash"] == config.config_hash()
    assert 0.0 <= summary["mean_data_wait_fraction"] <= 1.0


def test_benchmark_is_reproducible_for_a_fixed_seed(base_config_dict):
    """Two runs of one configuration must read the same samples in the order."""
    config = config_from_dict(base_config_dict)
    first = run_loader_benchmark(config)
    second = run_loader_benchmark(config)
    # Timings differ; the data read must not.
    assert first["bytes_read"] == second["bytes_read"]
    assert first["samples_measured"] == second["samples_measured"]
    assert first["unique_samples"] == second["unique_samples"]


def test_a_different_seed_changes_the_sample_order(base_config_dict):
    base_config_dict["loader"]["batch_size"] = 4
    base_config_dict["run"]["measured_batches"] = 2
    first = run_loader_benchmark(config_from_dict(base_config_dict))
    base_config_dict["run"]["seed"] = 99
    second = run_loader_benchmark(config_from_dict(base_config_dict))
    assert first["bytes_read"] != second["bytes_read"]


def test_measuring_past_the_end_of_the_dataset_cycles_epochs(base_config_dict):
    """A short dataset must not cut a measurement short."""
    base_config_dict["loader"]["batch_size"] = 8
    base_config_dict["run"]["measured_batches"] = 12  # 32 samples / 8 = 4 per epoch
    summary = run_loader_benchmark(config_from_dict(base_config_dict))
    assert summary["batches_measured"] == 12
    assert summary["samples_measured"] == 96
    # Repeats across epochs are expected and must not be flagged as duplicates.
    assert summary["duplicate_samples"] == 0
    assert summary["missing_samples"] == 0
    assert "complete epoch" in summary["notes"]


def test_batch_size_larger_than_the_dataset_is_rejected(base_config_dict):
    base_config_dict["loader"]["batch_size"] = 4096
    with pytest.raises(ValueError, match="fewer than loader.batch_size"):
        run_loader_benchmark(config_from_dict(base_config_dict))


def test_distributed_runs_are_not_supported_yet(base_config_dict):
    base_config_dict["distributed"]["enabled"] = True
    with pytest.raises(NotImplementedError, match="Part VI"):
        run_loader_benchmark(config_from_dict(base_config_dict))


def test_worker_processes_produce_the_same_data_as_no_workers(base_config_dict):
    base_config_dict["loader"]["num_workers"] = 0
    serial = run_loader_benchmark(config_from_dict(base_config_dict))
    base_config_dict["loader"]["num_workers"] = 2
    base_config_dict["loader"]["persistent_workers"] = True
    parallel = run_loader_benchmark(config_from_dict(base_config_dict))
    assert serial["bytes_read"] == parallel["bytes_read"]
    assert serial["samples_measured"] == parallel["samples_measured"]


def test_synthetic_compute_is_shape_stable():
    batch = torch.randint(0, 255, (4, 3, 8, 8), dtype=torch.uint8)
    assert synthetic_compute(batch, steps=2).shape == (4, 3)
    # Zero steps still reduces the batch, so the metric denominator exists.
    assert synthetic_compute(batch, steps=0).shape == (4, 3)


def test_synthetic_compute_does_not_modify_the_batch():
    batch = torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8)
    before = batch.clone()
    synthetic_compute(batch, steps=1)
    assert torch.equal(batch, before)
