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
    coverage_expectation,
    prepared_layout,
    run_loader_benchmark,
    synthetic_compute,
)
from dataaware.manifest import read_manifest  # noqa: E402
from dataaware.schema import validate_run_summary  # noqa: E402


# --- sample accounting -------------------------------------------------------


def accounting(dataset_size=10, batch_size=5, drop_last=True, streaming=False, num_workers=0):
    """Build accounting the way the benchmark does, from the coverage rules."""
    expected, allowance = coverage_expectation(
        total_samples=dataset_size,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        streaming=streaming,
    )
    return SampleAccounting(expected_coverage=expected, drop_allowance=allowance)


# --- coverage rules ----------------------------------------------------------


def test_map_style_coverage_lowers_the_expectation_precisely():
    """A map-style dataset drops a knowable remainder, so nothing is tolerated."""
    expected, allowance = coverage_expectation(10, 4, True, 2, streaming=False)
    assert (expected, allowance) == (8, 0)


def test_streaming_coverage_uses_a_per_worker_allowance():
    """Each streaming worker batches its own stream, so each may drop a remainder."""
    expected, allowance = coverage_expectation(100, 8, True, 4, streaming=True)
    assert expected == 100
    assert allowance == 7 * 4


def test_coverage_without_drop_last_expects_everything():
    assert coverage_expectation(10, 4, False, 4, streaming=True) == (10, 0)
    assert coverage_expectation(10, 4, False, 0, streaming=False) == (10, 0)


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


def test_a_single_rank_run_reports_rank_zero(base_config_dict):
    """world_size 1 is the degenerate distributed case and must still work."""
    summary = run_loader_benchmark(config_from_dict(base_config_dict))
    assert summary["world_size"] == 1
    assert summary["rank"] == 0


def test_observed_indices_can_be_collected(base_config_dict):
    """Cross-rank duplicate detection needs the identities, not just a count."""
    indices: list[int] = []
    summary = run_loader_benchmark(
        config_from_dict(base_config_dict), collect_indices=indices
    )
    assert len(indices) == summary["unique_samples"]
    assert len(set(indices)) == len(indices)


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


# --- layouts -----------------------------------------------------------------


def test_squashfs_prebound_reads_exactly_like_loose_files(base_config_dict):
    """A mounted image presents ordinary paths, so the reader must be identical.

    Pointing a squashfs run at a plain directory exercises that claim: the layout
    label changes, the data read does not.
    """
    loose = run_loader_benchmark(config_from_dict(base_config_dict))

    base_config_dict["dataset"]["layout"] = "squashfs"
    base_config_dict["dataset"]["squashfs_mode"] = "prebound"
    packaged = run_loader_benchmark(config_from_dict(base_config_dict))

    assert packaged["layout"] == "squashfs"
    assert packaged["bytes_read"] == loose["bytes_read"]
    assert packaged["samples_measured"] == loose["samples_measured"]
    assert packaged["unique_samples"] == loose["unique_samples"]
    # One object on the filesystem, whatever the tree inside contains.
    assert packaged["filesystem_objects"] == 1
    assert loose["filesystem_objects"] > 1


def test_squashfs_prebound_requires_a_readable_directory(base_config_dict, tmp_path):
    base_config_dict["dataset"]["layout"] = "squashfs"
    base_config_dict["dataset"]["root"] = str(tmp_path / "not-mounted")
    with pytest.raises(ValueError, match="not a directory"):
        run_loader_benchmark(config_from_dict(base_config_dict))


def _shard_config(base_config_dict, tiny_dataset, tmp_path, **loader):
    from dataaware.shards import ShardPlan, build_shards

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    shard_dir = tmp_path / "shards"
    build_shards(root, samples, shard_dir, ShardPlan(samples_per_shard=8, seed=7))

    base_config_dict["dataset"]["layout"] = "webdataset"
    base_config_dict["dataset"]["root"] = str(shard_dir)
    base_config_dict["loader"]["shuffle"] = False
    base_config_dict["loader"].update(loader)
    return config_from_dict(base_config_dict)


def test_streaming_layout_covers_the_dataset_without_duplicates(
    base_config_dict, tiny_dataset, tmp_path
):
    root, manifest = tiny_dataset
    total = len(read_manifest(manifest))
    config = _shard_config(
        base_config_dict, tiny_dataset, tmp_path, batch_size=4, num_workers=0
    )
    # Long enough to complete at least one epoch.
    summary = run_loader_benchmark(
        config_from_dict(
            {**config.resolved, "run": {**config.resolved["run"], "measured_batches": total}}
        )
    )
    assert summary["layout"] == "webdataset"
    assert summary["duplicate_samples"] == 0
    assert summary["missing_samples"] == 0
    assert summary["unique_samples"] == total


def test_streaming_workers_read_disjoint_shards(base_config_dict, tiny_dataset, tmp_path):
    """Two workers over four shards must not read the same sample twice."""
    root, manifest = tiny_dataset
    total = len(read_manifest(manifest))
    config = _shard_config(
        base_config_dict,
        tiny_dataset,
        tmp_path,
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
    )
    summary = run_loader_benchmark(
        config_from_dict(
            {**config.resolved, "run": {**config.resolved["run"], "measured_batches": total}}
        )
    )
    assert summary["duplicate_samples"] == 0
    assert summary["unique_samples"] == total


def test_streaming_reports_shard_metrics(base_config_dict, tiny_dataset, tmp_path):
    config = _shard_config(
        base_config_dict, tiny_dataset, tmp_path, batch_size=4, num_workers=0
    )
    summary = run_loader_benchmark(config)
    assert summary["num_shards"] == 4
    assert summary["shard_opens"] >= 1
    assert summary["shard_open_seconds"] >= 0.0
    # Shards plus their index, against one file per sample for a loose tree.
    assert summary["filesystem_objects"] == 5
    # Opens track shards, not samples: that is the difference the layout makes.
    assert summary["files_opened"] == summary["shard_opens"]
    assert summary["files_opened"] < summary["samples_measured"]


def test_all_layouts_return_identical_sample_bytes(tiny_dataset, tmp_path):
    """The release criterion: the core layouts must read equivalent samples."""
    from dataaware.manifest import checksum_bytes
    from dataaware.shards import ShardPlan, build_shards

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    expected = {sample.sample_id: sample.checksum for sample in samples}

    loose = LooseFileDataset(root, samples)
    from_loose = {
        samples[index].sample_id: checksum_bytes(
            (root / samples[index].relative_path).read_bytes()
        )
        for index in range(len(loose))
    }

    build_shards(root, samples, tmp_path / "shards", ShardPlan(samples_per_shard=8))
    from dataaware.shards import iter_shard_samples, read_shard_index

    index = read_shard_index(tmp_path / "shards" / "shard_index.json")
    from_shards = {}
    for record in index["shards"]:
        for sample_id, payload, _ in iter_shard_samples(
            tmp_path / "shards" / record["shard"]
        ):
            from_shards[sample_id] = checksum_bytes(payload)

    assert from_loose == expected
    assert from_shards == expected


def test_streaming_rejects_index_shuffling(base_config_dict, tiny_dataset, tmp_path):
    """Accepting shuffle: true would promise ordering the layout cannot deliver."""
    from dataaware.config import ConfigError

    base_config_dict["dataset"]["layout"] = "webdataset"
    base_config_dict["loader"]["shuffle"] = True
    with pytest.raises(ConfigError, match="loader.shuffle must be false"):
        config_from_dict(base_config_dict)


def test_shuffle_buffer_does_not_lose_or_duplicate_samples(
    base_config_dict, tiny_dataset, tmp_path
):
    root, manifest = tiny_dataset
    total = len(read_manifest(manifest))
    config = _shard_config(
        base_config_dict,
        tiny_dataset,
        tmp_path,
        batch_size=4,
        num_workers=0,
        shuffle_buffer=16,
    )
    summary = run_loader_benchmark(
        config_from_dict(
            {**config.resolved, "run": {**config.resolved["run"], "measured_batches": total}}
        )
    )
    assert summary["duplicate_samples"] == 0
    assert summary["unique_samples"] == total


def test_prepared_layout_reports_object_counts(base_config_dict):
    config = config_from_dict(base_config_dict)
    with prepared_layout(config) as (resolved, layout_metrics):
        assert resolved.dataset.root == config.dataset.root
        assert layout_metrics["filesystem_objects"] > 0


# --- epoch-mode measurement --------------------------------------------------


def test_epoch_mode_reads_each_sample_exactly_once(base_config_dict, tiny_dataset):
    """Coverage validation depends on this being exact.

    An epoch boundary is only visible once a batch of the next pass arrives; counting
    that batch would register duplicate reads in a run whose partitioning is correct.
    """
    total = len(read_manifest(tiny_dataset[1]))
    base_config_dict["run"]["measured_epochs"] = 1
    base_config_dict["run"]["warmup_batches"] = 0
    base_config_dict["loader"]["batch_size"] = 8
    base_config_dict["loader"]["drop_last"] = False
    summary = run_loader_benchmark(config_from_dict(base_config_dict))

    assert summary["samples_measured"] == total
    assert summary["unique_samples"] == total
    assert summary["duplicate_samples"] == 0
    assert summary["missing_samples"] == 0
    assert summary["measured_epochs"] == 1


def test_epoch_mode_can_measure_several_passes(base_config_dict, tiny_dataset):
    total = len(read_manifest(tiny_dataset[1]))
    base_config_dict["run"]["measured_epochs"] = 2
    base_config_dict["run"]["warmup_batches"] = 0
    base_config_dict["loader"]["batch_size"] = 8
    base_config_dict["loader"]["drop_last"] = False
    summary = run_loader_benchmark(config_from_dict(base_config_dict))

    assert summary["measured_epochs"] == 2
    assert summary["samples_measured"] == total * 2
    # Visiting every sample once per epoch is correct, not duplication.
    assert summary["duplicate_samples"] == 0


def test_batch_mode_remains_the_default(base_config_dict):
    summary = run_loader_benchmark(config_from_dict(base_config_dict))
    assert summary["batches_measured"] == base_config_dict["run"]["measured_batches"]
