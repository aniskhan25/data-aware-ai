"""Dataset inspection.

The inspector runs before anything expensive, often on a tree someone else built,
so it has to survive whatever it finds: empty directories, unreadable
subdirectories, symlink cycles, and stray files among the samples.
"""

from __future__ import annotations


import pytest

from dataaware.inspection import (
    DEFAULT_MIN_SHARDS as MIN_SHARDS,
    MANY_FILES_TRIGGER,
    detect_allocated_memory,
    inspect_path,
)


def build_tree(root, files: dict[str, int]) -> None:
    """Create ``{relative_path: byte_size}`` under ``root``."""
    for relative, size in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)


def uniform_tree(root, count: int, size: int = 1024, per_dir: int = 100) -> None:
    files = {
        f"images/class_{index // per_dir:04d}/s{index:08d}.jpg": size
        for index in range(count)
    }
    build_tree(root, files)


# --- basic counting ----------------------------------------------------------


def test_counts_files_bytes_and_directories(tmp_path):
    build_tree(tmp_path, {"a/one.jpg": 10, "a/two.jpg": 20, "b/three.png": 30})
    report = inspect_path(tmp_path)

    tree = report["tree"]
    assert tree["total_files"] == 3
    assert tree["total_bytes"] == 60
    # The root counts as a directory, alongside a/ and b/.
    assert tree["directories"] == 3
    assert tree["filesystem_objects"] == 6
    assert report["directories"]["max_depth"] == 1
    assert report["directories"]["max_files_in_one_directory"] == 2


def test_p95_to_median_ratio_ignores_a_single_outlier(tmp_path):
    """One stray large file must not be read as a variable-size dataset."""
    files = {f"images/s{i:04d}.jpg": 1000 for i in range(200)}
    files["manifest.jsonl"] = 5_000_000
    build_tree(tmp_path, files)

    sizes = inspect_path(tmp_path)["file_sizes"]
    assert sizes["p95_to_median_ratio"] < 1.5
    # The coefficient of variation is dominated by that one file, which is exactly
    # why it does not drive the shard-balancing suggestion.
    assert sizes["coefficient_of_variation"] > 1.0
    assert "shard-balancing" not in _experiments(inspect_path(tmp_path))


def test_small_file_threshold_is_configurable(tmp_path):
    build_tree(tmp_path, {"a.bin": 100, "b.bin": 5000, "c.bin": 100_000})
    assert inspect_path(tmp_path, small_file_bytes=1000)["small_files"]["files"] == 1
    assert inspect_path(tmp_path, small_file_bytes=10_000)["small_files"]["files"] == 2

    fraction = inspect_path(tmp_path, small_file_bytes=10_000)["small_files"]["fraction"]
    assert fraction == pytest.approx(2 / 3)


# --- awkward trees -----------------------------------------------------------


def test_symlinks_are_counted_but_not_followed(tmp_path):
    build_tree(tmp_path, {"real/a.jpg": 10})
    (tmp_path / "link.jpg").symlink_to(tmp_path / "real/a.jpg")
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 1
    assert report["tree"]["symlinks"] == 1


# --- determinism -------------------------------------------------------------


def test_report_is_deterministic_apart_from_provenance(tmp_path):
    uniform_tree(tmp_path, 50)
    first = inspect_path(tmp_path)
    second = inspect_path(tmp_path)
    del first["provenance"], second["provenance"]
    assert first == second


# --- packaging arithmetic ----------------------------------------------------


def test_packaging_collapses_objects_to_one(tmp_path):
    uniform_tree(tmp_path, 30)
    packaging = inspect_path(tmp_path)["packaging"]
    assert packaging["filesystem_objects_now"] > 30
    assert packaging["filesystem_objects_as_squashfs"] == 1


def test_already_compressed_data_is_recognised(tmp_path):
    build_tree(tmp_path, {f"a{i}.jpg": 1000 for i in range(10)})
    packaging = inspect_path(tmp_path)["packaging"]
    assert packaging["already_compressed_byte_fraction"] == 1.0
    assert packaging["compression_likely_to_help"] is False


def test_uncompressed_data_suggests_compression_may_help(tmp_path):
    build_tree(tmp_path, {f"a{i}.npy": 1000 for i in range(10)})
    packaging = inspect_path(tmp_path)["packaging"]
    assert packaging["already_compressed_byte_fraction"] == 0.0
    assert packaging["compression_likely_to_help"] is True


def test_shard_suggestion_follows_the_target_size(tmp_path):
    build_tree(tmp_path, {f"a{i}.jpg": 1000 for i in range(100)})
    packaging = inspect_path(tmp_path, target_shard_bytes=10_000)["packaging"]
    assert packaging["suggested_shards"] == 10
    assert packaging["suggested_samples_per_shard"] == 10


def test_shard_suggestion_never_drops_below_the_useful_floor(tmp_path):
    """A size-based target alone would suggest one shard, which feeds one reader.

    Observed on LUMI: a 143 MiB dataset against the 512 MiB default target produced
    "about 1 shards", which is exactly the too-few-shards failure mode Part VI exists
    to expose.
    """
    build_tree(tmp_path, {f"a{i}.jpg": 1000 for i in range(100)})
    packaging = inspect_path(tmp_path, target_shard_bytes=512 * 1024 * 1024)["packaging"]
    assert packaging["size_based_shards"] == 1
    assert packaging["suggested_shards"] == MIN_SHARDS
    assert packaging["suggested_samples_per_shard"] == 100 // MIN_SHARDS


# --- memory and staging advice ----------------------------------------------


def test_staging_is_suggested_when_the_dataset_fits(tmp_path):
    build_tree(tmp_path, {f"a{i}.jpg": 1000 for i in range(10)})
    report = inspect_path(tmp_path, memory_bytes=1_000_000)
    memory = report["memory"]
    assert memory["source"] == "explicit"
    assert memory["tmp_staging_within_safety_margin"] is True
    assert "tmp-staging" in _experiments(report)


def test_staging_is_discouraged_when_the_dataset_is_too_large(tmp_path):
    build_tree(tmp_path, {f"a{i}.jpg": 1000 for i in range(10)})
    report = inspect_path(tmp_path, memory_bytes=12_000)
    assert report["memory"]["tmp_staging_within_safety_margin"] is False
    assert "avoid-tmp-staging" in _experiments(report)


def test_memory_detected_from_slurm_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    memory, source = detect_allocated_memory()
    assert memory == 2048 * 1024 * 1024
    assert source == "SLURM_MEM_PER_NODE"


# --- candidate suggestions ---------------------------------------------------


def test_many_small_files_suggest_packaging_and_sharding(tmp_path):
    uniform_tree(tmp_path, MANY_FILES_TRIGGER, size=2048, per_dir=500)
    experiments = _experiments(inspect_path(tmp_path))
    assert "squashfs" in experiments
    assert "webdataset" in experiments


def test_limitations_are_always_reported(tmp_path):
    """A reader must never mistake a suggestion for a finding."""
    uniform_tree(tmp_path, 10)
    limitations = inspect_path(tmp_path)["limitations"]
    assert len(limitations) >= 5
    assert any("hypothesis" in text for text in limitations)


# --- validation and rendering ------------------------------------------------


def _experiments(report: dict) -> list[str]:
    return [candidate["experiment"] for candidate in report["candidates"]]
