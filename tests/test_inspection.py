"""Dataset inspection.

The inspector runs before anything expensive, often on a tree someone else built,
so it has to survive whatever it finds: empty directories, unreadable
subdirectories, symlink cycles, and stray files among the samples.
"""

from __future__ import annotations

import os

import pytest

from dataaware.inspection import (
    DEFAULT_MIN_SHARDS as MIN_SHARDS,
    MANY_FILES_TRIGGER,
    InspectionError,
    detect_allocated_memory,
    format_keyvalue,
    inspect_path,
    walk_tree,
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


def test_size_statistics(tmp_path):
    build_tree(tmp_path, {f"f{i}.bin": i * 100 for i in range(1, 11)})
    sizes = inspect_path(tmp_path)["file_sizes"]
    assert sizes["min_bytes"] == 100
    assert sizes["max_bytes"] == 1000
    assert sizes["median_bytes"] == 550.0
    assert sizes["mean_bytes"] == 550.0


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


def test_genuinely_variable_sizes_are_detected(tmp_path):
    files = {f"s{i:04d}.jpg": 1000 for i in range(100)}
    files.update({f"big{i:04d}.jpg": 100_000 for i in range(20)})
    build_tree(tmp_path, files)
    report = inspect_path(tmp_path)
    assert report["file_sizes"]["p95_to_median_ratio"] >= 4.0
    assert "shard-balancing" in _experiments(report)


def test_small_file_threshold_is_configurable(tmp_path):
    build_tree(tmp_path, {"a.bin": 100, "b.bin": 5000, "c.bin": 100_000})
    assert inspect_path(tmp_path, small_file_bytes=1000)["small_files"]["files"] == 1
    assert inspect_path(tmp_path, small_file_bytes=10_000)["small_files"]["files"] == 2

    fraction = inspect_path(tmp_path, small_file_bytes=10_000)["small_files"]["fraction"]
    assert fraction == pytest.approx(2 / 3)


def test_size_histogram_thresholds(tmp_path):
    build_tree(tmp_path, {"a.bin": 100, "b.bin": 5000})
    rows = inspect_path(tmp_path, thresholds=[1000, 10_000])["size_thresholds"]
    assert [row["bytes"] for row in rows] == [1000, 10_000]
    assert [row["files"] for row in rows] == [1, 2]


def test_extension_distribution_is_ordered_and_stable(tmp_path):
    build_tree(
        tmp_path,
        {"a.jpg": 1, "b.jpg": 1, "c.JPG": 1, "d.png": 1, "noext": 1},
    )
    rows = inspect_path(tmp_path)["extensions"]
    # Extensions are lower-cased, so .JPG and .jpg are one group of three.
    assert rows[0]["extension"] == ".jpg"
    assert rows[0]["files"] == 3
    assert {row["extension"] for row in rows} == {".jpg", ".png", "(none)"}


def test_many_extensions_collapse_into_an_other_bucket(tmp_path):
    build_tree(tmp_path, {f"f{i}.e{i:03d}": 1 for i in range(30)})
    rows = inspect_path(tmp_path)["extensions"]
    assert rows[-1]["extension"] == "(other)"
    assert sum(row["files"] for row in rows) == 30


# --- awkward trees -----------------------------------------------------------


def test_empty_directory_is_safe(tmp_path):
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 0
    assert report["file_sizes"]["median_bytes"] == 0.0
    assert report["small_files"]["fraction"] == 0.0
    assert _experiments(report) == ["none"]
    # Must still render without dividing by zero anywhere.
    assert "TOTAL_FILES=0" in format_keyvalue(report)


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(InspectionError, match="does not exist"):
        inspect_path(tmp_path / "absent")


def test_a_single_file_can_be_inspected(tmp_path):
    target = tmp_path / "dataset.tar"
    target.write_bytes(b"x" * 4096)
    report = inspect_path(target)
    assert report["tree"]["total_files"] == 1
    assert report["tree"]["total_bytes"] == 4096
    assert report["extensions"][0]["extension"] == ".tar"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permissions",
)
def test_unreadable_directory_is_counted_not_fatal(tmp_path):
    build_tree(tmp_path, {"open/a.jpg": 10, "closed/b.jpg": 10})
    locked = tmp_path / "closed"
    locked.chmod(0o000)
    try:
        report = inspect_path(tmp_path)
    finally:
        locked.chmod(0o755)

    # The readable half is still measured.
    assert report["tree"]["total_files"] == 1
    assert report["tree"]["unreadable_directories"] == 1
    assert "fix-permissions-first" in _experiments(report)


def test_symlinks_are_counted_but_not_followed(tmp_path):
    build_tree(tmp_path, {"real/a.jpg": 10})
    (tmp_path / "link.jpg").symlink_to(tmp_path / "real/a.jpg")
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 1
    assert report["tree"]["symlinks"] == 1


def test_a_symlink_cycle_does_not_trap_the_walk(tmp_path):
    """A tree that links to itself must terminate, not exhaust the stack."""
    build_tree(tmp_path, {"sub/a.jpg": 10})
    (tmp_path / "sub/loop").symlink_to(tmp_path)
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 1
    assert report["tree"]["symlinks"] == 1


def test_hardlinks_are_reported(tmp_path):
    original = tmp_path / "a.bin"
    original.write_bytes(b"x" * 100)
    try:
        os.link(original, tmp_path / "b.bin")
    except OSError:
        pytest.skip("hard links unsupported here")
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 2
    assert report["tree"]["hardlinked_files"] == 2


def test_deep_trees_are_walked_without_recursion_limits(tmp_path):
    deep = "/".join(f"d{i}" for i in range(60))
    build_tree(tmp_path, {f"{deep}/leaf.jpg": 10})
    report = inspect_path(tmp_path)
    assert report["tree"]["total_files"] == 1
    assert report["directories"]["max_depth"] == 60


# --- determinism -------------------------------------------------------------


def test_report_is_deterministic_apart_from_provenance(tmp_path):
    uniform_tree(tmp_path, 50)
    first = inspect_path(tmp_path)
    second = inspect_path(tmp_path)
    del first["provenance"], second["provenance"]
    assert first == second


def test_provenance_is_the_only_nondeterministic_block(tmp_path):
    uniform_tree(tmp_path, 10)
    report = inspect_path(tmp_path)
    assert set(report["provenance"]) == {
        "generated_utc",
        "hostname",
        "walk_seconds",
        "slurm_job_id",
    }


def test_walk_result_does_not_depend_on_traversal_order(tmp_path):
    """Directory iteration order is not guaranteed; the totals must not vary."""
    uniform_tree(tmp_path, 40, per_dir=7)
    first = walk_tree(tmp_path)
    second = walk_tree(tmp_path)
    assert sorted(first.file_sizes) == sorted(second.file_sizes)
    assert first.total_bytes == second.total_bytes
    assert first.directories == second.directories
    assert first.max_files_in_one_directory == second.max_files_in_one_directory


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


def test_shard_suggestion_never_exceeds_the_sample_count(tmp_path):
    build_tree(tmp_path, {"only.jpg": 10, "two.jpg": 10})
    packaging = inspect_path(tmp_path)["packaging"]
    assert packaging["suggested_shards"] == 2


def test_shard_floor_is_explained_in_the_suggestion(tmp_path):
    uniform_tree(tmp_path, MANY_FILES_TRIGGER, size=2048, per_dir=500)
    reasons = {c["experiment"]: c["reason"] for c in inspect_path(tmp_path)["candidates"]}
    assert "too few to feed one node" in reasons["webdataset"]
    assert "at least as numerous as the readers" in reasons["webdataset"]


def test_format_hints_come_from_extensions(tmp_path):
    build_tree(tmp_path, {"a.parquet": 10, "b.h5": 10, "c.jpg": 10})
    hints = {hint["extension"] for hint in inspect_path(tmp_path)["format_hints"]}
    assert hints == {".parquet", ".h5"}


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


def test_no_staging_advice_when_memory_is_unknown(tmp_path, monkeypatch):
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    build_tree(tmp_path, {"a.jpg": 10})
    report = inspect_path(tmp_path)

    assert report["memory"]["allocated_bytes"] is None
    assert report["memory"]["tmp_staging_within_safety_margin"] is None
    assert "--memory-bytes" in report["memory"]["note"]
    experiments = _experiments(report)
    assert "tmp-staging" not in experiments
    assert "avoid-tmp-staging" not in experiments


def test_safety_fraction_is_respected(tmp_path):
    build_tree(tmp_path, {"a.jpg": 600})
    within = inspect_path(tmp_path, memory_bytes=1000, tmp_safety_fraction=0.9)
    outside = inspect_path(tmp_path, memory_bytes=1000, tmp_safety_fraction=0.5)
    assert within["memory"]["tmp_staging_within_safety_margin"] is True
    assert outside["memory"]["tmp_staging_within_safety_margin"] is False


def test_memory_detected_from_slurm_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    memory, source = detect_allocated_memory()
    assert memory == 2048 * 1024 * 1024
    assert source == "SLURM_MEM_PER_NODE"


def test_memory_detected_from_slurm_per_cpu(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    memory, source = detect_allocated_memory()
    assert memory == 7 * 1024 * 1024 * 1024
    assert "SLURM_MEM_PER_CPU" in source


def test_memory_unknown_outside_slurm(monkeypatch):
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    memory, source = detect_allocated_memory()
    assert memory is None
    assert "unknown" in source


# --- candidate suggestions ---------------------------------------------------


def test_baseline_is_always_the_first_suggestion(tmp_path):
    uniform_tree(tmp_path, 20)
    assert _experiments(inspect_path(tmp_path))[0] == "loose-file-baseline"


def test_many_small_files_suggest_packaging_and_sharding(tmp_path):
    uniform_tree(tmp_path, MANY_FILES_TRIGGER, size=2048, per_dir=500)
    experiments = _experiments(inspect_path(tmp_path))
    assert "squashfs" in experiments
    assert "webdataset" in experiments


def test_a_few_large_files_suggest_measuring_them_as_they_are(tmp_path):
    build_tree(tmp_path, {f"a{i}.bin": 10 * 1024 * 1024 for i in range(4)})
    experiments = _experiments(inspect_path(tmp_path))
    assert "benchmark-native-representation" in experiments
    assert "squashfs" not in experiments


def test_many_large_files_suggest_streaming_but_not_packaging(tmp_path):
    # Lowering the threshold puts these files above it, which is what the
    # suggestion depends on. Writing genuinely large files would cost gigabytes.
    uniform_tree(tmp_path, MANY_FILES_TRIGGER, size=1024)
    experiments = _experiments(inspect_path(tmp_path, small_file_bytes=512))
    assert "webdataset" in experiments
    assert "squashfs" not in experiments


def test_every_candidate_explains_itself(tmp_path):
    uniform_tree(tmp_path, 50)
    for candidate in inspect_path(tmp_path)["candidates"]:
        assert candidate["experiment"]
        assert len(candidate["reason"]) > 20


def test_limitations_are_always_reported(tmp_path):
    """A reader must never mistake a suggestion for a finding."""
    uniform_tree(tmp_path, 10)
    limitations = inspect_path(tmp_path)["limitations"]
    assert len(limitations) >= 5
    assert any("hypothesis" in text for text in limitations)


# --- validation and rendering ------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"small_file_bytes": 0}, "small_file_bytes"),
        ({"tmp_safety_fraction": 0.0}, "tmp_safety_fraction"),
        ({"tmp_safety_fraction": 1.5}, "tmp_safety_fraction"),
        ({"target_shard_bytes": 0}, "target_shard_bytes"),
    ],
)
def test_invalid_settings_are_rejected(tmp_path, kwargs, message):
    with pytest.raises(InspectionError, match=message):
        inspect_path(tmp_path, **kwargs)


def test_keyvalue_output_shape(tmp_path):
    uniform_tree(tmp_path, 20)
    text = format_keyvalue(inspect_path(tmp_path, memory_bytes=1 << 30))
    for key in (
        "DATASET_PATH",
        "TOTAL_FILES",
        "TOTAL_BYTES",
        "MEDIAN_FILE_BYTES",
        "P95_FILE_BYTES",
        "SMALL_FILE_FRACTION",
        "MAX_DIRECTORY_DEPTH",
        "FILESYSTEM_OBJECTS",
        "DATASET_FRACTION_OF_MEMORY",
        "CANDIDATE_EXPERIMENTS",
    ):
        assert f"{key}=" in text


def test_keyvalue_omits_memory_when_unknown(tmp_path, monkeypatch):
    for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    uniform_tree(tmp_path, 5)
    assert "DATASET_FRACTION_OF_MEMORY" not in format_keyvalue(inspect_path(tmp_path))


def test_report_is_json_serialisable(tmp_path):
    import json

    uniform_tree(tmp_path, 10)
    report = inspect_path(tmp_path)
    assert json.loads(json.dumps(report)) == report


def _experiments(report: dict) -> list[str]:
    return [candidate["experiment"] for candidate in report["candidates"]]
