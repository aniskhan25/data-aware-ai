"""The adapter mechanism and the optional format tracks.

The property that matters: an optional track must return the *same bytes* as the core
layouts. If it does not, a comparison between them is meaningless however fast either
one looks. Tests for a track skip themselves when its dependency is absent, which is the
same isolation the core tutorial relies on.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from dataaware.adapters import AdapterError, DatasetAdapter, load_adapter
from dataaware.manifest import checksum_bytes, read_manifest

has_pyarrow = importlib.util.find_spec("pyarrow") is not None
has_h5py = importlib.util.find_spec("h5py") is not None
has_datasets = importlib.util.find_spec("datasets") is not None

needs_pyarrow = pytest.mark.skipif(not has_pyarrow, reason="pyarrow is not installed")
needs_h5py = pytest.mark.skipif(not has_h5py, reason="h5py is not installed")
needs_datasets = pytest.mark.skipif(not has_datasets, reason="datasets is not installed")

TRACKS = [
    pytest.param("parquet", "examples.parquet_track:ParquetAdapter", marks=needs_pyarrow),
    pytest.param("hdf5", "examples.hdf5_track:HDF5Adapter", marks=needs_h5py),
    pytest.param(
        "huggingface", "examples.huggingface_track:HuggingFaceAdapter", marks=needs_datasets
    ),
]


def convert_track(track, source_root, samples, output_dir, group_size=64):
    import importlib

    module = importlib.import_module(f"examples.{track}_track")
    kwargs = {"progress_every": 0}
    if track == "parquet":
        kwargs["row_group_size"] = group_size
    elif track == "hdf5":
        kwargs["chunk_size"] = group_size
    else:
        kwargs["writer_batch_size"] = group_size
    return module.convert(source_root, samples, output_dir, **kwargs)


# --- the mechanism -----------------------------------------------------------


class _Stub(DatasetAdapter):
    name = "stub"
    opened = 0

    def open_resource(self):
        type(self).opened += 1
        return {"pid": os.getpid()}

    def read_payload(self, index):
        return b"x" * (index + 1)


def test_loading_requires_a_module_and_class():
    with pytest.raises(AdapterError, match="module:Class"):
        load_adapter("examples.parquet_track", "/tmp", [])


def test_an_unimportable_module_names_the_extras():
    with pytest.raises(AdapterError, match=r"pip install"):
        load_adapter("definitely_not_a_module:Thing", "/tmp", [])


def test_a_missing_class_is_reported():
    with pytest.raises(AdapterError, match="has no attribute"):
        load_adapter("dataaware.adapters:NoSuchAdapter", "/tmp", [])


def test_a_non_adapter_class_is_rejected():
    with pytest.raises(AdapterError, match="not a DatasetAdapter"):
        load_adapter("dataaware.manifest:Sample", "/tmp", [])


def test_resources_open_lazily_and_once_per_process():
    """Opening in __init__ would share a handle across DataLoader workers."""
    _Stub.opened = 0
    adapter = _Stub(root="/tmp", samples=[None, None])
    assert _Stub.opened == 0, "nothing may be opened before first use"

    first = adapter.resource()
    assert _Stub.opened == 1
    assert adapter.resource() is first, "the handle must be reused within a process"
    assert _Stub.opened == 1


def test_a_resource_is_reopened_after_a_simulated_fork():
    _Stub.opened = 0
    adapter = _Stub(root="/tmp", samples=[])
    adapter.resource()
    # Pretend this process is a fresh fork: the recorded pid no longer matches.
    adapter._resource_pid = -1
    adapter.resource()
    assert _Stub.opened == 2


def test_closing_is_safe_when_nothing_was_opened():
    _Stub(root="/tmp", samples=[]).close()


def test_the_default_describe_is_empty():
    assert _Stub(root="/tmp", samples=[]).describe() == {}


# --- the tracks --------------------------------------------------------------


@pytest.mark.parametrize(("track", "spec"), TRACKS)
def test_a_track_returns_the_same_bytes_as_the_manifest(track, spec, tiny_dataset, tmp_path):
    """The property that makes an optional track comparable with a core layout."""
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / track
    result = convert_track(track, root, samples, output)
    assert result["rows"] == len(samples)

    adapter = load_adapter(spec, output, samples)
    try:
        for index, sample in enumerate(samples):
            assert checksum_bytes(adapter.read_payload(index)) == sample.checksum
    finally:
        adapter.close()


@pytest.mark.parametrize(("track", "spec"), TRACKS)
def test_a_track_reports_artifact_metrics(track, spec, tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / track
    convert_track(track, root, samples, output)

    adapter = load_adapter(spec, output, samples)
    try:
        described = adapter.describe()
    finally:
        adapter.close()

    assert described["artifact_bytes"] > 0
    assert described["filesystem_objects"] >= 1
    # Every reported key must already exist in the schema, which rejects unknown fields.
    from dataaware.schema import COMMON_FIELDS, OPTIONAL_FIELDS

    for key in described:
        assert key in COMMON_FIELDS or key in OPTIONAL_FIELDS, key


@pytest.mark.parametrize(("track", "spec"), TRACKS)
def test_a_track_rejects_an_artifact_that_does_not_match_the_manifest(
    track, spec, tiny_dataset, tmp_path
):
    """Reading a stale artifact against a newer manifest would silently misalign."""
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / track
    convert_track(track, root, samples[: len(samples) // 2], output)

    adapter = load_adapter(spec, output, samples)
    with pytest.raises(AdapterError, match="Reconvert from the same manifest"):
        adapter.read_payload(0)


@pytest.mark.parametrize(("track", "spec"), TRACKS)
def test_a_missing_artifact_names_the_converter(track, spec, tiny_dataset, tmp_path):
    _, manifest = tiny_dataset
    adapter = load_adapter(spec, tmp_path / "absent", read_manifest(manifest))
    with pytest.raises(AdapterError, match="convert_dataset.py"):
        adapter.read_payload(0)


@needs_pyarrow
def test_parquet_row_groups_follow_the_requested_size(tiny_dataset, tmp_path):
    """Row-group size is the knob that decides how much is read to reach one row."""
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    result = convert_track("parquet", root, samples, tmp_path / "p", group_size=8)
    assert result["row_groups"] == -(-len(samples) // 8)


@needs_pyarrow
def test_parquet_reads_correctly_across_row_group_boundaries(tiny_dataset, tmp_path):
    """The group cache must not return a stale row when access jumps between groups."""
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / "p"
    convert_track("parquet", root, samples, output, group_size=4)

    adapter = load_adapter("examples.parquet_track:ParquetAdapter", output, samples)
    try:
        # Deliberately alternate between distant groups.
        order = [0, len(samples) - 1, 5, len(samples) - 2, 1]
        for index in order:
            assert checksum_bytes(adapter.read_payload(index)) == samples[index].checksum
    finally:
        adapter.close()


@needs_h5py
def test_hdf5_chunking_follows_the_requested_size(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / "h"
    convert_track("hdf5", root, samples, output, group_size=32)

    adapter = load_adapter("examples.hdf5_track:HDF5Adapter", output, samples)
    try:
        assert adapter.describe()["chunk_size"] == 32
    finally:
        adapter.close()


@needs_datasets
def test_huggingface_cache_advice_points_away_from_home(tmp_path):
    """The cache defaults to home, which is small and the wrong filesystem for job I/O."""
    from examples.huggingface_track import cache_dir_advice

    advice = cache_dir_advice(tmp_path)
    assert str(tmp_path) in advice["HF_HOME"]
    assert str(tmp_path) in advice["HF_DATASETS_CACHE"]
    assert advice["HF_DATASETS_OFFLINE"] == "1"


# --- through the shared loader ----------------------------------------------


@pytest.mark.parametrize(("track", "spec"), TRACKS)
def test_a_track_measures_through_the_shared_loader(
    track, spec, tiny_dataset, tmp_path, base_config_dict
):
    """An optional track must produce a schema-valid summary, labelled with its own name."""
    pytest.importorskip("torch", reason="the loader benchmark requires PyTorch")
    from dataaware.config import config_from_dict
    from dataaware.loaders import run_loader_benchmark
    from dataaware.schema import validate_run_summary

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    output = tmp_path / track
    convert_track(track, root, samples, output)

    base_config_dict["dataset"]["layout"] = "adapter"
    base_config_dict["dataset"]["adapter"] = spec
    base_config_dict["dataset"]["root"] = str(output)
    summary = validate_run_summary(
        run_loader_benchmark(config_from_dict(base_config_dict))
    )

    assert summary["layout"] == track, "each track must be its own row in a comparison"
    assert summary["adapter"] == spec
    assert summary["failed_samples"] == 0
    assert summary["duplicate_samples"] == 0
    assert summary["bytes_read"] > 0
    assert summary["artifact_bytes"] > 0


@needs_pyarrow
def test_a_track_reads_the_same_bytes_as_the_loose_files(tiny_dataset, tmp_path, base_config_dict):
    """Same manifest, same bytes: the precondition for comparing the two runs."""
    pytest.importorskip("torch", reason="the loader benchmark requires PyTorch")
    from dataaware.config import config_from_dict
    from dataaware.loaders import run_loader_benchmark

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    loose = run_loader_benchmark(config_from_dict(base_config_dict))

    output = tmp_path / "parquet"
    convert_track("parquet", root, samples, output)
    base_config_dict["dataset"]["layout"] = "adapter"
    base_config_dict["dataset"]["adapter"] = "examples.parquet_track:ParquetAdapter"
    base_config_dict["dataset"]["root"] = str(output)
    converted = run_loader_benchmark(config_from_dict(base_config_dict))

    assert converted["bytes_read"] == loose["bytes_read"]
    assert converted["manifest_hash"] == loose["manifest_hash"]
    assert converted["samples_measured"] == loose["samples_measured"]
