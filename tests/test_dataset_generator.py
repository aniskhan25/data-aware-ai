"""Dataset generation.

The tutorial's conclusions depend on every learner measuring the same bytes, so
determinism is a correctness property here, not a nicety.
"""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from dataaware.generate import (
    PROFILES,
    Profile,
    encode_sample,
    generate_dataset,
    load_profile,
    relative_path_for,
    sample_id_for,
)
from dataaware.manifest import checksum_bytes, read_manifest

TEST_PROFILE = replace(PROFILES["tiny"], samples=16, classes=4)


def test_sample_bytes_depend_only_on_seed_and_index():
    first = encode_sample(TEST_PROFILE, 5)
    second = encode_sample(TEST_PROFILE, 5)
    assert first == second


def test_different_indices_produce_different_samples():
    assert encode_sample(TEST_PROFILE, 0) != encode_sample(TEST_PROFILE, 1)


def test_different_seeds_produce_different_samples():
    other = replace(TEST_PROFILE, seed=TEST_PROFILE.seed + 1)
    assert encode_sample(TEST_PROFILE, 0) != encode_sample(other, 0)


def test_encoded_image_has_requested_geometry():
    with Image.open(io.BytesIO(encode_sample(TEST_PROFILE, 0))) as image:
        assert image.size == (TEST_PROFILE.width, TEST_PROFILE.height)
        assert image.mode == "RGB"


def test_png_profile_encodes_as_png():
    profile = replace(TEST_PROFILE, encoding="png")
    with Image.open(io.BytesIO(encode_sample(profile, 0))) as image:
        assert image.format == "PNG"


def test_generate_writes_every_sample_and_a_matching_manifest(tmp_path):
    manifest_path = generate_dataset(TEST_PROFILE, tmp_path)
    samples = read_manifest(manifest_path)

    assert len(samples) == TEST_PROFILE.samples
    for sample in samples:
        payload = (tmp_path / sample.relative_path).read_bytes()
        assert len(payload) == sample.byte_size
        assert checksum_bytes(payload) == sample.checksum
        assert sample.estimated_decode_cost == TEST_PROFILE.width * TEST_PROFILE.height


def test_classes_are_balanced(tmp_path):
    samples = read_manifest(generate_dataset(TEST_PROFILE, tmp_path))
    counts: dict[int, int] = {}
    for sample in samples:
        counts[sample.class_id] = counts.get(sample.class_id, 0) + 1
    assert set(counts) == set(range(TEST_PROFILE.classes))
    assert len(set(counts.values())) == 1


def test_worker_count_does_not_change_the_result(tmp_path):
    serial = generate_dataset(TEST_PROFILE, tmp_path / "serial", workers=1)
    parallel = generate_dataset(TEST_PROFILE, tmp_path / "parallel", workers=3)
    assert serial.read_bytes() == parallel.read_bytes()


def test_regeneration_is_byte_identical(tmp_path):
    first = generate_dataset(TEST_PROFILE, tmp_path / "a")
    second = generate_dataset(TEST_PROFILE, tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()


def test_non_empty_output_requires_overwrite(tmp_path):
    generate_dataset(TEST_PROFILE, tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_dataset(TEST_PROFILE, tmp_path)
    generate_dataset(TEST_PROFILE, tmp_path, overwrite=True)


def test_manifest_can_live_outside_the_dataset(tmp_path):
    manifest_path = generate_dataset(
        TEST_PROFILE, tmp_path / "data", manifest_path=tmp_path / "manifests/m.jsonl"
    )
    assert manifest_path == tmp_path / "manifests/m.jsonl"
    assert manifest_path.exists()


def test_profile_record_is_written(tmp_path):
    import json

    generate_dataset(TEST_PROFILE, tmp_path)
    record = json.loads((tmp_path / "dataset_profile.json").read_text())
    assert record["dataset"]["seed"] == TEST_PROFILE.seed
    assert record["total_samples"] == TEST_PROFILE.samples


def test_ids_and_paths_are_stable():
    assert sample_id_for(42) == "s00000042"
    assert relative_path_for(TEST_PROFILE, 5) == "images/class_0001/s00000005.jpg"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 0),
        ("width", 0),
        ("classes", 0),
        ("encoding", "tiff"),
        ("noise", 300),
        ("jpeg_quality", 0),
        ("png_compress_level", 10),
    ],
)
def test_invalid_profile_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(TEST_PROFILE, **{field: value}).validate()


def test_profile_from_mapping_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown dataset profile field"):
        Profile.from_mapping(
            {"profile": "x", "samples": 1, "width": 1, "height": 1, "widht": 2}
        )


def test_profile_from_mapping_requires_core_fields():
    with pytest.raises(ValueError, match="missing 'height'"):
        Profile.from_mapping({"profile": "x", "samples": 1, "width": 1})


def test_shipped_dataset_profiles_load(repo_root):
    paths = sorted((repo_root / "configs/datasets").glob("*.yaml"))
    assert paths, "no dataset profiles found"
    for path in paths:
        profile = load_profile(path)
        assert profile.samples >= 1


def test_shipped_profiles_match_builtin_definitions(repo_root):
    """A profile file and its built-in twin must not drift apart."""
    for path in sorted((repo_root / "configs/datasets").glob("*.yaml")):
        profile = load_profile(path)
        assert profile == PROFILES[profile.profile], f"{path} differs from PROFILES"
