"""Dataset generation.

The tutorial's conclusions depend on every learner measuring the same bytes, so
determinism is a correctness property here, not a nicety.
"""

from __future__ import annotations

from dataclasses import replace


from dataaware.generate import (
    PROFILES,
    encode_sample,
    generate_dataset,
    load_profile,
)
from dataaware.manifest import checksum_bytes, read_manifest

TEST_PROFILE = replace(PROFILES["tiny"], samples=16, classes=4)


def test_sample_bytes_depend_only_on_seed_and_index():
    first = encode_sample(TEST_PROFILE, 5)
    second = encode_sample(TEST_PROFILE, 5)
    assert first == second


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


def test_shipped_dataset_profiles_load(repo_root):
    paths = sorted((repo_root / "configs/datasets").glob("*.yaml"))
    assert paths, "no dataset profiles found"
    for path in paths:
        profile = load_profile(path)
        assert profile.samples >= 1


