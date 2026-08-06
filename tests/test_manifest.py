"""Manifest parsing, stable IDs, and the hash that keeps comparisons honest."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from dataaware.manifest import (
    DataError,
    Sample,
    manifest_hash,
    read_manifest,
    write_manifest,
)


def make_sample(index: int = 0, **overrides) -> Sample:
    fields = {
        "sample_id": f"s{index:08d}",
        "relative_path": f"images/class_0000/s{index:08d}.jpg",
        "class_id": 0,
        "byte_size": 1024,
        "width": 32,
        "height": 32,
        "checksum": "0123456789abcdef",
        "estimated_decode_cost": 1024,
    }
    fields.update(overrides)
    return Sample(**fields)


def test_records_are_sorted_by_sample_id(tmp_path):
    unordered = [make_sample(3), make_sample(1), make_sample(2)]
    path = write_manifest(tmp_path / "m.jsonl", unordered)
    ids = [sample.sample_id for sample in read_manifest(path)]
    assert ids == sorted(ids)


def test_writing_is_byte_identical_regardless_of_input_order(tmp_path):
    """Two generations that produce the same samples must produce the same file."""
    samples = [make_sample(i) for i in range(5)]
    first = write_manifest(tmp_path / "a.jsonl", samples)
    second = write_manifest(tmp_path / "b.jsonl", list(reversed(samples)))
    assert first.read_bytes() == second.read_bytes()
    assert manifest_hash(first) == manifest_hash(second)


def test_duplicate_sample_id_on_write_is_rejected(tmp_path):
    with pytest.raises(DataError, match="duplicate sample_id"):
        write_manifest(tmp_path / "m.jsonl", [make_sample(1), make_sample(1)])


def test_duplicate_sample_id_on_read_is_rejected(tmp_path):
    path = tmp_path / "m.jsonl"
    record = json.dumps(asdict(make_sample(1)), sort_keys=True)
    path.write_text(record + "\n" + record + "\n")
    with pytest.raises(DataError, match="duplicate sample_id"):
        read_manifest(path)


def test_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "m.jsonl"
    record = asdict(make_sample(1)).copy()
    record["extra"] = 1
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(DataError, match=r"unknown field\(s\)"):
        read_manifest(path)


def test_manifest_hash_detects_any_change(tmp_path):
    original = write_manifest(tmp_path / "a.jsonl", [make_sample(0)])
    before = manifest_hash(original)
    changed = write_manifest(tmp_path / "b.jsonl", [make_sample(0, byte_size=2048)])
    assert before != manifest_hash(changed)


