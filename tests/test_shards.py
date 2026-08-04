"""Tar shard planning, building, and reader assignment.

Assignment is the property that matters most here: many readers must cover a
dataset without any two reading the same bytes. Needs no PyTorch.
"""

from __future__ import annotations

import tarfile

import pytest

from dataaware.manifest import Sample, read_manifest
from dataaware.shards import (
    ShardError,
    ShardPlan,
    assign_shards,
    build_shards,
    iter_shard_samples,
    plan_shards,
    read_shard_index,
    shard_statistics,
    write_shard_index,
)


def make_samples(count: int, cost: int = 1024) -> list[Sample]:
    return [
        Sample(
            sample_id=f"s{index:08d}",
            relative_path=f"images/class_0000/s{index:08d}.jpg",
            class_id=index % 4,
            byte_size=100,
            width=32,
            height=32,
            checksum="0" * 16,
            estimated_decode_cost=cost,
        )
        for index in range(count)
    ]


# --- planning ----------------------------------------------------------------


def test_planning_covers_every_sample_exactly_once():
    samples = make_samples(25)
    groups = plan_shards(samples, ShardPlan(samples_per_shard=10))
    packed = [sample.sample_id for group in groups for sample in group]
    assert len(groups) == 3
    assert sorted(packed) == sorted(s.sample_id for s in samples)
    assert len(set(packed)) == 25


def test_planning_is_deterministic_for_a_seed():
    samples = make_samples(30)
    plan = ShardPlan(samples_per_shard=7, seed=99)
    first = [[s.sample_id for s in group] for group in plan_shards(samples, plan)]
    second = [[s.sample_id for s in group] for group in plan_shards(samples, plan)]
    assert first == second


def test_a_different_seed_changes_the_arrangement():
    samples = make_samples(30)
    first = plan_shards(samples, ShardPlan(samples_per_shard=7, seed=1))
    second = plan_shards(samples, ShardPlan(samples_per_shard=7, seed=2))
    assert [s.sample_id for s in first[0]] != [s.sample_id for s in second[0]]


def test_no_shuffle_preserves_manifest_order():
    samples = make_samples(10)
    groups = plan_shards(
        samples, ShardPlan(samples_per_shard=5, shuffle_before_sharding=False)
    )
    assert [s.sample_id for s in groups[0]] == [s.sample_id for s in samples[:5]]


def test_empty_manifest_plans_no_shards():
    assert plan_shards([], ShardPlan()) == []


def test_balancing_by_work_evens_out_estimated_cost():
    """Equal sample counts do not mean equal work; balancing by work should."""
    cheap = make_samples(20, cost=100)
    expensive = [
        Sample(**{**s.__dict__, "sample_id": f"b{i:08d}", "estimated_decode_cost": 10_000})
        for i, s in enumerate(make_samples(4))
    ]
    groups = plan_shards(cheap + expensive, ShardPlan(samples_per_shard=6, balance_by="work"))

    loads = [sum(s.estimated_decode_cost for s in group) for group in groups]
    spread = (max(loads) - min(loads)) / max(loads)
    assert spread < 0.35, f"work still uneven across shards: {loads}"


def test_balancing_by_count_leaves_work_uneven():
    """The contrast the tutorial teaches: equal counts, unequal work."""
    cheap = make_samples(20, cost=100)
    expensive = [
        Sample(**{**s.__dict__, "sample_id": f"b{i:08d}", "estimated_decode_cost": 10_000})
        for i, s in enumerate(make_samples(4))
    ]
    groups = plan_shards(cheap + expensive, ShardPlan(samples_per_shard=6, balance_by="count"))
    loads = [sum(s.estimated_decode_cost for s in group) for group in groups]
    assert max(loads) > 2 * min(loads)


def test_balancing_by_work_still_covers_everything():
    samples = make_samples(31)
    groups = plan_shards(samples, ShardPlan(samples_per_shard=8, balance_by="work"))
    packed = [s.sample_id for group in groups for s in group]
    assert sorted(packed) == sorted(s.sample_id for s in samples)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples_per_shard": 0}, "samples_per_shard"),
        ({"balance_by": "size"}, "balance_by"),
    ],
)
def test_invalid_plans_are_rejected(kwargs, message):
    with pytest.raises(ShardError, match=message):
        ShardPlan(**kwargs).validate()


# --- building ----------------------------------------------------------------


def test_building_round_trips_every_sample(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    index = build_shards(root, samples, tmp_path / "shards", ShardPlan(samples_per_shard=8))

    assert index["total_samples"] == len(samples)
    recovered = {}
    for record in index["shards"]:
        for sample_id, payload, class_id in iter_shard_samples(
            tmp_path / "shards" / record["shard"]
        ):
            recovered[sample_id] = (payload, class_id)

    assert set(recovered) == {s.sample_id for s in samples}
    by_id = {s.sample_id: s for s in samples}
    for sample_id, (payload, class_id) in recovered.items():
        assert len(payload) == by_id[sample_id].byte_size
        assert class_id == by_id[sample_id].class_id


def test_shard_bytes_match_the_original_files(tiny_dataset, tmp_path):
    """Packaging must not alter sample bytes, or the comparison is meaningless."""
    from dataaware.manifest import checksum_bytes

    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    index = build_shards(root, samples, tmp_path / "shards", ShardPlan(samples_per_shard=8))

    by_id = {s.sample_id: s for s in samples}
    for record in index["shards"]:
        for sample_id, payload, _ in iter_shard_samples(
            tmp_path / "shards" / record["shard"]
        ):
            assert checksum_bytes(payload) == by_id[sample_id].checksum


def test_rebuilding_produces_identical_archives(tiny_dataset, tmp_path):
    """Fixed member metadata means a rebuild is byte-identical."""
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    plan = ShardPlan(samples_per_shard=8)
    build_shards(root, samples, tmp_path / "a", plan)
    build_shards(root, samples, tmp_path / "b", plan)

    for shard in sorted((tmp_path / "a").glob("shard-*.tar")):
        assert shard.read_bytes() == (tmp_path / "b" / shard.name).read_bytes()


def test_index_records_sizes_and_work(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    index = build_shards(root, samples, tmp_path / "shards", ShardPlan(samples_per_shard=8))

    for record in index["shards"]:
        assert record["samples"] == len(record["sample_ids"])
        assert record["bytes"] > 0
        assert record["estimated_work"] > 0
    assert sum(r["samples"] for r in index["shards"]) == len(samples)


def test_no_partial_shards_are_left_behind(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    build_shards(
        root, read_manifest(manifest), tmp_path / "shards", ShardPlan(samples_per_shard=8)
    )
    assert not list((tmp_path / "shards").glob("*.partial"))


# --- index -------------------------------------------------------------------


def test_missing_index_is_reported_clearly(tmp_path):
    with pytest.raises(ShardError, match="build_webdataset.py"):
        read_shard_index(tmp_path / "shard_index.json")


def test_unsupported_index_version_is_rejected(tmp_path):
    path = write_shard_index(
        tmp_path / "shard_index.json",
        {"schema_version": "0.1", "shards": [{}], "total_samples": 1, "plan": {}},
    )
    with pytest.raises(ShardError, match="not supported"):
        read_shard_index(path)


def test_index_missing_required_key_is_rejected(tmp_path):
    path = write_shard_index(
        tmp_path / "shard_index.json", {"schema_version": "1.0", "shards": [{}]}
    )
    with pytest.raises(ShardError, match="missing"):
        read_shard_index(path)


def test_index_with_no_shards_is_rejected(tmp_path):
    path = write_shard_index(
        tmp_path / "shard_index.json",
        {"schema_version": "1.0", "shards": [], "total_samples": 0, "plan": {}},
    )
    with pytest.raises(ShardError, match="no shards"):
        read_shard_index(path)


def test_statistics_describe_balance(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    index = build_shards(
        root, read_manifest(manifest), tmp_path / "shards", ShardPlan(samples_per_shard=8)
    )
    statistics = shard_statistics(index)
    assert statistics["num_shards"] == len(index["shards"])
    assert statistics["min_shard_bytes"] <= statistics["mean_shard_bytes"]
    assert statistics["mean_shard_bytes"] <= statistics["max_shard_bytes"]
    assert statistics["shard_bytes_cv"] >= 0.0


# --- reader assignment -------------------------------------------------------


@pytest.mark.parametrize("readers", [1, 2, 3, 4, 8])
def test_assignment_is_disjoint_and_complete(readers):
    shards = [f"shard-{i:05d}.tar" for i in range(8)]
    parts = [assign_shards(shards, index, readers) for index in range(readers)]
    flat = [shard for part in parts for shard in part]
    assert sorted(flat) == sorted(shards)
    assert len(set(flat)) == len(shards)


def test_too_few_shards_leaves_readers_idle_rather_than_duplicating():
    """The Part VI failure mode. Filling idle readers would hide it."""
    shards = ["shard-00000.tar", "shard-00001.tar"]
    parts = [assign_shards(shards, index, 8) for index in range(8)]
    assert sum(1 for part in parts if not part) == 6
    flat = [shard for part in parts for shard in part]
    assert len(flat) == len(set(flat)) == 2


def test_assignment_interleaves_so_large_shards_spread_out():
    shards = [f"shard-{i:05d}.tar" for i in range(6)]
    assert assign_shards(shards, 0, 3) == [
        "shard-00000.tar",
        "shard-00003.tar",
    ]


@pytest.mark.parametrize(("index", "total"), [(0, 0), (-1, 2), (2, 2), (5, 3)])
def test_invalid_reader_indices_are_rejected(index, total):
    with pytest.raises(ShardError):
        assign_shards(["a"], index, total)


# --- reading -----------------------------------------------------------------


def test_a_sample_missing_its_label_is_skipped(tmp_path):
    """An incomplete group is skipped rather than guessed at."""
    shard = tmp_path / "shard-00000.tar"
    with tarfile.open(shard, "w") as archive:
        for name, payload in (
            ("s00000000.jpg", b"image-bytes"),
            ("s00000000.cls", b"3"),
            ("s00000001.jpg", b"orphan-without-a-label"),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            import io

            archive.addfile(info, io.BytesIO(payload))

    recovered = list(iter_shard_samples(shard))
    assert [sample_id for sample_id, _, _ in recovered] == ["s00000000"]


def test_reading_preserves_stored_order(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    samples = read_manifest(manifest)
    index = build_shards(
        root,
        samples,
        tmp_path / "shards",
        ShardPlan(samples_per_shard=8, shuffle_before_sharding=False),
    )
    first = index["shards"][0]
    read_ids = [
        sample_id
        for sample_id, _, _ in iter_shard_samples(tmp_path / "shards" / first["shard"])
    ]
    assert read_ids == first["sample_ids"]


def test_random_imbalance_survives_round_robin_assignment():
    """A monotonic size ramp would be cancelled by round-robin assignment.

    Each reader would receive one shard from every size band, so the totals would come
    out balanced and the imbalance challenge would demonstrate nothing. Sizes are drawn
    at random for exactly this reason.
    """
    samples = make_samples(4000)
    names_and_loads = {}
    for factor in (1.0, 6.0):
        groups = plan_shards(
            samples, ShardPlan(samples_per_shard=100, imbalance_factor=factor)
        )
        names = [f"shard-{index:05d}" for index in range(len(groups))]
        size_by_name = dict(zip(names, (len(group) for group in groups)))
        loads = [
            sum(size_by_name[name] for name in assign_shards(names, reader, 8))
            for reader in range(8)
        ]
        counts = {len(assign_shards(names, reader, 8)) for reader in range(8)}
        # Equal shard counts per reader in both cases: only the work differs.
        assert counts == {5}
        names_and_loads[factor] = loads

    assert len(set(names_and_loads[1.0])) == 1, "balanced shards must give equal loads"
    balanced_spread = 0.0
    imbalanced = names_and_loads[6.0]
    imbalanced_spread = (max(imbalanced) - min(imbalanced)) / max(imbalanced)
    assert imbalanced_spread > 0.2, f"imbalance was cancelled out: {imbalanced}"
    assert imbalanced_spread > balanced_spread
