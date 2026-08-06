"""Tar shard planning, building, and reader assignment.

Assignment is the property that matters most here: many readers must cover a
dataset without any two reading the same bytes. Needs no PyTorch.
"""

from __future__ import annotations


import pytest

from dataaware.manifest import Sample, read_manifest
from dataaware.shards import (
    ShardPlan,
    assign_shards,
    build_shards,
    iter_shard_samples,
    plan_shards,
    shard_statistics,
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


def test_no_shuffle_preserves_manifest_order():
    samples = make_samples(10)
    groups = plan_shards(
        samples, ShardPlan(samples_per_shard=5, shuffle_before_sharding=False)
    )
    assert [s.sample_id for s in groups[0]] == [s.sample_id for s in samples[:5]]


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


# --- index -------------------------------------------------------------------


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


# --- reading -----------------------------------------------------------------


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
