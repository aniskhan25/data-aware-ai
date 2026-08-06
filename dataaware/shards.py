"""Tar-based sample shards, and a reader that streams them.

This is the WebDataset convention: each sample is a group of members in a tar
archive sharing a basename, written in order so that reading a shard is a
sequential pass over one file.

    shard-00000.tar
        s00000000.jpg   <- the sample bytes
        s00000000.cls   <- its class label
        s00000001.jpg
        s00000001.cls

Implemented on the standard library's ``tarfile`` rather than the ``webdataset``
package, for two reasons: the tutorial gains no dependency, and shard-to-reader
assignment stays visible in code you can read. Assignment is the thing Part VI
tests, so it should not be hidden inside a library.
"""

from __future__ import annotations

import io
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import metrics
from .manifest import Sample
from .errors import DataError

SHARD_INDEX_VERSION = "1.0"

#: Balancing keys. ``count`` gives every shard the same number of samples;
#: ``work`` gives every shard a similar total ``estimated_decode_cost``. Equal
#: counts do not imply equal work, which is why both exist.
BALANCE_KEYS = ("count", "work")


@dataclass(frozen=True)
class ShardPlan:
    """How samples are distributed over shards."""

    samples_per_shard: int = 1000
    shuffle_before_sharding: bool = True
    seed: int = 1234
    balance_by: str = "count"
    #: Deliberately unbalance shard sizes by this factor, for the Part VI imbalance
    #: challenge. 1 means balanced. With factor N the largest shard holds about N times
    #: as many samples as the smallest, so ranks receive visibly unequal work while
    #: every rank still gets the same *number* of shards.
    imbalance_factor: float = 1.0

    def validate(self) -> None:
        if self.samples_per_shard < 1:
            raise DataError("samples_per_shard must be >= 1")
        if self.balance_by not in BALANCE_KEYS:
            raise DataError(f"balance_by must be one of {list(BALANCE_KEYS)}")
        if self.imbalance_factor < 1.0:
            raise DataError("imbalance_factor must be >= 1.0")


def plan_shards(samples: Sequence[Sample], plan: ShardPlan) -> list[list[Sample]]:
    """Split samples into shards.

    Deterministic for a fixed seed: the shuffle uses a seeded generator and the
    balancing pass is a stable sort, so the same manifest and plan always produce
    the same shards.
    """
    plan.validate()
    if not samples:
        return []

    ordered = list(samples)
    if plan.shuffle_before_sharding:
        random.Random(plan.seed).shuffle(ordered)

    if plan.imbalance_factor > 1.0:
        return _imbalanced(ordered, plan)

    if plan.balance_by == "count":
        # Contiguous chunks. Keeps the shuffled order inside each shard, so a
        # sequential read of one shard is a random sample of the dataset.
        return [
            ordered[index : index + plan.samples_per_shard]
            for index in range(0, len(ordered), plan.samples_per_shard)
        ]

    shard_count = max(1, -(-len(ordered) // plan.samples_per_shard))
    return _balance_by_work(ordered, shard_count)


def _imbalanced(samples: list[Sample], plan: ShardPlan) -> list[list[Sample]]:
    """Split into deliberately unequal shards, for the imbalance challenge.

    Sizes are drawn at random (seeded) from a range spanning ``imbalance_factor``, not
    ramped from smallest to largest. That matters: reader assignment is round-robin, so
    a monotonic ramp hands every reader one shard from each size band and the totals
    come out *balanced* - the challenge would then only appear to work when the shard
    count happened not to divide evenly among readers, which is a different defect.

    Random sizes are also what real datasets look like: shards differ because the
    samples in them differ. Every reader still receives the same number of shards,
    which is the lesson - equal shard counts per rank do not mean equal work per rank.
    """
    shard_count = max(1, -(-len(samples) // plan.samples_per_shard))
    if shard_count == 1:
        return [samples]

    rng = random.Random(plan.seed + 1)
    weights = [rng.uniform(1.0, plan.imbalance_factor) for _ in range(shard_count)]
    total_weight = sum(weights)
    groups: list[list[Sample]] = []
    start = 0
    for position, weight in enumerate(weights):
        if position == len(weights) - 1:
            chunk = samples[start:]
        else:
            size = max(1, int(round(len(samples) * weight / total_weight)))
            chunk = samples[start : start + size]
            start += len(chunk)
        if chunk:
            groups.append(chunk)
    return groups


def _balance_by_work(samples: list[Sample], shard_count: int) -> list[list[Sample]]:
    """Greedy longest-processing-time-first packing on estimated decode cost.

    Equal sample counts per shard do not mean equal work per shard: a shard of
    large images takes longer than a shard of small ones, and with synchronised
    ranks the slowest shard sets the pace.
    """
    buckets: list[list[Sample]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    # Descending cost, then sample_id so ties break deterministically.
    for sample in sorted(samples, key=lambda s: (-s.estimated_decode_cost, s.sample_id)):
        target = loads.index(min(loads))
        buckets[target].append(sample)
        loads[target] += sample.estimated_decode_cost
    return [bucket for bucket in buckets if bucket]


def build_shards(
    source_root: str | Path,
    samples: Sequence[Sample],
    output_dir: str | Path,
    plan: ShardPlan,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Write shards from a loose-file tree and return the shard index.

    The index records which samples landed in which shard, with byte sizes and
    estimated work. Later parts need it to check that readers are assigned
    disjoint, comparably sized shards.
    """
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = plan_shards(samples, plan)
    shard_records = []
    written = 0

    for shard_id, shard_samples in enumerate(grouped):
        name = f"shard-{shard_id:05d}.tar"
        target = output_dir / name
        with tarfile.open(target, "w") as archive:
            for sample in shard_samples:
                _add_sample(archive, source_root, sample)
                written += 1
                if progress_every and written % progress_every == 0:
                    print(f"packed {written}/{len(samples)} samples", flush=True)

        shard_records.append(
            {
                "shard": name,
                "samples": len(shard_samples),
                "bytes": target.stat().st_size,
                "sample_bytes": sum(s.byte_size for s in shard_samples),
                "estimated_work": sum(s.estimated_decode_cost for s in shard_samples),
                "sample_ids": [s.sample_id for s in shard_samples],
            }
        )

    index = {
        "schema_version": SHARD_INDEX_VERSION,
        "plan": {
            "samples_per_shard": plan.samples_per_shard,
            "shuffle_before_sharding": plan.shuffle_before_sharding,
            "seed": plan.seed,
            "balance_by": plan.balance_by,
            "imbalance_factor": plan.imbalance_factor,
        },
        "total_samples": sum(record["samples"] for record in shard_records),
        "total_bytes": sum(record["bytes"] for record in shard_records),
        "shards": shard_records,
    }
    write_shard_index(output_dir / "shard_index.json", index)
    return index


def _add_sample(archive: tarfile.TarFile, source_root: Path, sample: Sample) -> None:
    """Append one sample's members, with reproducible metadata.

    Timestamps, ownership, and permissions are fixed so that building the same
    shards twice produces identical archives. Without that, a rebuild would change
    every shard's bytes and there would be no way to tell a content change from a
    repackaging.
    """
    payload = (source_root / sample.relative_path).read_bytes()
    suffix = Path(sample.relative_path).suffix.lstrip(".") or "bin"
    _append(archive, f"{sample.sample_id}.{suffix}", payload)
    _append(archive, f"{sample.sample_id}.cls", str(sample.class_id).encode())


def _append(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def write_shard_index(path: str | Path, index: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return path


def read_shard_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        index = json.loads(path.read_text())
    except FileNotFoundError:
        raise DataError(
            f"shard index not found: {path}\n"
            "Build shards first with scripts/build_webdataset.py."
        ) from None
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON: {exc}") from None

    if index.get("schema_version") != SHARD_INDEX_VERSION:
        raise DataError(
            f"{path}: shard index version {index.get('schema_version')!r} is not "
            f"supported (expected {SHARD_INDEX_VERSION!r})"
        )
    for key in ("shards", "total_samples", "plan"):
        if key not in index:
            raise DataError(f"{path}: shard index is missing '{key}'")
    if not index["shards"]:
        raise DataError(f"{path}: shard index lists no shards")
    return index


def shard_statistics(index: dict[str, Any]) -> dict[str, Any]:
    """Shard size and balance statistics, for the run summary."""
    sizes = [record["bytes"] for record in index["shards"]]
    works = [record["estimated_work"] for record in index["shards"]]
    counts = [record["samples"] for record in index["shards"]]
    return {
        "num_shards": len(sizes),
        "mean_shard_bytes": metrics.mean(sizes),
        "min_shard_bytes": min(sizes),
        "max_shard_bytes": max(sizes),
        "shard_bytes_cv": metrics.coefficient_of_variation(sizes),
        "shard_work_cv": metrics.coefficient_of_variation(works),
        "samples_per_shard": min(counts),
        "max_samples_per_shard": max(counts),
    }


def iter_shard_samples(shard_path: Path) -> Iterator[tuple[str, bytes, int]]:
    """Yield ``(sample_id, payload, class_id)`` from one shard, in stored order.

    Members are grouped by basename. A sample missing either member is skipped
    rather than guessed at, and the caller sees it as a coverage gap.
    """
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard_path, "r|") as archive:
        # Stream mode ("r|") reads forward only, which is the access pattern shards
        # exist for: no seeking back and forth inside the archive.
        for member in archive:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            suffix = Path(member.name).suffix.lstrip(".")
            handle = archive.extractfile(member)
            if handle is None:
                continue
            group = pending.setdefault(stem, {})
            group[suffix] = handle.read()

            if "cls" in group and len(group) >= 2:
                payload_key = next(key for key in group if key != "cls")
                yield stem, group[payload_key], int(group["cls"])
                del pending[stem]


def assign_shards(
    shards: Sequence[str],
    index: int,
    total: int,
) -> list[str]:
    """Assign shards to one reader out of ``total``, round-robin.

    Round-robin rather than contiguous blocks: when shard sizes vary, interleaving
    spreads large shards across readers instead of concentrating them.

    When there are fewer shards than readers, some readers get nothing. That is
    not corrected here - it is a real failure mode the tutorial makes visible in
    Part VI, and silently duplicating shards to fill idle readers would hide it.
    """
    if total < 1:
        raise DataError("total readers must be >= 1")
    if not 0 <= index < total:
        raise DataError(f"reader index {index} is outside 0..{total - 1}")
    return list(shards[index::total])
