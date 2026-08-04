#!/usr/bin/env python3
"""Pack a loose-file dataset into tar shards.

    python scripts/build_webdataset.py \\
        --source "$TUTORIAL_ROOT/source" \\
        --manifest "$TUTORIAL_ROOT/manifests/balanced.jsonl" \\
        --output "$TUTORIAL_ROOT/shards" \\
        --samples-per-shard 1000

Shards are written with fixed member metadata, so building twice from the same
manifest and plan produces identical archives. A ``shard_index.json`` records which
samples landed where, which later parts use to check reader assignment.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.manifest import ManifestError, read_manifest  # noqa: E402
from dataaware.shards import (  # noqa: E402
    BALANCE_KEYS,
    ShardError,
    ShardPlan,
    build_shards,
    shard_statistics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True, help="loose-file dataset root")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="shard directory")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument(
        "--balance-by",
        choices=sorted(BALANCE_KEYS),
        default="count",
        help=(
            "'count' gives shards equal sample counts; 'work' gives them similar "
            "total estimated decode cost (default: %(default)s)"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--imbalance-factor",
        type=float,
        default=1.0,
        help=(
            "deliberately unbalance shard sizes by this factor, for the Part VI "
            "imbalance challenge; 1.0 means balanced (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="keep manifest order instead of shuffling before sharding",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing shard directory"
    )
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        print(
            f"ERROR {args.output} is not empty; pass --overwrite to rebuild",
            file=sys.stderr,
        )
        return 2

    try:
        samples = read_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    if not samples:
        print(f"ERROR {args.manifest} contains no samples", file=sys.stderr)
        return 2

    plan = ShardPlan(
        samples_per_shard=args.samples_per_shard,
        shuffle_before_sharding=not args.no_shuffle,
        seed=args.seed,
        balance_by=args.balance_by,
        imbalance_factor=args.imbalance_factor,
    )

    if args.overwrite:
        for stale in sorted(args.output.glob("shard-*.tar")):
            stale.unlink()

    try:
        index = build_shards(
            source_root=args.source,
            samples=samples,
            output_dir=args.output,
            plan=plan,
            progress_every=max(1, len(samples) // 10),
        )
    except (ShardError, OSError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    statistics = shard_statistics(index)
    print(f"SHARD_DIR={args.output}")
    print(f"SHARD_INDEX={args.output / 'shard_index.json'}")
    print(f"TOTAL_SAMPLES={index['total_samples']}")
    print(f"TOTAL_BYTES={index['total_bytes']}")
    print(f"NUM_SHARDS={statistics['num_shards']}")
    print(f"MEAN_SHARD_BYTES={statistics['mean_shard_bytes']:.0f}")
    print(f"MIN_SHARD_BYTES={statistics['min_shard_bytes']}")
    print(f"MAX_SHARD_BYTES={statistics['max_shard_bytes']}")
    print(f"SHARD_BYTES_CV={statistics['shard_bytes_cv']:.4g}")
    print(f"SHARD_WORK_CV={statistics['shard_work_cv']:.4g}")
    print(f"SAMPLES_PER_SHARD={statistics['samples_per_shard']}")
    print(f"BALANCE_BY={plan.balance_by}")
    print(f"IMBALANCE_FACTOR={plan.imbalance_factor}")

    if index["total_samples"] != len(samples):
        print(
            f"WARNING packed {index['total_samples']} of {len(samples)} manifest "
            "samples",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
