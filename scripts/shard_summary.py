#!/usr/bin/env python3
"""Report a shard set's balance from its index.

    python3 scripts/shard_summary.py "$TUTORIAL_ROOT"/shards

Answers two questions before a distributed run is launched: are there enough shards for
the readers you intend, and are they evenly sized? Both are cheaper to check here than
to diagnose from a rank imbalance afterwards.

Pass --readers to see how the shards would actually be divided, including whether any
reader would be left with nothing.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.metrics import coefficient_of_variation, spread  # noqa: E402
from dataaware.errors import DataError  # noqa: E402
from dataaware.shards import assign_shards, read_shard_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--readers",
        type=int,
        default=0,
        help="ranks x workers, to show how shards would be divided among readers",
    )
    args = parser.parse_args(argv)

    problems = 0
    for directory in args.shard_dirs:
        index_path = (
            directory if directory.is_file() else directory / "shard_index.json"
        )
        try:
            index = read_shard_index(index_path)
        except DataError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

        records = index["shards"]
        counts = [record["samples"] for record in records]
        byte_sizes = [record["bytes"] for record in records]

        print(f"# {directory}")
        print(f"NUM_SHARDS={len(records)}")
        print(f"TOTAL_SAMPLES={index['total_samples']}")
        print(f"MIN_SAMPLES_PER_SHARD={min(counts)}")
        print(f"MAX_SAMPLES_PER_SHARD={max(counts)}")
        print(f"SAMPLE_COUNT_RATIO={max(counts) / min(counts):.4g}")
        print(f"SHARD_BYTES_CV={coefficient_of_variation(byte_sizes):.4g}")
        print(f"BALANCE_BY={index['plan'].get('balance_by')}")
        print(f"IMBALANCE_FACTOR={index['plan'].get('imbalance_factor', 1.0)}")

        if args.readers > 0:
            names = [record["shard"] for record in records]
            size_by_name = {r["shard"]: r["samples"] for r in records}
            per_reader = [assign_shards(names, i, args.readers) for i in range(args.readers)]
            loads = [sum(size_by_name[name] for name in group) for group in per_reader]
            shard_counts = {len(group) for group in per_reader}
            idle = [i for i, group in enumerate(per_reader) if not group]

            print(f"READERS={args.readers}")
            print(f"SHARDS_PER_READER={sorted(shard_counts)}")
            print(f"MIN_READER_SAMPLES={min(loads)}")
            print(f"MAX_READER_SAMPLES={max(loads)}")
            print(f"READER_LOAD_SPREAD={spread(loads):.4g}")
            print(f"IDLE_READERS={idle}")

            if idle:
                problems += 1
                print(
                    f"! {len(idle)} of {args.readers} readers would get no shards at "
                    "all. Rebuild with at least as many shards as readers."
                )
            elif len(shard_counts) > 1:
                print(
                    "! Readers would receive different numbers of shards, so they will "
                    "finish at different times even if every shard is the same size. "
                    "Shard count should be a multiple of the reader count."
                )
            elif spread(loads) > 0.2:
                print(
                    f"! Readers would receive equal shard counts but "
                    f"{spread(loads):.0%} different amounts of data. With synchronised "
                    "ranks the slowest sets the pace; consider --balance-by work."
                )
        print()

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
