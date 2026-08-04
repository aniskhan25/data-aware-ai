#!/usr/bin/env python3
"""Check a distributed run's partitioning from its stored per-rank summaries.

    python3 scripts/validate_partitioning.py \
        "$TUTORIAL_ROOT"/outputs/distributed/healthy/distributed_verdict.json

Re-checks a verdict without re-running anything, and states plainly whether the
partitioning was valid. Use it to compare a healthy run against a broken one, or to
confirm a fix.

Exit codes: 0 valid, 4 a correctness problem, 2 the file could not be read.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.distributed import diagnose  # noqa: E402

REQUIRED = (
    "world_size",
    "total_samples",
    "samples_measured",
    "unique_samples",
    "duplicate_samples",
    "missing_samples",
    "coverage_fraction",
    "rank_throughput_spread",
    "rank_elapsed_spread",
    "min_rank_elapsed_seconds",
    "max_rank_elapsed_seconds",
    "idle_ranks",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("verdicts", nargs="+", type=Path)
    parser.add_argument(
        "--quiet", action="store_true", help="print only the verdict line per file"
    )
    args = parser.parse_args(argv)

    problems = 0
    for path in args.verdicts:
        try:
            verdict = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            return 2

        missing = [key for key in REQUIRED if key not in verdict]
        if missing:
            print(f"ERROR {path} is missing {missing}", file=sys.stderr)
            return 2

        valid = (
            verdict["duplicate_samples"] == 0
            and verdict["missing_samples"] == 0
            and not verdict["idle_ranks"]
        )
        print(f"# {path}")
        print(f"WORLD_SIZE={verdict['world_size']}")
        print(f"UNIQUE_SAMPLES={verdict['unique_samples']} of {verdict['total_samples']}")
        print(f"DUPLICATE_SAMPLES={verdict['duplicate_samples']}")
        print(f"MISSING_SAMPLES={verdict['missing_samples']}")
        print(f"IDLE_RANKS={verdict['idle_ranks']}")
        print(f"RANK_ELAPSED_SPREAD={verdict['rank_elapsed_spread']:.4g}")
        print(f"PARTITIONING_VALID={str(valid).lower()}")

        if not args.quiet:
            for finding in verdict.get("findings") or diagnose(verdict):
                print(f"* {finding}")
        print()
        if not valid:
            problems += 1

    return 4 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
