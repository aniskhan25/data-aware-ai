#!/usr/bin/env python3
"""Compare storage placements, including the cost of getting the data there.

    python3 scripts/compare_storage.py \
        "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json

Reports throughput per placement, then the arithmetic that actually decides the
question: per-epoch time saved, the one-off setup cost, break-even epochs, and total
cost at several epoch counts.

A placement that reads faster can still lose. Staging has to be paid for before the
first sample, so a one-pass workload frequently never recovers it.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.compare import blocking, find_mismatches  # noqa: E402
from dataaware.schema import SummaryError, read_run_summary  # noqa: E402
from dataaware.storage import DEFAULT_HORIZONS, compare, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--baseline",
        default="scratch",
        help="placement to compare against (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="epoch counts to tabulate total cost at (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/storage-comparison/summary.json"),
    )
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--allow-incompatible", action="store_true")
    args = parser.parse_args(argv)

    loaded = []
    for path in args.summaries:
        try:
            loaded.append(read_run_summary(path))
        except SummaryError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

    fatal = blocking(find_mismatches(loaded))
    if fatal and not args.allow_incompatible:
        print("ERROR these runs are not comparable:", file=sys.stderr)
        for mismatch in fatal:
            print(f"  {mismatch.describe()}", file=sys.stderr)
        print(
            "\nA placement comparison must move the same dataset. Rebuild from one "
            "manifest, or pass --allow-incompatible knowing the result is not evidence.",
            file=sys.stderr,
        )
        return 3

    if any(epochs < 1 for epochs in args.epochs):
        parser.error("--epochs values must be >= 1")

    try:
        report = compare(loaded, baseline=args.baseline, horizons=args.epochs)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if report["cautions"]:
        print("--- read this before the numbers ---")
        for caution in report["cautions"]:
            print(f"! {caution}")
        print()

    print(format_report(report))

    if not args.no_report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nSTORAGE_COMPARISON={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
