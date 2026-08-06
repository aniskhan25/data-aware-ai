#!/usr/bin/env python3
"""Compare dataset layouts from their run summaries.

    python scripts/compare_layouts.py \\
        outputs/loose-files/run_summary.json \\
        outputs/squashfs/run_summary.json \\
        outputs/webdataset/run_summary.json

Pass several summaries per layout to get medians and spread instead of single
numbers:

    python scripts/compare_layouts.py outputs/*/run_summary.json

The comparison refuses to proceed when the runs did not read the same data
(different manifest hash, or a different schema version): no table from those runs
would mean anything. Differences that leave the comparison merely *uncontrolled* -
batch size, worker count, seed - are reported loudly and the numbers are still
shown.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.compare import blocking, compare, find_mismatches, format_table  # noqa: E402
from dataaware.errors import SummaryError  # noqa: E402
from dataaware.schema import read_run_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--key",
        default="layout",
        help="summary field to group by (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="group to compare against (default: the first one given)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/layout-comparison/summary.json"),
        help="where to write the comparison JSON (default: %(default)s)",
    )
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="compare anyway when runs read different data; the result is not evidence",
    )
    args = parser.parse_args(argv)

    loaded = []
    for path in args.summaries:
        try:
            loaded.append(read_run_summary(path))
        except SummaryError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2
    if len(loaded) < 2:
        print(
            "ERROR at least two run summaries are needed for a comparison",
            file=sys.stderr,
        )
        return 2

    fatal = blocking(find_mismatches(loaded))
    if fatal and not args.allow_incompatible:
        print("ERROR these runs are not comparable:", file=sys.stderr)
        for mismatch in fatal:
            print(f"  {mismatch.describe()}", file=sys.stderr)
        print(
            "\nA differing manifest hash means the runs read different data. Rebuild "
            "the layouts from one manifest, or pass --allow-incompatible if you "
            "understand that the result is not evidence.",
            file=sys.stderr,
        )
        return 3

    try:
        report = compare(loaded, key=args.key, baseline=args.baseline)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if report["cautions"]:
        print("--- read this before the numbers ---")
        for caution in report["cautions"]:
            print(f"! {caution}")
        print()
    if report["notes"]:
        print("--- expected differences ---")
        for note in report["notes"]:
            print(f"- {note}")
        print()

    print(format_table(report))
    print(f"\nCONTROLLED_COMPARISON={str(report['controlled']).lower()}")

    if not args.no_report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"COMPARISON_REPORT={args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
