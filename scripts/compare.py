#!/usr/bin/env python3
"""Compare finished runs of one kind.

    python3 scripts/compare.py layouts "$TUTORIAL_ROOT"/outputs/*/run_summary.json
    python3 scripts/compare.py workers "$TUTORIAL_ROOT"/outputs/workers/*/run_summary.json
    python3 scripts/compare.py storage "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json

All three kinds do the same thing: read schema-valid run summaries, refuse to
compare runs that did not read the same data, and analyse what is left. Only the
analysis differs. Layouts are ranked against a baseline, worker rungs are read for
a plateau, and placements get break-even arithmetic that includes setup cost.

The refusal is the point. A differing manifest hash means the runs read different
data, and a table built from them would look perfectly reasonable while meaning
nothing.

Exit codes: 0 done, 2 a setup error, 3 the runs are not comparable.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware import storage as storage_module  # noqa: E402
from dataaware import workers as workers_module  # noqa: E402
from dataaware.compare import blocking, compare, find_mismatches, format_table  # noqa: E402
from dataaware.errors import SummaryError  # noqa: E402
from dataaware.schema import read_run_summary  # noqa: E402

#: Per kind: where the report goes, and the name printed with its path.
KINDS = {
    "layouts": ("outputs/layout-comparison/summary.json", "COMPARISON_REPORT"),
    "workers": ("outputs/worker-comparison/summary.json", "WORKER_COMPARISON"),
    "storage": ("outputs/storage-comparison/summary.json", "STORAGE_COMPARISON"),
}


def analyse(kind: str, runs: list[dict], args) -> tuple[dict, str]:
    """Run the analysis for one kind and return ``(report, rendered_table)``."""
    if kind == "layouts":
        report = compare(runs, key=args.key, baseline=args.baseline)
        return report, format_table(report)
    if kind == "workers":
        # A ladder varies num_workers on purpose, so that difference is expected
        # here rather than reported as uncontrolled. Differing data still is not.
        report = workers_module.analyse(runs)
        return report, workers_module.format_table(report)
    report = storage_module.compare(
        runs, baseline=args.baseline or "scratch", horizons=args.epochs
    )
    return report, storage_module.format_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--baseline",
        default=None,
        help="group to compare against (layouts: the first given; storage: scratch)",
    )
    parser.add_argument(
        "--key", default="layout", help="layouts only: summary field to group by"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(storage_module.DEFAULT_HORIZONS),
        help="storage only: epoch counts to tabulate total cost at",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="compare anyway when runs read different data; the result is not evidence",
    )
    args = parser.parse_args(argv)

    default_output, report_name = KINDS[args.kind]
    output = args.output or Path(default_output)

    runs = []
    for path in args.summaries:
        try:
            runs.append(read_run_summary(path))
        except SummaryError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2
    if len(runs) < 2:
        print("ERROR at least two run summaries are needed", file=sys.stderr)
        return 2

    fatal = blocking(find_mismatches(runs))
    if fatal and not args.allow_incompatible:
        print("ERROR these runs are not comparable:", file=sys.stderr)
        for mismatch in fatal:
            print(f"  {mismatch.describe()}", file=sys.stderr)
        print(
            "\nA differing manifest hash means the runs read different data. Rebuild "
            "them from one manifest, or pass --allow-incompatible knowing the result "
            "is not evidence.",
            file=sys.stderr,
        )
        return 3

    try:
        report, table = analyse(args.kind, runs, args)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if report.get("cautions"):
        print("--- read this before the numbers ---")
        for caution in report["cautions"]:
            print(f"! {caution}")
        print()
    if report.get("notes"):
        print("--- expected differences ---")
        for note in report["notes"]:
            print(f"- {note}")
        print()

    print(table)
    if "controlled" in report:
        print(f"\nCONTROLLED_COMPARISON={str(report['controlled']).lower()}")

    if not args.no_report:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{report_name}={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
