#!/usr/bin/env python3
"""Read a worker ladder and recommend a worker count.

    python scripts/compare_workers.py "$TUTORIAL_ROOT"/outputs/workers-*/run_summary.json

Groups the summaries by ``num_workers``, aggregates repeats on the median, then
classifies the ladder as still-improving, plateau, regression, or flat, and names the
limiting resource.

The recommendation is the *cheapest* worker count within 5 % of the best measured
throughput, not the fastest one: each worker is a process holding memory, and buying
2 % with four times the memory is a bad trade on a shared node.

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
from dataaware.workers import analyse, format_table  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/worker-comparison/summary.json"),
        help="where to write the analysis JSON (default: %(default)s)",
    )
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="analyse anyway when runs read different data; the result is not evidence",
    )
    args = parser.parse_args(argv)

    loaded = []
    for path in args.summaries:
        try:
            loaded.append(read_run_summary(path))
        except SummaryError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

    # A ladder deliberately varies num_workers, so that difference is expected here
    # and is not reported as uncontrolled. Reading different *data* still is not.
    fatal = blocking(find_mismatches(loaded))
    if fatal and not args.allow_incompatible:
        print("ERROR these runs are not comparable:", file=sys.stderr)
        for mismatch in fatal:
            print(f"  {mismatch.describe()}", file=sys.stderr)
        print(
            "\nA worker ladder must read one dataset. Rebuild from a single manifest, "
            "or pass --allow-incompatible knowing the result is not evidence.",
            file=sys.stderr,
        )
        return 3

    try:
        analysis = analyse(loaded)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if analysis["cautions"]:
        print("--- read this before the numbers ---")
        for caution in analysis["cautions"]:
            print(f"! {caution}")
        print()

    print(format_table(analysis))

    if not args.no_report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        print(f"\nWORKER_COMPARISON={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
