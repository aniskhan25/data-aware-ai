#!/usr/bin/env python3
"""Print a stored run summary, validating it against the schema first.

    python scripts/summarize_run.py outputs/loose-files/run_summary.json
    python scripts/summarize_run.py outputs/*/run_summary.json --table

Useful for reading a result out of a finished job without re-running anything,
and for confirming that a summary written by a job is actually schema-valid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.schema import (  # noqa: E402
    SummaryError,
    format_keyvalue,
    read_run_summary,
)

_TABLE_COLUMNS = (
    ("run_name", "Run", 28),
    ("layout", "Layout", 12),
    ("num_workers", "Workers", 7),
    ("samples_per_second", "Samples/s", 10),
    ("mib_per_second", "MiB/s", 9),
    ("mean_batch_wait_seconds", "Mean wait", 10),
    ("p95_batch_wait_seconds", "P95 wait", 9),
    ("mean_data_wait_fraction", "Wait frac", 9),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--table",
        action="store_true",
        help="one row per summary instead of KEY=VALUE blocks",
    )
    args = parser.parse_args(argv)

    loaded = []
    failures = 0
    for path in args.summaries:
        try:
            loaded.append((path, read_run_summary(path)))
        except SummaryError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            failures += 1

    if args.table and loaded:
        _print_table(loaded)
    else:
        for path, summary in loaded:
            print(f"# {path}")
            print(format_keyvalue(summary))
            print()

    return 1 if failures else 0


def _print_table(loaded: list[tuple[Path, dict]]) -> None:
    header = " ".join(f"{title:<{width}}" for _, title, width in _TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for _, summary in loaded:
        cells = []
        for key, _, width in _TABLE_COLUMNS:
            value = summary.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            cells.append(f"{str(value):<{width}}")
        print(" ".join(cells))
    print(
        "\nNote: rows are comparable only when the manifest hash, batch size, and "
        "measurement length match. Use the comparison tools from Part III onwards, "
        "which check this."
    )


if __name__ == "__main__":
    raise SystemExit(main())
