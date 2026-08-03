#!/usr/bin/env python3
"""Characterise a dataset before allocating any GPUs.

Reads metadata only — it never opens a file — so it is cheap and needs no GPU and
no PyTorch.

    python scripts/inspect_dataset.py --path "$TUTORIAL_ROOT/source"

    # Inside the allocation you intend to use, so staging advice is meaningful
    srun --account=$LUMI_PROJECT --partition=small --mem=32G --time=00:10:00 \\
        python scripts/inspect_dataset.py --path "$TUTORIAL_ROOT/source"

The tool suggests experiments; it does not choose a format. It cannot see whether
your workload needs random access, whether sample order matters, or how expensive
a record is to decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.inspection import (  # noqa: E402
    DEFAULT_SMALL_FILE_BYTES,
    DEFAULT_TARGET_SHARD_BYTES,
    DEFAULT_THRESHOLDS,
    DEFAULT_TMP_SAFETY_FRACTION,
    InspectionError,
    format_keyvalue,
    inspect_path,
)


def parse_byte_list(text: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of byte counts, got {text!r}"
        ) from None
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("byte thresholds must all be >= 1")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", type=Path, required=True, help="dataset to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inspection/dataset_report.json"),
        help="where to write the JSON report (default: %(default)s)",
    )
    parser.add_argument(
        "--small-file-bytes",
        type=int,
        default=DEFAULT_SMALL_FILE_BYTES,
        help="files below this size count as small (default: %(default)s)",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_byte_list,
        default=list(DEFAULT_THRESHOLDS),
        help="comma-separated size histogram thresholds in bytes",
    )
    parser.add_argument(
        "--memory-bytes",
        type=int,
        default=None,
        help="allocated memory, when not running under Slurm; enables staging advice",
    )
    parser.add_argument(
        "--tmp-safety-fraction",
        type=float,
        default=DEFAULT_TMP_SAFETY_FRACTION,
        help=(
            "largest share of allocated memory a staged dataset may occupy "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--target-shard-bytes",
        type=int,
        default=DEFAULT_TARGET_SHARD_BYTES,
        help="target tar shard size used for the shard-count suggestion",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every N files; useful on very large trees",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print the size histogram, extension mix, and candidate reasoning",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="print to the terminal without writing a JSON report",
    )
    args = parser.parse_args(argv)

    try:
        report = inspect_path(
            args.path,
            small_file_bytes=args.small_file_bytes,
            thresholds=args.thresholds,
            memory_bytes=args.memory_bytes,
            tmp_safety_fraction=args.tmp_safety_fraction,
            target_shard_bytes=args.target_shard_bytes,
            progress_every=args.progress_every,
        )
    except InspectionError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    print(format_keyvalue(report))

    if args.verbose:
        _print_details(report)

    if not args.no_report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"REPORT_PATH={args.output}")

    if report["tree"]["total_files"] == 0:
        print(
            f"WARNING no files found under {report['dataset_path']}", file=sys.stderr
        )
    if report["tree"]["unreadable_directories"] or report["tree"]["unreadable_files"]:
        print(
            "WARNING some entries could not be read; the report covers only what "
            "was readable",
            file=sys.stderr,
        )
    return 0


def _print_details(report: dict) -> None:
    print("\n--- file size distribution ---")
    for row in report["size_thresholds"]:
        print(f"under {row['bytes']:>10} bytes: {row['files']:>10} ({row['fraction']:.1%})")

    print("\n--- extensions ---")
    for row in report["extensions"]:
        print(
            f"{row['extension']:<12} {row['files']:>10} files "
            f"{row['bytes']:>14} bytes ({row['fraction']:.1%})"
        )

    if report["format_hints"]:
        print("\n--- format hints (from file names only) ---")
        for hint in report["format_hints"]:
            print(f"{hint['extension']}: {hint['hint']}")

    print("\n--- candidate next experiments ---")
    for index, candidate in enumerate(report["candidates"], start=1):
        print(f"{index}. {candidate['experiment']}\n   {candidate['reason']}")

    print("\n--- what this cannot tell you ---")
    for limitation in report["limitations"]:
        print(f"- {limitation}")


if __name__ == "__main__":
    raise SystemExit(main())
