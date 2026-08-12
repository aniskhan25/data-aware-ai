#!/usr/bin/env python3
"""Turn the measurements from Parts I-VI into a data-readiness decision.

    python3 scripts/render_decision.py \
        --inspection  "$TUTORIAL_ROOT"/outputs/inspection/dataset_report.json \
        --layouts     "$TUTORIAL_ROOT"/outputs/layout-comparison/repeated.json \
        --workers     "$TUTORIAL_ROOT"/outputs/worker-comparison/webdataset-1000.json \
        --storage     "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json \
        --distributed "$TUTORIAL_ROOT"/outputs/distributed/healthy/distributed_verdict.json \
        --planned-epochs 3

Writes data_readiness.json and data_readiness.md, and prints the greppable summary.

--planned-epochs is not cosmetic. The same measurements recommend staging for a long
campaign and reject it for a one-pass job, and nothing in the data can tell which you
intend to run.

Readiness is a statement about correctness and completeness, not speed: a pipeline that
reads the wrong data quickly is not ready, and a missing input yields INCONCLUSIVE
rather than a cheerful default.

Exit codes: 0 READY or READY_WITH_CAUTION, 5 NOT_READY, 6 INCONCLUSIVE, 2 a read error.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.decision import decide, format_keyvalue, render_markdown  # noqa: E402

EXIT_CODES = {"READY": 0, "READY_WITH_CAUTION": 0, "NOT_READY": 5, "INCONCLUSIVE": 6}


def load(path: Path | None, label: str) -> dict | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        print(f"WARNING {label} not found at {path}; continuing without it", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"ERROR {label} at {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


#: Where each step writes the result the decision reads, under $TUTORIAL_ROOT.
DEFAULT_INPUTS = {
    "inspection": "outputs/inspection/dataset_report.json",
    "layouts": "outputs/layout-comparison/summary.json",
    "workers": "outputs/worker-comparison/summary.json",
    "storage": "outputs/storage-comparison/summary.json",
    "distributed": "outputs/distributed/healthy/distributed_verdict.json",
}


def _under_root(relative: str) -> Path | None:
    """Resolve against TUTORIAL_ROOT, and only if the file is actually there."""
    root = os.environ.get("TUTORIAL_ROOT")
    if not root:
        return None
    path = Path(root) / relative
    return path if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Each input defaults to where the step that produces it writes. Pass a path
    # explicitly to read a result from somewhere else; pass nothing and the report
    # says which inputs were missing rather than assuming they were fine.
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(
            f"--{name}", type=Path, default=_under_root(default),
            help=f"default: {default}",
        )
    parser.add_argument(
        "--planned-epochs",
        type=int,
        default=3,
        help="epochs the real workload will run (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final"),
        help="where to write data_readiness.json and .md (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.planned_epochs < 1:
        parser.error("--planned-epochs must be >= 1")

    report = decide(
        inspection=load(args.inspection, "inspection report"),
        layouts=load(args.layouts, "layout comparison"),
        workers=load(args.workers, "worker comparison"),
        storage=load(args.storage, "storage comparison"),
        distributed=load(args.distributed, "distributed verdict"),
        planned_epochs=args.planned_epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "data_readiness.json"
    md_path = args.output_dir / "data_readiness.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))

    print(format_keyvalue(report))
    if report["blocking_issues"]:
        print("\n--- blocking issues ---")
        for issue in report["blocking_issues"]:
            print(f"! {issue}")
    if report["cautions"]:
        print("\n--- cautions ---")
        for caution in report["cautions"]:
            print(f"? {caution}")
    print(f"\nREADINESS_JSON={json_path}")
    print(f"READINESS_REPORT={md_path}")
    return EXIT_CODES[report["data_readiness"]]


if __name__ == "__main__":
    raise SystemExit(main())
