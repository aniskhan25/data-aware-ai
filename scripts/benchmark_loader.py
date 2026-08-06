#!/usr/bin/env python3
"""Measure application-level data-loading throughput for one configuration.

    python scripts/benchmark_loader.py --config configs/test/tiny.yaml

Writes ``run_summary.json`` into the configured output directory and prints the
key metrics as ``KEY=VALUE`` lines so that job logs stay greppable.

Overrides let one configuration drive a ladder of runs without duplicating files:

    python scripts/benchmark_loader.py --config configs/test/tiny.yaml \\
        --set loader.num_workers=2 --set run.name=tiny-2workers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataaware.config import load_config  # noqa: E402
from dataaware.errors import ConfigError  # noqa: E402
from dataaware.schema import format_keyvalue, write_run_summary  # noqa: E402


def parse_override(text: str) -> tuple[str, object]:
    """Parse ``section.option=value``, inferring the value type from YAML rules."""
    import yaml

    key, separator, raw = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            f"--set expects section.option=value, got {text!r}"
        )
    return key.strip(), yaml.safe_load(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.OPTION=VALUE",
        help="override one configuration value; may be repeated",
    )
    parser.add_argument(
        "--summary-name",
        default="run_summary.json",
        help="file name written inside the output directory",
    )
    args = parser.parse_args(argv)

    try:
        overrides = dict(parse_override(text) for text in args.overrides)
        config = load_config(args.config, overrides=overrides)
    except (ConfigError, argparse.ArgumentTypeError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    # Imported here so that a configuration error is reported without paying the
    # cost of importing PyTorch, and so this script's --help works without it.
    try:
        from dataaware.loaders import run_loader_benchmark
    except ImportError as exc:
        print(
            f"ERROR PyTorch is required to run the loader benchmark: {exc}\n"
            "Install it with: pip install '.[loader]'",
            file=sys.stderr,
        )
        return 2

    print(f"CONFIG_PATH={config.source_path}")
    print(f"CONFIG_HASH={config.config_hash()}")
    print(f"DATASET_ROOT={config.dataset_root}")
    print(f"MANIFEST_PATH={config.manifest_path}")
    print(f"OUTPUT_DIRECTORY={config.output_directory}", flush=True)

    try:
        summary = run_loader_benchmark(config, repo_root=REPO_ROOT)
    except NotImplementedError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 3

    summary_path = write_run_summary(
        config.output_directory / args.summary_name, summary
    )
    print(f"RUN_SUMMARY={summary_path}")
    print(format_keyvalue(summary))
    if summary["failed_samples"]:
        print(
            f"WARNING {summary['failed_samples']} sample(s) failed to load; "
            "this run is not a valid basis for comparison",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
