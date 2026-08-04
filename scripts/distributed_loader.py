#!/usr/bin/env python3
"""Validate that many ranks read unique, balanced data.

Launched one process per rank, normally under srun:

    srun --ntasks=8 --cpus-per-task=7 \\
        python3 scripts/distributed_loader.py --config configs/distributed/healthy.yaml

Each rank measures its own share and writes its own summary. Rank 0 then gathers every
rank's report and writes the verdict: unique samples, duplicates, missing samples, idle
readers, and the spread between the fastest and slowest rank.

The question here is correctness, not speed. Aggregate throughput means nothing until
sample assignment is known to be right — eight ranks reading the same data look fast
and accomplish an eighth of the work.

Exit codes: 0 valid partitioning, 4 a correctness problem was found, 2 a setup error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataaware import distributed  # noqa: E402
from dataaware.config import ConfigError, load_config  # noqa: E402
from dataaware.manifest import read_manifest  # noqa: E402
from dataaware.schema import write_run_summary  # noqa: E402


def parse_override(text: str) -> tuple[str, object]:
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
        "--set", dest="overrides", action="append", default=[],
        metavar="SECTION.OPTION=VALUE",
    )
    parser.add_argument(
        "--verdict-name",
        default="distributed_verdict.json",
        help="file rank 0 writes the aggregate verdict to",
    )
    args = parser.parse_args(argv)

    try:
        overrides = dict(parse_override(text) for text in args.overrides)
        config = load_config(args.config, overrides=overrides)
    except (ConfigError, argparse.ArgumentTypeError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    rank, world_size = distributed.rank_and_world_size()
    if rank == 0:
        print(f"CONFIG_PATH={config.source_path}")
        print(f"CONFIG_HASH={config.config_hash()}")
        print(f"WORLD_SIZE={world_size}")
        print(f"PARTITION_BY_RANK={config.distributed.partition_by_rank}")
        print(f"BACKEND={config.distributed.backend}", flush=True)

    try:
        from dataaware.loaders import run_loader_benchmark
    except ImportError as exc:
        print(f"ERROR PyTorch is required: {exc}", file=sys.stderr)
        return 2

    distributed.init_process_group(config.distributed.backend)
    try:
        indices: list[int] = []
        summary = run_loader_benchmark(
            config,
            repo_root=REPO_ROOT,
            rank=rank,
            world_size=world_size,
            collect_indices=indices,
        )
        # Per-rank summaries are kept, not just the aggregate. A verdict that says
        # "imbalanced" is only actionable if you can see which rank was slow.
        write_run_summary(
            config.output_directory / f"rank_{rank:03d}_summary.json", summary
        )
        print(
            f"RANK={rank} SAMPLES={summary['samples_measured']} "
            f"UNIQUE={summary['unique_samples']} "
            f"SAMPLES_PER_SECOND={summary['samples_per_second']:.4g} "
            f"ELAPSED={summary['measured_seconds']:.4g}",
            flush=True,
        )

        report = distributed.RankReport(
            rank=rank,
            samples_observed=summary["samples_measured"],
            unique_indices=indices,
            samples_per_second=summary["samples_per_second"],
            elapsed_seconds=summary["measured_seconds"],
            data_wait_fraction=summary["mean_data_wait_fraction"],
            duplicate_samples_within_rank=summary["duplicate_samples"],
            batches_measured=summary.get("batches_measured", 0),
            shard_opens=summary.get("shard_opens", 0),
        )
        reports = distributed.gather_reports(report, world_size)
    finally:
        distributed.shutdown()

    if rank != 0:
        return 0

    total_samples = len(read_manifest(config.manifest_path))
    verdict = distributed.aggregate(reports, total_samples=total_samples)
    findings = distributed.diagnose(verdict)

    verdict_path = config.output_directory / args.verdict_name
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(
        json.dumps({**verdict, "findings": findings}, indent=2, sort_keys=True) + "\n"
    )

    print()
    for key in (
        "world_size",
        "total_samples",
        "samples_measured",
        "unique_samples",
        "duplicate_samples",
        "missing_samples",
        "coverage_fraction",
        "total_samples_per_second",
        "min_rank_throughput",
        "max_rank_throughput",
        "rank_throughput_spread",
        "min_rank_elapsed_seconds",
        "max_rank_elapsed_seconds",
        "rank_elapsed_spread",
        "mean_data_wait_fraction",
        "max_data_wait_fraction",
    ):
        value = verdict[key]
        print(f"{key.upper()}={value:.4g}" if isinstance(value, float) else f"{key.upper()}={value}")
    print(f"IDLE_RANKS={verdict['idle_ranks']}")
    print(f"PARTITIONING_VALID={str(verdict['partitioning_valid']).lower()}")
    print(f"VERDICT_PATH={verdict_path}")

    print("\n--- findings ---")
    for finding in findings:
        print(f"* {finding}")
    print(f"\n{verdict['notes']}")

    if not verdict["partitioning_valid"] and config.distributed.validate_unique_samples:
        print(
            "\nERROR partitioning is not valid; see the findings above.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
