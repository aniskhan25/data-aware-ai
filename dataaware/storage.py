"""Comparing storage placements, including what it cost to get there.

The rule this module exists to enforce:

    Faster steady-state reads do not automatically mean a faster end-to-end job.

Staging a dataset to node-local storage has to be paid for before the first sample is
read. A comparison that reports only throughput will recommend staging for workloads
that never recover the copy, so break-even is computed and reported alongside.

Needs no PyTorch: it reads run summaries.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import metrics

#: Epoch counts at which total cost is tabulated. A one-pass workload and a
#: hundred-epoch campaign reach opposite conclusions from the same measurements.
DEFAULT_HORIZONS = (1, 3, 10, 50)


def placement_rows(summaries: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One row per storage placement, aggregating repeats on the median."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary.get("storage", "unknown")), []).append(summary)

    rows = {}
    for name, group in grouped.items():
        rows[name] = {
            "runs": len(group),
            "samples_per_second": _median(group, "samples_per_second"),
            "samples_per_second_cv": _cv(group, "samples_per_second"),
            "estimated_epoch_seconds": _median(group, "estimated_epoch_seconds"),
            "staging_seconds": _median(group, "staging_seconds"),
            "validation_seconds": _median(group, "validation_seconds"),
            "startup_seconds": _median(group, "startup_seconds"),
            "measured_seconds": _median(group, "measured_seconds"),
            "total_job_seconds": _median(group, "total_job_seconds"),
            "mean_batch_wait_seconds": _median(group, "mean_batch_wait_seconds"),
            "peak_tmp_bytes": _median(group, "peak_tmp_bytes"),
            "peak_memory_bytes": _median(group, "peak_memory_bytes"),
            "dataset_fraction_of_memory": _median(group, "dataset_fraction_of_memory"),
            "staged_bytes": _median(group, "staged_bytes"),
            "correctness": {
                "failed_samples": sum(int(s.get("failed_samples", 0)) for s in group),
                "duplicate_samples": sum(int(s.get("duplicate_samples", 0)) for s in group),
                "missing_samples": sum(int(s.get("missing_samples", 0)) for s in group),
            },
        }
    return rows


def setup_cost(row: dict[str, Any]) -> float:
    """One-off cost paid before any epoch: staging plus validating the copy.

    Validation is included deliberately. It is work the job does because it staged,
    and leaving it out would flatter staging.
    """
    return row["staging_seconds"] + row["validation_seconds"]


def break_even(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare one placement against the baseline, in full-cost terms."""
    saved = baseline["estimated_epoch_seconds"] - candidate["estimated_epoch_seconds"]
    cost = setup_cost(candidate) - setup_cost(baseline)
    epochs = metrics.break_even_epochs(cost, saved) if cost > 0 else (0.0 if saved > 0 else None)

    return {
        "per_epoch_time_saved": saved,
        "setup_cost_seconds": cost,
        "break_even_epochs": epochs,
        "throughput_change_percent": (
            (candidate["samples_per_second"] - baseline["samples_per_second"])
            / baseline["samples_per_second"]
            * 100.0
            if baseline["samples_per_second"]
            else None
        ),
    }


def total_cost(row: dict[str, Any], epochs: int) -> float:
    """Wall-clock cost of running ``epochs`` epochs from this placement."""
    return setup_cost(row) + row["estimated_epoch_seconds"] * epochs


def compare(
    summaries: Sequence[dict[str, Any]],
    baseline: str = "scratch",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Build a storage-placement report with break-even arithmetic."""
    if not summaries:
        raise ValueError("no run summaries to compare")

    rows = placement_rows(summaries)
    if baseline not in rows:
        # Project scratch is the documented default for active job I/O, so it is the
        # natural reference. Fall back rather than refuse: a flash-versus-tmp
        # comparison is still meaningful.
        baseline = next(iter(rows))

    comparisons = {
        name: break_even(rows[baseline], row)
        for name, row in rows.items()
        if name != baseline
    }
    totals = {
        name: {str(epochs): total_cost(row, epochs) for epochs in horizons}
        for name, row in rows.items()
    }
    cheapest = {
        str(epochs): min(rows, key=lambda name: total_cost(rows[name], epochs))
        for epochs in horizons
    }

    return {
        "baseline": baseline,
        "placements": rows,
        "comparisons": comparisons,
        "total_cost_seconds": totals,
        "cheapest_at_epochs": cheapest,
        "horizons": list(horizons),
        "cautions": _cautions(rows),
    }


def _cautions(rows: dict[str, dict[str, Any]]) -> list[str]:
    cautions: list[str] = []

    for name, row in rows.items():
        correctness = row["correctness"]
        for field, message in (
            ("failed_samples", "sample(s) failed to load; this run did not read its data"),
            ("duplicate_samples", "duplicate sample read(s) within an epoch"),
            ("missing_samples", "sample(s) never read in a complete epoch"),
        ):
            if correctness[field]:
                cautions.append(f"{name}: {correctness[field]} {message}.")

        if row["runs"] < 2:
            continue
        if row["samples_per_second_cv"] > 0.1:
            cautions.append(
                f"{name}: throughput varied by more than 10 % between repeats, so "
                "small differences here are noise."
            )

    if any(row["runs"] < 2 for row in rows.values()):
        cautions.append(
            "Some placements were measured once, so run-to-run variation is unknown."
        )

    staged = {
        name: row
        for name, row in rows.items()
        if row["dataset_fraction_of_memory"] and row["dataset_fraction_of_memory"] > 0
    }
    for name, row in staged.items():
        cautions.append(
            f"{name}: the staged dataset occupied "
            f"{row['dataset_fraction_of_memory']:.0%} of the job's memory allocation. "
            "Node-local /tmp is memory, so that share was unavailable to the workload."
        )
    return cautions


def format_report(report: dict[str, Any]) -> str:
    """Render placements, break-even, and total cost at several horizons."""
    lines = []
    columns = (
        ("Placement", 11),
        ("Runs", 4),
        ("Samples/s", 9),
        ("Epoch s", 9),
        ("Staging s", 9),
        ("Validate s", 10),
        ("Job s", 8),
    )
    lines.append("| " + " | ".join(f"{t:<{w}}" for t, w in columns) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for _, w in columns) + "|")

    for name, row in report["placements"].items():
        marker = " *" if name == report["baseline"] else ""
        lines.append(
            f"| {name + marker:<11} | {row['runs']:<4} | "
            f"{row['samples_per_second']:<9.4g} | "
            f"{row['estimated_epoch_seconds']:<9.4g} | "
            f"{row['staging_seconds']:<9.4g} | "
            f"{row['validation_seconds']:<10.4g} | "
            f"{row['total_job_seconds']:<8.4g} |"
        )
    lines.append(f"\n* baseline: {report['baseline']}")

    for name, comparison in report["comparisons"].items():
        lines.append(f"\n--- {name} against {report['baseline']} ---")
        change = comparison["throughput_change_percent"]
        lines.append(
            f"THROUGHPUT_CHANGE_PERCENT={change:+.4g}"
            if change is not None
            else "THROUGHPUT_CHANGE_PERCENT=undefined"
        )
        lines.append(f"PER_EPOCH_TIME_SAVED={comparison['per_epoch_time_saved']:.4g}")
        lines.append(f"SETUP_COST_SECONDS={comparison['setup_cost_seconds']:.4g}")
        epochs = comparison["break_even_epochs"]
        if epochs is None:
            lines.append("BREAK_EVEN_EPOCHS=never")
            lines.append(
                "  This placement saves no per-epoch time, so its setup cost is never "
                "recovered however many epochs you run."
            )
        else:
            lines.append(f"BREAK_EVEN_EPOCHS={epochs:.4g}")

    lines.append("\n--- total cost, setup included ---")
    header = "| Epochs | " + " | ".join(
        f"{name:<11}" for name in report["placements"]
    ) + " | Cheapest    |"
    lines.append(header)
    lines.append("|" + "|".join("-" * (len(part) + 2) for part in
                                ["Epochs"] + list(report["placements"]) + ["Cheapest   "]) + "|")
    for epochs in report["horizons"]:
        cells = " | ".join(
            f"{report['total_cost_seconds'][name][str(epochs)]:<11.4g}"
            for name in report["placements"]
        )
        lines.append(
            f"| {epochs:<6} | {cells} | {report['cheapest_at_epochs'][str(epochs)]:<11} |"
        )

    lines.append(
        "\nFaster steady-state reads do not automatically mean a faster end-to-end "
        "job. Read the row matching the number of epochs you actually intend to run."
    )
    return "\n".join(lines)


def _median(group: Sequence[dict[str, Any]], field: str) -> float:
    return metrics.percentile([_number(s.get(field)) for s in group], 50.0)


def _cv(group: Sequence[dict[str, Any]], field: str) -> float:
    return metrics.coefficient_of_variation([_number(s.get(field)) for s in group])


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
