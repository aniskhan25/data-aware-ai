"""Reading a worker ladder.

Adding DataLoader workers helps until something else saturates. This module turns a
set of runs at different worker counts into a recommendation, and — more importantly —
into a statement of *why*, because "use 4 workers" is only useful if you know whether
the limit was storage, CPU, or memory.

Needs no PyTorch: it reads run summaries.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import metrics

#: Throughput gain below this counts as no gain. Two runs on a shared filesystem
#: routinely differ by a few percent, so a smaller "improvement" is not evidence of
#: one.
PLATEAU_GAIN = 0.05

#: Throughput loss above this counts as a genuine regression rather than noise.
REGRESSION_LOSS = 0.05

#: CPU utilisation, as a fraction of the allocation, above which the pipeline is
#: considered CPU-bound. Not 1.0: a pipeline pinned near its ceiling is saturated in
#: practice, and the last few percent are rarely reachable.
CPU_SATURATED = 0.85

#: Data-wait fraction above which the workload is still mostly waiting for data.
STILL_WAITING = 0.5


def ladder_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse summaries into one row per worker count, ordered by worker count.

    Repeats at the same worker count are aggregated on the median, so a single slow
    run does not decide the shape of the ladder.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(int(summary.get("num_workers", 0)), []).append(summary)

    rows = []
    for workers in sorted(grouped):
        group = grouped[workers]
        rows.append(
            {
                "num_workers": workers,
                "runs": len(group),
                "samples_per_second": _median(group, "samples_per_second"),
                "samples_per_second_cv": _cv(group, "samples_per_second"),
                "mean_batch_wait_seconds": _median(group, "mean_batch_wait_seconds"),
                "p95_batch_wait_seconds": _median(group, "p95_batch_wait_seconds"),
                "mean_data_wait_fraction": _median(group, "mean_data_wait_fraction"),
                "cpu_utilization": _median(group, "cpu_utilization"),
                "peak_memory_bytes": _median(group, "peak_memory_bytes"),
                "involuntary_switches_per_second": _median(
                    group, "involuntary_switches_per_second"
                ),
                "child_processes": _median(group, "child_processes"),
                "oversubscription_ratio": _median(group, "oversubscription_ratio"),
                "cpus_available": _median(group, "cpus_available"),
                "startup_seconds": _median(group, "startup_seconds"),
            }
        )
    return rows


def analyse(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Recommend a worker count, and say what limited the pipeline.

    The recommendation is the *cheapest* rung within ``PLATEAU_GAIN`` of the best
    measured throughput, not the fastest rung. Workers are not free: each is a
    process holding memory, and buying a 2 % gain with four times the memory is a bad
    trade on a shared node.
    """
    rows = ladder_rows(summaries)
    if not rows:
        raise ValueError("no run summaries to analyse")

    result: dict[str, Any] = {
        "rows": rows,
        "cautions": _cautions(rows, summaries),
        "pattern": "inconclusive",
        "recommended_workers": rows[0]["num_workers"],
        "best_workers": rows[0]["num_workers"],
        "limiting_factor": "unknown",
        "explanation": "",
    }
    if len(rows) < 2:
        result["explanation"] = (
            "Only one worker count was measured, so nothing can be said about the "
            "shape of the ladder. Run at least two rungs."
        )
        return result

    best = max(rows, key=lambda row: row["samples_per_second"])
    result["best_workers"] = best["num_workers"]

    # Cheapest rung that is within PLATEAU_GAIN of the best.
    peak = best["samples_per_second"]
    adequate = [
        row
        for row in rows
        if peak > 0 and row["samples_per_second"] >= peak * (1.0 - PLATEAU_GAIN)
    ]
    recommended = adequate[0] if adequate else best
    result["recommended_workers"] = recommended["num_workers"]

    result["pattern"] = _pattern(rows, best)
    result["limiting_factor"], result["explanation"] = _diagnose(
        rows, best, recommended, result["pattern"]
    )
    return result


def _pattern(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    """Classify the ladder's shape."""
    first, last = rows[0], rows[-1]
    if first["samples_per_second"] <= 0:
        return "inconclusive"

    # Did anything after the peak get materially worse?
    after_peak = [row for row in rows if row["num_workers"] > best["num_workers"]]
    worst_after = min(
        (row["samples_per_second"] for row in after_peak), default=best["samples_per_second"]
    )
    if best["samples_per_second"] > 0 and worst_after < best["samples_per_second"] * (
        1.0 - REGRESSION_LOSS
    ):
        return "regression"

    # Still improving means the *last step* still bought something, not merely that
    # the top rung beats the bottom one — which is true of almost any ladder.
    previous = rows[-2]
    if best["num_workers"] == last["num_workers"] and last[
        "samples_per_second"
    ] > previous["samples_per_second"] * (1.0 + PLATEAU_GAIN):
        return "still-improving"

    if best["samples_per_second"] > first["samples_per_second"] * (1.0 + PLATEAU_GAIN):
        return "plateau"

    return "flat"


def _diagnose(
    rows: list[dict[str, Any]],
    best: dict[str, Any],
    recommended: dict[str, Any],
    pattern: str,
) -> tuple[str, str]:
    """Name the limiting resource and explain the recommendation."""
    cpu = recommended["cpu_utilization"]
    waiting = recommended["mean_data_wait_fraction"]

    if pattern == "still-improving":
        factor = "not-yet-saturated"
        explanation = (
            f"Throughput was still rising at {best['num_workers']} workers, the "
            "highest rung measured. Add a higher rung before concluding anything: the "
            "ladder has not found its limit."
        )
    elif pattern == "regression":
        factor = "oversubscribed"
        worst = min(rows, key=lambda row: row["samples_per_second"])
        explanation = (
            f"Throughput peaked at {best['num_workers']} workers and fell to "
            f"{worst['samples_per_second']:.0f} samples/s at {worst['num_workers']}. "
            "Past the peak the workers compete for the same cores, so each batch "
            "takes longer to assemble. num_workers is not a free knob."
        )
    elif pattern == "flat":
        factor = "not-worker-bound"
        explanation = (
            "Adding workers changed little. The bottleneck is not the number of "
            "loader processes: look at the layout (Part III) or the storage placement "
            "(Part V) instead."
        )
    elif cpu >= CPU_SATURATED:
        factor = "cpu-decode"
        explanation = (
            f"Throughput plateaued at {recommended['num_workers']} workers with CPU "
            f"utilisation at {cpu:.0%} of the allocation. The pipeline is CPU-bound: "
            "more workers cannot help because there are no idle cores left. Reduce "
            "decode cost, or allocate more CPUs per rank."
        )
    elif waiting >= STILL_WAITING:
        factor = "storage-or-synchronisation"
        explanation = (
            f"Throughput plateaued at {recommended['num_workers']} workers while CPU "
            f"utilisation stayed at {cpu:.0%} and the workload still spent "
            f"{waiting:.0%} of its time waiting for data. Idle CPUs with high waits "
            "point at storage or synchronisation, not at the worker count."
        )
    else:
        factor = "balanced"
        explanation = (
            f"Throughput plateaued at {recommended['num_workers']} workers with CPU "
            f"utilisation at {cpu:.0%} and waits low. The input pipeline is keeping "
            "up; adding workers would only add memory."
        )

    if recommended["num_workers"] != best["num_workers"]:
        explanation += (
            f" {recommended['num_workers']} workers is recommended over "
            f"{best['num_workers']} because it reaches within "
            f"{PLATEAU_GAIN:.0%} of the best throughput with fewer processes, and "
            "each worker costs memory."
        )
    return factor, explanation


def _cautions(
    rows: list[dict[str, Any]], summaries: Sequence[dict[str, Any]]
) -> list[str]:
    """Reasons to distrust the ladder, stated before its numbers are read."""
    cautions: list[str] = []

    for name in ("failed_samples", "duplicate_samples", "missing_samples"):
        total = sum(int(summary.get(name, 0)) for summary in summaries)
        if total:
            cautions.append(
                f"{total} {name.replace('_', ' ')} across the ladder. Fix correctness "
                "before tuning throughput."
            )

    if any(row["runs"] < 2 for row in rows):
        cautions.append(
            "Some rungs were measured once, so run-to-run variation is unknown. A "
            f"difference below about {PLATEAU_GAIN:.0%} should not be trusted."
        )
    noisy = [row["num_workers"] for row in rows if row["samples_per_second_cv"] > 0.1]
    if noisy:
        cautions.append(
            f"Throughput varied by more than 10 % between repeats at {noisy} workers. "
            "The shape of the ladder there is not reliable."
        )

    for row in rows:
        actual = row["child_processes"]
        if actual >= 0 and 0 < actual < row["num_workers"]:
            cautions.append(
                f"At {row['num_workers']} workers only {actual:.0f} child process(es) "
                "were seen. The configured worker count may not be what actually ran."
            )

    cpus = max((row["cpus_available"] for row in rows), default=0)
    if cpus:
        over = [row["num_workers"] for row in rows if row["oversubscription_ratio"] > 1.0]
        if over:
            cautions.append(
                f"Rungs {over} request more processes than the {cpus:.0f} allocated "
                "CPUs. Their results describe an oversubscribed pipeline, which is the "
                "point of that rung but not a configuration to adopt."
            )
    return cautions


def format_table(analysis: dict[str, Any]) -> str:
    """Render the ladder as a table plus the recommendation."""
    columns = (
        ("Workers", 7),
        ("Runs", 4),
        ("Samples/s", 9),
        ("Mean wait", 9),
        ("P95 wait", 8),
        ("Wait frac", 9),
        ("CPU util", 8),
        ("Peak MiB", 8),
        ("Invol cs/s", 10),
    )
    header = "| " + " | ".join(f"{title:<{width}}" for title, width in columns) + " |"
    lines = [header, "|" + "|".join("-" * (width + 2) for _, width in columns) + "|"]

    for row in analysis["rows"]:
        marker = " *" if row["num_workers"] == analysis["recommended_workers"] else ""
        lines.append(
            f"| {str(row['num_workers']) + marker:<7} | {row['runs']:<4} | "
            f"{row['samples_per_second']:<9.4g} | "
            f"{row['mean_batch_wait_seconds']:<9.4g} | "
            f"{row['p95_batch_wait_seconds']:<8.4g} | "
            f"{row['mean_data_wait_fraction']:<9.4g} | "
            f"{row['cpu_utilization']:<8.3g} | "
            f"{row['peak_memory_bytes'] / metrics.MIB:<8.4g} | "
            f"{row['involuntary_switches_per_second']:<10.4g} |"
        )

    lines.append("\n* recommended")
    lines.append(f"\nLADDER_PATTERN={analysis['pattern']}")
    lines.append(f"BEST_WORKERS={analysis['best_workers']}")
    lines.append(f"RECOMMENDED_WORKERS={analysis['recommended_workers']}")
    lines.append(f"MAIN_LIMITING_FACTOR={analysis['limiting_factor']}")
    lines.append(f"\n{analysis['explanation']}")
    return "\n".join(lines)


def _median(group: Sequence[dict[str, Any]], field: str) -> float:
    values = [_number(summary.get(field)) for summary in group]
    return metrics.percentile(values, 50.0)


def _cv(group: Sequence[dict[str, Any]], field: str) -> float:
    return metrics.coefficient_of_variation(
        [_number(summary.get(field)) for summary in group]
    )


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
