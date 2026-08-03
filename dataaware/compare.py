"""Comparing runs, and refusing to compare runs that are not comparable.

A comparison is only evidence if the things not under test were held fixed. This
module distinguishes two kinds of mismatch, because they are not equally serious:

**Blocking** — the runs did not read the same data, or do not speak the same schema.
No table of numbers from these is meaningful, so the comparison stops.

**Uncontrolled** — the runs read the same data but differed in something that
affects the result: batch size, worker count, measurement length, seed, compute
step. The numbers still mean something individually, so they are shown, loudly
labelled as not a controlled comparison.

The alternative — printing a tidy table whatever went in — is how a configuration
mistake becomes a published conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from . import metrics

#: Differences that make a comparison meaningless.
BLOCKING_FIELDS = (
    "manifest_hash",
    "schema_version",
)

#: Differences that leave a comparison uncontrolled but still informative.
CONTROLLED_FIELDS = (
    "batch_size",
    "num_workers",
    "measured_batches",
    "world_size",
    "seed",
    "compute_steps",
)

#: Fields whose meaning depends on the layout, so they are only required to match
#: *within* a group.
#:
#: ``shuffle`` is the case that matters: a map-style dataset shuffles an index, and a
#: streaming layout cannot — it shuffles shard order and a buffer instead. Demanding
#: the same flag across layouts would mark every layout comparison in this tutorial
#: uncontrolled, which would teach readers to ignore the warning. Within one layout
#: it is still a real inconsistency.
WITHIN_GROUP_FIELDS = ("shuffle", "shuffle_buffer")


@dataclass(frozen=True)
class Mismatch:
    field: str
    values: tuple[Any, ...]
    blocking: bool

    def describe(self) -> str:
        shown = ", ".join(repr(value) for value in self.values)
        kind = "BLOCKING" if self.blocking else "UNCONTROLLED"
        return f"{kind} {self.field} differs across runs: {shown}"


def find_mismatches(summaries: Sequence[dict[str, Any]]) -> list[Mismatch]:
    """Report every field that differs across the given summaries."""
    mismatches: list[Mismatch] = []
    if len(summaries) < 2:
        return mismatches

    for field, blocking in [(name, True) for name in BLOCKING_FIELDS] + [
        (name, False) for name in CONTROLLED_FIELDS
    ]:
        values = tuple(dict.fromkeys(summary.get(field) for summary in summaries))
        if len(values) > 1:
            mismatches.append(Mismatch(field=field, values=values, blocking=blocking))
    return mismatches


def blocking(mismatches: Iterable[Mismatch]) -> list[Mismatch]:
    return [mismatch for mismatch in mismatches if mismatch.blocking]


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    """Summarise repeated measurements of one quantity.

    The median leads rather than the mean: on a shared filesystem a single slow run
    skews a mean, and the spread is reported alongside so a difference smaller than
    the noise is visible as such.
    """
    if not values:
        return {"runs": 0, "median": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}
    return {
        "runs": len(values),
        "median": metrics.percentile(values, 50.0),
        "min": min(values),
        "max": max(values),
        "cv": metrics.coefficient_of_variation(values),
    }


#: Metrics aggregated for every group in a comparison.
COMPARED_METRICS = (
    "samples_per_second",
    "mib_per_second",
    "mean_batch_wait_seconds",
    "p95_batch_wait_seconds",
    "mean_data_wait_fraction",
    "startup_seconds",
    "files_opened",
    "filesystem_objects",
)


def group_summaries(
    summaries: Sequence[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    """Group summaries by a field, preserving first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        groups.setdefault(str(summary.get(key, "unknown")), []).append(summary)
    return groups


def compare(
    summaries: Sequence[dict[str, Any]],
    key: str = "layout",
    baseline: str | None = None,
) -> dict[str, Any]:
    """Build a comparison report.

    The first group is the baseline unless one is named. Changes are expressed
    relative to it, because "40 % faster than loose files" is the claim a reader
    can act on.
    """
    if not summaries:
        raise ValueError("no run summaries to compare")

    groups = group_summaries(summaries, key)
    mismatches = find_mismatches(summaries)

    baseline_name = baseline or next(iter(groups))
    if baseline_name not in groups:
        raise ValueError(
            f"baseline {baseline_name!r} is not among the compared groups "
            f"{sorted(groups)}"
        )

    rows = {}
    for name, group in groups.items():
        row = {
            metric: aggregate([_number(s.get(metric)) for s in group])
            for metric in COMPARED_METRICS
        }
        row["runs"] = len(group)
        row["run_names"] = [s.get("run_name", "") for s in group]
        row["correctness"] = {
            "failed_samples": sum(int(s.get("failed_samples", 0)) for s in group),
            "duplicate_samples": sum(int(s.get("duplicate_samples", 0)) for s in group),
            "missing_samples": sum(int(s.get("missing_samples", 0)) for s in group),
        }
        rows[name] = row

    report = {
        "key": key,
        "baseline": baseline_name,
        "groups": rows,
        "changes": {
            name: _changes(rows[baseline_name], row)
            for name, row in rows.items()
            if name != baseline_name
        },
        "mismatches": [
            {"field": m.field, "values": [str(v) for v in m.values], "blocking": m.blocking}
            for m in mismatches
        ],
        "controlled": not mismatches,
        "comparable": not blocking(mismatches),
        "cautions": _cautions(rows, mismatches) + _within_group_cautions(groups),
        "notes": _layout_notes(groups),
    }
    return report


def _within_group_cautions(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Flag layout-specific settings that disagree between runs of one layout."""
    cautions = []
    for name, group in groups.items():
        for field in WITHIN_GROUP_FIELDS:
            values = tuple(dict.fromkeys(summary.get(field) for summary in group))
            if len(values) > 1:
                shown = ", ".join(repr(value) for value in values)
                cautions.append(
                    f"UNCONTROLLED {name}: {field} differs between runs of the same "
                    f"group: {shown}"
                )
    return cautions


def _layout_notes(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Expected, inherent differences worth stating so they are not read as faults."""
    notes = []
    shuffles = {
        name: {summary.get("shuffle") for summary in group}
        for name, group in groups.items()
    }
    if len({frozenset(values) for values in shuffles.values()}) > 1:
        notes.append(
            "Shuffle settings differ across groups. This is expected when comparing a "
            "map-style layout against a streaming one: only the former can shuffle an "
            "index. Sample order is therefore not identical across these runs, which "
            "is a property of the layouts rather than a configuration mistake."
        )
    return notes


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _changes(baseline_row: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Percentage change against the baseline, on median values."""
    changes = {}
    for metric in COMPARED_METRICS:
        before = baseline_row[metric]["median"]
        after = row[metric]["median"]
        changes[metric] = {
            "baseline": before,
            "value": after,
            "percent": ((after - before) / before * 100.0) if before else None,
        }
    return changes


def _cautions(
    rows: dict[str, dict[str, Any]], mismatches: Sequence[Mismatch]
) -> list[str]:
    """Reasons not to trust this comparison, stated before the numbers are read."""
    cautions: list[str] = []

    for name, row in rows.items():
        correctness = row["correctness"]
        if correctness["failed_samples"]:
            cautions.append(
                f"{name}: {correctness['failed_samples']} sample(s) failed to load. "
                "This run did not read its data; its throughput is not a result."
            )
        if correctness["duplicate_samples"]:
            cautions.append(
                f"{name}: {correctness['duplicate_samples']} duplicate sample read(s) "
                "within an epoch. Throughput inflated by redundant work is not "
                "throughput."
            )
        if correctness["missing_samples"]:
            cautions.append(
                f"{name}: {correctness['missing_samples']} sample(s) were never read "
                "in a complete epoch. This layout is not covering the dataset."
            )

    single_run = [name for name, row in rows.items() if row["runs"] < 2]
    if single_run:
        cautions.append(
            f"Single run per group ({', '.join(sorted(single_run))}), so run-to-run "
            "variation is unknown. Repeat before acting on a small difference."
        )
    else:
        noisy = [
            name
            for name, row in rows.items()
            if row["samples_per_second"]["cv"] > 0.1
        ]
        if noisy:
            cautions.append(
                f"Throughput varied by more than 10 % between repeats for "
                f"{', '.join(sorted(noisy))}. Treat differences of that size as noise."
            )

    for mismatch in mismatches:
        cautions.append(mismatch.describe())
    return cautions


def format_table(report: dict[str, Any]) -> str:
    """Render the comparison as a Markdown table plus headline changes."""
    key = report["key"]
    columns = (
        (key.capitalize(), 14),
        ("Runs", 4),
        ("Samples/s", 9),
        ("MiB/s", 5),
        ("Mean wait", 9),
        ("P95 wait", 8),
        ("Wait frac", 9),
        ("Opens", 5),
        ("FS objects", 10),
    )
    header = "| " + " | ".join(f"{title:<{width}}" for title, width in columns) + " |"
    divider = "|" + "|".join("-" * (width + 2) for _, width in columns) + "|"
    lines = [header, divider]

    for name, row in report["groups"].items():
        marker = " *" if name == report["baseline"] else ""
        lines.append(
            f"| {name + marker:<14} | {row['runs']:<4} | "
            f"{row['samples_per_second']['median']:<9.4g} | "
            f"{row['mib_per_second']['median']:<5.4g} | "
            f"{row['mean_batch_wait_seconds']['median']:<9.4g} | "
            f"{row['p95_batch_wait_seconds']['median']:<8.4g} | "
            f"{row['mean_data_wait_fraction']['median']:<9.4g} | "
            f"{row['files_opened']['median']:<5.4g} | "
            f"{row['filesystem_objects']['median']:<10.4g} |"
        )
    lines.append(f"\n* baseline: {report['baseline']}")

    for name, changes in report["changes"].items():
        lines.append(f"\n--- {name} against {report['baseline']} ---")
        for metric, label in (
            ("samples_per_second", "THROUGHPUT_CHANGE_PERCENT"),
            ("mean_batch_wait_seconds", "WAIT_CHANGE_PERCENT"),
            ("p95_batch_wait_seconds", "P95_WAIT_CHANGE_PERCENT"),
            ("startup_seconds", "STARTUP_CHANGE_PERCENT"),
        ):
            percent = changes[metric]["percent"]
            lines.append(
                f"{label}={percent:+.4g}" if percent is not None else f"{label}=undefined"
            )
        objects = changes["filesystem_objects"]
        if objects["baseline"] and objects["value"]:
            reduction = objects["baseline"] / objects["value"]
            lines.append(f"FILESYSTEM_OBJECT_REDUCTION={reduction:.4g}x")

    return "\n".join(lines)
