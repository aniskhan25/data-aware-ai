"""Turning the measurements into a defensible recommendation.

Two rules shape this module, and both are constraints on what it may conclude:

* **No single throughput number decides readiness.** Readiness is a statement about
  correctness and completeness. A pipeline that reads the wrong data quickly is not
  ready; one that reads the right data slowly may well be.
* **Absent evidence is not good news.** A missing input produces ``INCONCLUSIVE``, never
  a cheerful default. The most damaging output this tool could produce is ``READY`` on
  the strength of experiments nobody ran.

Every recommendation carries the metric that supports it and the limitation that
qualifies it, because a recommendation without either cannot be argued with.

Needs no PyTorch: it reads the JSON the earlier parts wrote.
"""

from __future__ import annotations

from typing import Any

from . import storage as storage_module

#: Readiness verdicts, worst to best.
READINESS_STATES = ("NOT_READY", "INCONCLUSIVE", "READY_WITH_CAUTION", "READY")

#: Inputs without which no readiness verdict is possible: what to use, and whether it
#: is read correctly by many readers.
REQUIRED_INPUTS = ("layouts", "distributed")

#: Inputs that strengthen a verdict. Their absence is a caution, not a blocker.
ADVISORY_INPUTS = ("inspection", "workers", "storage")

#: Rank elapsed spread above which the slowest rank is materially holding the others up.
RANK_SPREAD_CAUTION = 0.2

#: Data-wait fraction above which the input path is still the dominant cost.
DATA_WAIT_CAUTION = 0.5


def decide(
    inspection: dict[str, Any] | None = None,
    layouts: dict[str, Any] | None = None,
    workers: dict[str, Any] | None = None,
    storage: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
    planned_epochs: int = 3,
) -> dict[str, Any]:
    """Produce a data-readiness verdict from whichever parts were run.

    ``planned_epochs`` is not cosmetic: it decides the staging recommendation outright.
    The same measurements say "stage" for a long campaign and "do not stage" for a
    one-pass job, and nothing in the data can tell which you intend to run.
    """
    if planned_epochs < 1:
        raise ValueError("planned_epochs must be >= 1")

    inputs = {
        "inspection": inspection,
        "layouts": layouts,
        "workers": workers,
        "storage": storage,
        "distributed": distributed,
    }
    provided = {name for name, value in inputs.items() if value}

    blocking: list[str] = []
    cautions: list[str] = []
    limitations: list[str] = []

    absent_required = [name for name in REQUIRED_INPUTS if name not in provided]
    absent_advisory = [name for name in ADVISORY_INPUTS if name not in provided]
    for name in absent_advisory:
        cautions.append(
            f"No {name} results were supplied, so that dimension is unverified. The "
            "recommendation below is weaker than it looks."
        )

    layout = _layout_recommendation(layouts, blocking, cautions, limitations)
    worker = _worker_recommendation(workers, cautions, limitations)
    place = _storage_recommendation(storage, planned_epochs, cautions, limitations)
    partitioning = _distributed_verdict(distributed, blocking, cautions, limitations)

    readiness = _readiness(blocking, cautions, absent_required)
    return {
        "data_readiness": readiness,
        "planned_epochs": planned_epochs,
        "inputs_provided": sorted(provided),
        "inputs_missing": sorted(set(inputs) - provided),
        "recommended_layout": layout["layout"],
        "layout_evidence": layout["evidence"],
        "recommended_workers_per_rank": worker["workers"],
        "worker_evidence": worker["evidence"],
        "main_limiting_factor": worker["limiting_factor"],
        "recommended_storage": place["storage"],
        "storage_evidence": place["evidence"],
        "node_local_staging": place["staging"],
        "staging_evidence": place["staging_evidence"],
        "distributed_partitioning": partitioning["state"],
        "distributed_evidence": partitioning["evidence"],
        "blocking_issues": blocking,
        "cautions": cautions,
        "limitations": limitations + list(GENERAL_LIMITATIONS),
        "next_experiment": _next_experiment(readiness, blocking),
        "dataset_summary": _dataset_summary(inspection),
    }


#: Stated in every report. A recommendation that hides these invites over-reading.
GENERAL_LIMITATIONS = (
    "These measurements describe one dataset, on one machine, under whatever load the "
    "shared filesystem carried at the time. They are not a benchmark of LUMI storage.",
    "The layout recommendation is based on measured throughput and correctness only. It "
    "cannot account for requirements that were never measured: arbitrary path-based "
    "access, full-dataset shuffling each epoch, or mutable records. Check those against "
    "the table in docs/dataset-layouts.md before committing.",
    "Readiness concerns the input path. It says nothing about whether the model, the "
    "communication pattern, or the GPU utilisation will scale.",
)


def _readiness(
    blocking: list[str], cautions: list[str], absent_required: list[str]
) -> str:
    """Correctness first, then completeness, then caveats.

    Order matters: a correctness failure is reported as NOT_READY even when inputs are
    also missing, because that is the more actionable finding.
    """
    if blocking:
        return "NOT_READY"
    if absent_required:
        return "INCONCLUSIVE"
    if cautions:
        return "READY_WITH_CAUTION"
    return "READY"


def _layout_recommendation(
    layouts: dict[str, Any] | None,
    blocking: list[str],
    cautions: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if not layouts:
        return {"layout": "unknown", "evidence": "No layout comparison was supplied."}

    groups = layouts.get("groups") or {}
    if not groups:
        return {"layout": "unknown", "evidence": "The layout comparison had no groups."}

    if not layouts.get("comparable", True):
        cautions.append(
            "The layout comparison was not comparable - the runs read different data - "
            "so no layout can be recommended from it."
        )
        return {
            "layout": "unknown",
            "evidence": "Layout runs were not comparable; see the comparison's mismatches.",
        }

    # A layout that failed to read its data is not a candidate, however fast it was.
    sound = {}
    for name, row in groups.items():
        correctness = row.get("correctness") or {}
        failures = {
            key: correctness.get(key, 0)
            for key in ("failed_samples", "duplicate_samples", "missing_samples")
            if correctness.get(key, 0)
        }
        if failures:
            blocking.append(
                f"Layout '{name}' had {failures} during measurement. A layout that does "
                "not read its data correctly cannot be recommended."
            )
        else:
            sound[name] = row

    if not sound:
        return {
            "layout": "unknown",
            "evidence": "Every measured layout had correctness failures.",
        }

    best = max(sound, key=lambda name: _median(sound[name], "samples_per_second"))
    throughput = _median(sound[best], "samples_per_second")
    baseline = layouts.get("baseline")
    evidence = f"{best} reached {throughput:.0f} samples/s (median of {sound[best].get('runs', 1)} run(s))"
    if baseline and baseline in groups and baseline != best:
        change = ((layouts.get("changes") or {}).get(best) or {}).get(
            "samples_per_second"
        ) or {}
        percent = change.get("percent")
        if percent is not None:
            evidence += f", {percent:+.0f}% against {baseline}"
    evidence += "."

    if not layouts.get("controlled", True):
        cautions.append(
            "The layout comparison was not fully controlled: something other than the "
            "layout differed between runs. Treat the ranking as indicative."
        )
    if any(row.get("runs", 1) < 2 for row in sound.values()):
        cautions.append(
            "At least one layout was measured only once, so run-to-run variation is "
            "unknown and a small margin between layouts is not evidence."
        )
    return {"layout": best, "evidence": evidence}


def _worker_recommendation(
    workers: dict[str, Any] | None,
    cautions: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if not workers:
        return {
            "workers": 0,
            "limiting_factor": "unknown",
            "evidence": "No worker ladder was supplied, so the worker count is untuned.",
        }

    recommended = int(workers.get("recommended_workers", 0))
    pattern = workers.get("pattern", "unknown")
    factor = workers.get("limiting_factor", "unknown")
    evidence = workers.get("explanation") or f"Ladder pattern: {pattern}."

    if pattern == "still-improving":
        cautions.append(
            "The worker ladder was still improving at its highest rung, so the best "
            "worker count may be higher than the one recommended. Add a rung."
        )
    if pattern == "inconclusive":
        cautions.append(
            "The worker ladder was inconclusive - too few rungs to see its shape."
        )
    if factor in ("storage-or-synchronisation", "cpu-decode"):
        limitations.append(
            f"The input path is limited by {factor.replace('-', ' ')}. Tuning the worker "
            "count further will not move it."
        )
    return {"workers": recommended, "limiting_factor": factor, "evidence": evidence}


def _storage_recommendation(
    storage: dict[str, Any] | None,
    planned_epochs: int,
    cautions: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if not storage:
        return {
            "storage": "scratch",
            "evidence": (
                "No storage comparison was supplied. Project scratch is the documented "
                "default for job I/O, so it is assumed rather than measured."
            ),
            "staging": "unknown",
            "staging_evidence": "Node-local staging was not measured.",
        }

    placements = storage.get("placements") or {}
    if not placements:
        return {
            "storage": "scratch",
            "evidence": "The storage comparison had no placements.",
            "staging": "unknown",
            "staging_evidence": "Node-local staging was not measured.",
        }

    # Cost at the horizon that was actually planned, not at a default one.
    costs = {
        name: storage_module.total_cost(row, planned_epochs)
        for name, row in placements.items()
    }
    cheapest = min(costs, key=lambda name: costs[name])

    # Break a near-tie in favour of project scratch. Scratch is LUMI's documented
    # location for job I/O and is neither scarce nor small, so preferring a different
    # placement needs a difference that survives the measurement noise. Choosing flash
    # because it was 4 % faster when repeats varied by 5 % would be exactly the
    # over-reading the rest of this tutorial warns against.
    tie_broken = False
    if cheapest != "scratch" and "scratch" in costs:
        margin = (costs["scratch"] - costs[cheapest]) / max(costs[cheapest], 1e-9)
        if margin <= _noise_band(placements, cheapest, "scratch"):
            cheapest = "scratch"
            tie_broken = True

    evidence = (
        f"Over {planned_epochs} epoch(s), {cheapest} costs {costs[cheapest]:.1f}s in "
        f"total including setup"
    )
    others = {name: value for name, value in costs.items() if name != cheapest}
    if others:
        runner_up = min(others, key=lambda name: others[name])
        evidence += f", against {others[runner_up]:.1f}s for {runner_up}"
    evidence += "."
    if tie_broken:
        evidence += (
            " Placements were within measurement noise of each other, so project "
            "scratch is preferred as the documented default for job I/O."
        )

    staging, staging_evidence = _staging_recommendation(
        storage, placements, planned_epochs, cheapest
    )

    if any("indistinguishable" in caution for caution in storage.get("cautions") or []):
        cautions.append(
            "Storage placements were indistinguishable on speed within the measured "
            "noise. The recommendation rests on setup cost, not on read performance."
        )
    return {
        "storage": cheapest,
        "evidence": evidence,
        "staging": staging,
        "staging_evidence": staging_evidence,
    }


def _noise_band(
    placements: dict[str, Any], first: str, second: str
) -> float:
    """Relative difference below which two placements are indistinguishable.

    Two independent relative uncertainties combine in quadrature. With no repeats there
    is no measured variability, so nothing can be called a tie and the cheaper
    placement stands.
    """
    cvs = [
        float(placements[name].get("samples_per_second_cv", 0.0) or 0.0)
        for name in (first, second)
    ]
    return (cvs[0] ** 2 + cvs[1] ** 2) ** 0.5


def _staging_recommendation(
    storage: dict[str, Any],
    placements: dict[str, Any],
    planned_epochs: int,
    cheapest: str,
) -> tuple[str, str]:
    """Decide staging on break-even against the planned epoch count."""
    if "tmp" not in placements:
        return (
            "not-measured",
            "Node-local staging was not among the measured placements.",
        )

    comparison = (storage.get("comparisons") or {}).get("tmp") or {}
    break_even = comparison.get("break_even_epochs")
    setup = storage_module.setup_cost(placements["tmp"])

    if break_even is None:
        return (
            "not-recommended",
            f"Staging cost {setup:.1f}s and saved no per-epoch time, so the cost is "
            "never recovered however many epochs you run.",
        )
    if break_even > planned_epochs:
        return (
            "not-recommended",
            f"Staging breaks even after {break_even:.0f} epochs but only "
            f"{planned_epochs} are planned. The {setup:.1f}s copy would not be repaid.",
        )
    return (
        "recommended" if cheapest == "tmp" else "viable",
        f"Staging breaks even after {break_even:.1f} epochs, within the "
        f"{planned_epochs} planned.",
    )


def _distributed_verdict(
    distributed: dict[str, Any] | None,
    blocking: list[str],
    cautions: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    if not distributed:
        return {
            "state": "unverified",
            "evidence": "No distributed validation was supplied.",
        }

    duplicates = int(distributed.get("duplicate_samples", 0))
    missing = int(distributed.get("missing_samples", 0))
    idle = distributed.get("idle_ranks") or []
    world = int(distributed.get("world_size", 1))
    unique = int(distributed.get("unique_samples", 0))
    spread = float(distributed.get("rank_elapsed_spread", 0.0))
    wait = float(distributed.get("max_data_wait_fraction", 0.0))

    if duplicates:
        blocking.append(
            f"{duplicates} duplicate sample read(s) across {world} ranks: the ranks are "
            "not partitioning the dataset, so aggregate throughput is measuring "
            "redundant work."
        )
    if missing:
        blocking.append(
            f"{missing} sample(s) were never read: some samples are assigned to nobody."
        )
    if idle:
        blocking.append(
            f"Rank(s) {idle} received no data at all, so part of the allocation does "
            "nothing. There are fewer shards than readers."
        )

    if spread > RANK_SPREAD_CAUTION:
        cautions.append(
            f"Rank elapsed times differ by {spread:.0%}. Partitioning is correct but "
            "unbalanced, and with synchronised ranks the slowest sets the pace. "
            "Consider building shards balanced by estimated work."
        )
    if wait > DATA_WAIT_CAUTION:
        limitations.append(
            f"At least one rank spent {wait:.0%} of its time waiting for data. The input "
            "path is still the dominant cost even where it is correct."
        )

    state = "valid" if not (duplicates or missing or idle) else "invalid"
    evidence = (
        f"{world} ranks read {unique} distinct samples; duplicates {duplicates}, "
        f"missing {missing}, idle ranks {len(idle)}; rank elapsed spread {spread:.1%}."
    )
    return {"state": state, "evidence": evidence}


def _next_experiment(readiness: str, blocking: list[str]) -> str:
    if readiness == "NOT_READY":
        return "fix-blocking-issues-then-revalidate"
    if readiness == "INCONCLUSIVE":
        return "complete-the-missing-experiments"
    return "scaling-aware-ai-one-gcd-baseline"


def _dataset_summary(inspection: dict[str, Any] | None) -> str:
    if not inspection:
        return ""
    tree = inspection.get("tree") or {}
    sizes = inspection.get("file_sizes") or {}
    small = inspection.get("small_files") or {}
    files = tree.get("total_files", 0)
    if not files:
        return ""
    return (
        f"The source dataset contains {files} files totalling "
        f"{tree.get('total_gib', 0.0):.2f} GiB, with a median size of "
        f"{sizes.get('median_bytes', 0.0) / 1024:.1f} KiB. "
        f"{small.get('fraction', 0.0):.0%} of them are under "
        f"{small.get('threshold_bytes', 0)} bytes."
    )


def _median(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, dict):
        return float(value.get("median", 0.0))
    return float(value or 0.0)


def format_keyvalue(report: dict[str, Any]) -> str:
    """The greppable summary, as the tutorial plan specifies it."""
    lines = [
        f"DATA_READINESS={report['data_readiness']}",
        f"RECOMMENDED_LAYOUT={report['recommended_layout']}",
        f"RECOMMENDED_STORAGE={report['recommended_storage']}",
        f"RECOMMENDED_WORKERS_PER_RANK={report['recommended_workers_per_rank']}",
        f"NODE_LOCAL_STAGING={report['node_local_staging']}",
        f"DISTRIBUTED_PARTITIONING={report['distributed_partitioning']}",
        f"MAIN_LIMITING_FACTOR={report['main_limiting_factor']}",
        f"PLANNED_EPOCHS={report['planned_epochs']}",
        f"NEXT_EXPERIMENT={report['next_experiment']}",
    ]
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    """The written conclusion - the tutorial's principal deliverable."""
    state = report["data_readiness"]
    lines = [
        "# Data-readiness decision",
        "",
        f"**{state}** - for a workload of {report['planned_epochs']} epoch(s).",
        "",
    ]

    if report["dataset_summary"]:
        lines += ["## The dataset", "", report["dataset_summary"], ""]

    lines += [
        "## Recommendation",
        "",
        "| Decision | Value | Evidence |",
        "| -------- | ----- | -------- |",
        f"| Layout | `{report['recommended_layout']}` | {report['layout_evidence']} |",
        f"| Storage | `{report['recommended_storage']}` | {report['storage_evidence']} |",
        f"| Workers per rank | `{report['recommended_workers_per_rank']}` | {report['worker_evidence']} |",
        f"| Node-local staging | `{report['node_local_staging']}` | {report['staging_evidence']} |",
        f"| Distributed partitioning | `{report['distributed_partitioning']}` | {report['distributed_evidence']} |",
        f"| Main limiting factor | `{report['main_limiting_factor']}` | |",
        "",
    ]

    if report["blocking_issues"]:
        lines += ["## Blocking issues", "", "These must be fixed before scaling.", ""]
        lines += [f"{index}. {issue}" for index, issue in enumerate(report["blocking_issues"], 1)]
        lines.append("")

    if report["cautions"]:
        lines += ["## Cautions", ""]
        lines += [f"- {caution}" for caution in report["cautions"]]
        lines.append("")

    lines += ["## What this does not establish", ""]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    lines += [
        "",
        "## Next experiment",
        "",
        f"`{report['next_experiment']}`",
        "",
    ]
    if state in ("READY", "READY_WITH_CAUTION"):
        lines += [
            "The input path is ready for a one-node scaling experiment. Continue with",
            "[Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai),",
            "which asks whether additional GCDs and nodes produce useful throughput.",
        ]
    elif state == "NOT_READY":
        lines.append(
            "Fix the blocking issues above and re-run the affected part. Scaling a "
            "workload with an incorrect input path multiplies the waste."
        )
    else:
        lines.append(
            "Run the missing experiments before drawing a conclusion. An absent "
            "measurement is not a passing one: "
            f"{', '.join(report['inputs_missing']) or 'none'} were not supplied."
        )
    return "\n".join(lines) + "\n"
