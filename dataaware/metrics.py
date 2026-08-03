"""Statistics used when summarising and comparing runs.

Deliberately small. The tutorial argues that a single number is not evidence, so
these helpers make it cheap to report spread alongside a central value.
"""

from __future__ import annotations

from typing import Sequence

MIB = 1024.0 * 1024.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100].

    Implemented here rather than pulled from numpy so that reporting tools stay
    usable in a minimal environment, and so the interpolation rule is visible
    where the tutorial explains p95 batch wait.
    """
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Standard deviation divided by the mean, or 0.0 for a zero mean.

    Used for shard-size balance and for run-to-run variability. Unitless, so it
    compares across metrics.
    """
    if len(values) < 2:
        return 0.0
    average = mean(values)
    if average == 0.0:
        return 0.0
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return (variance**0.5) / average


def spread(values: Sequence[float]) -> float:
    """Relative spread ``(max - min) / max``, reported for rank imbalance.

    Zero means perfectly balanced. Expressed against the maximum so that the
    slowest rank, which sets synchronised progress, anchors the comparison.
    """
    if not values:
        return 0.0
    highest = max(values)
    if highest == 0.0:
        return 0.0
    return (highest - min(values)) / highest


def break_even_epochs(staging_seconds: float, per_epoch_time_saved: float) -> float | None:
    """Epochs needed before staging pays for itself.

    Returns ``None`` when staging saves nothing per epoch: the cost is then never
    recovered, and reporting a large number would suggest it eventually would be.
    """
    if per_epoch_time_saved <= 0.0:
        return None
    return staging_seconds / per_epoch_time_saved


def mib_per_second(byte_count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return (byte_count / MIB) / seconds


def per_second(count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return count / seconds
