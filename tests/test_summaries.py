"""Run-summary schema and the statistics used to report results.

Later phases read only summaries, so a schema violation must fail at the moment
the summary is written, not when a comparison script tries to use it.
"""

from __future__ import annotations

import json

import pytest

from dataaware import metrics
from dataaware.schema import (
    COMMON,
    SCHEMA_VERSION,
    SummaryError,
    format_keyvalue,
    new_run_summary,
    read_run_summary,
    validate_run_summary,
    write_run_summary,
)


def test_new_summary_has_every_common_field():
    summary = new_run_summary(run_name="x")
    assert set(COMMON) <= set(summary)
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["run_name"] == "x"


def test_unknown_field_is_rejected_at_construction():
    with pytest.raises(SummaryError, match="unknown summary field"):
        new_run_summary(sampels_per_second=1.0)


def test_missing_required_field_is_rejected():
    summary = new_run_summary()
    del summary["samples_per_second"]
    with pytest.raises(SummaryError, match="missing required field"):
        validate_run_summary(summary)


def test_unsupported_schema_version_is_rejected():
    summary = new_run_summary()
    summary["schema_version"] = "0.9"
    with pytest.raises(SummaryError, match="not supported"):
        validate_run_summary(summary)


def test_write_then_read_round_trip(tmp_path):
    summary = new_run_summary(run_name="round-trip", samples_per_second=12.5)
    path = write_run_summary(tmp_path / "nested/run_summary.json", summary)
    assert read_run_summary(path) == summary
    # Written sorted and indented, so a summary is readable and diffable.
    assert json.loads(path.read_text())["run_name"] == "round-trip"


def test_keyvalue_output_is_uppercase_and_ordered():
    summary = new_run_summary(layout="loose-files", samples_per_second=1234.5678)
    text = format_keyvalue(summary)
    assert "LAYOUT=loose-files" in text
    assert "SAMPLES_PER_SECOND=1235" in text
    assert text.index("LAYOUT=") < text.index("SAMPLES_PER_SECOND=")


# --- metrics -----------------------------------------------------------------


def test_percentile_endpoints_and_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert metrics.percentile(values, 0) == 1.0
    assert metrics.percentile(values, 100) == 4.0
    assert metrics.percentile(values, 50) == 2.5


def test_coefficient_of_variation():
    assert metrics.coefficient_of_variation([5.0, 5.0, 5.0]) == 0.0
    assert metrics.coefficient_of_variation([1.0]) == 0.0
    assert metrics.coefficient_of_variation([]) == 0.0
    assert metrics.coefficient_of_variation([1.0, 3.0]) > 0.0


def test_break_even_epochs():
    assert metrics.break_even_epochs(100.0, 10.0) == 10.0
    # Staging that saves nothing is never recovered; a number would imply it is.
    assert metrics.break_even_epochs(100.0, 0.0) is None
    assert metrics.break_even_epochs(100.0, -5.0) is None


def test_rate_helpers_guard_against_zero_time():
    assert metrics.per_second(10, 0.0) == 0.0
    assert metrics.mib_per_second(1024, 0.0) == 0.0
    assert metrics.mib_per_second(int(metrics.MIB), 2.0) == 0.5
