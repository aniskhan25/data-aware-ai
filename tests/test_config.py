"""Configuration validation.

A misspelled or out-of-range option must stop the run. The alternative is a
confident measurement of a configuration nobody chose.
"""

from __future__ import annotations

import pytest

from dataaware.config import build_config, load_config
from dataaware.errors import ConfigError


def test_unknown_section_is_rejected(base_config_dict):
    base_config_dict["loaders"] = {"batch_size": 8}
    with pytest.raises(ConfigError, match="unknown section"):
        build_config(base_config_dict)


def test_unknown_option_is_rejected(base_config_dict):
    base_config_dict["loader"]["num_worker"] = 4
    with pytest.raises(ConfigError, match="unknown option"):
        build_config(base_config_dict)


def test_unknown_layout_is_rejected(base_config_dict):
    base_config_dict["dataset"]["layout"] = "parquet"
    with pytest.raises(ConfigError, match="dataset.layout must be one of"):
        build_config(base_config_dict)


def test_persistent_workers_requires_workers(base_config_dict):
    base_config_dict["loader"]["num_workers"] = 0
    base_config_dict["loader"]["persistent_workers"] = True
    with pytest.raises(ConfigError, match="persistent_workers requires"):
        build_config(base_config_dict)


def test_stage_to_tmp_requires_tmp_location(base_config_dict):
    base_config_dict["storage"] = {"location": "scratch", "stage_to_tmp": True}
    with pytest.raises(ConfigError, match="only valid with storage.location: tmp"):
        build_config(base_config_dict)


def test_environment_variables_are_expanded(base_config_dict):
    base_config_dict["dataset"]["root"] = "${MY_ROOT}/source"
    base_config_dict["dataset"]["manifest"] = "$MY_ROOT/m.jsonl"
    config = build_config(base_config_dict, environ={"MY_ROOT": "/scratch/x"})
    assert str(config.dataset_root) == "/scratch/x/source"
    assert str(config.manifest_path) == "/scratch/x/m.jsonl"


def test_unset_environment_variable_is_an_error(base_config_dict):
    base_config_dict["dataset"]["root"] = "${DEFINITELY_NOT_SET_12345}/source"
    with pytest.raises(ConfigError, match="undefined environment variable"):
        build_config(base_config_dict, environ={})


def test_config_hash_ignores_output_directory(base_config_dict):
    first = build_config(base_config_dict)
    base_config_dict["output"]["directory"] = "/somewhere/else"
    second = build_config(base_config_dict)
    assert first.config_hash() == second.config_hash()


def test_config_hash_changes_with_measured_settings(base_config_dict):
    first = build_config(base_config_dict)
    base_config_dict["loader"]["num_workers"] = 4
    base_config_dict["loader"]["persistent_workers"] = True
    second = build_config(base_config_dict)
    assert first.config_hash() != second.config_hash()


def test_shipped_configs_are_valid(repo_root):
    """Every committed run configuration must load, given its variables."""
    environ = {
        "TUTORIAL_ROOT": "/scratch/project_placeholder/data-aware-ai",
        "TUTORIAL_FLASH_ROOT": "/flash/project_placeholder/data-aware-ai",
        "DAAI_TEST_ROOT": "/tmp/daai-test",
    }
    configs = sorted((repo_root / "configs").rglob("*.yaml"))
    run_configs = [p for p in configs if p.parent.name != "datasets"]
    assert run_configs, "no run configurations found"
    for path in run_configs:
        load_config(path, environ=environ)


def test_shipped_configs_contain_no_project_id(repo_root):
    """A committed config must never carry a real project allocation."""
    import re

    pattern = re.compile(r"project_\d{4,}")
    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        assert not pattern.search(path.read_text()), f"{path} contains a project ID"
