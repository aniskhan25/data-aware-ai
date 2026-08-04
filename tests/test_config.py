"""Configuration validation.

A misspelled or out-of-range option must stop the run. The alternative is a
confident measurement of a configuration nobody chose.
"""

from __future__ import annotations

import pytest

from dataaware.config import ConfigError, config_from_dict, expand_vars, load_config


def test_valid_config_loads(base_config_dict):
    config = config_from_dict(base_config_dict)
    assert config.dataset.layout == "loose-files"
    assert config.loader.batch_size == 8
    # Defaults fill in for options the file omitted.
    assert config.loader.prefetch_factor == 2
    assert config.distributed.validate_unique_samples is True


def test_unknown_section_is_rejected(base_config_dict):
    base_config_dict["loaders"] = {"batch_size": 8}
    with pytest.raises(ConfigError, match="unknown section"):
        config_from_dict(base_config_dict)


def test_unknown_option_is_rejected(base_config_dict):
    base_config_dict["loader"]["num_worker"] = 4
    with pytest.raises(ConfigError, match="unknown option"):
        config_from_dict(base_config_dict)


def test_missing_required_option_is_rejected(base_config_dict):
    del base_config_dict["dataset"]["manifest"]
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict(base_config_dict)


def test_missing_required_section_is_rejected(base_config_dict):
    del base_config_dict["output"]
    with pytest.raises(ConfigError, match="section 'output' is required"):
        config_from_dict(base_config_dict)


def test_wrong_type_is_rejected(base_config_dict):
    base_config_dict["loader"]["batch_size"] = "64"
    with pytest.raises(ConfigError, match="must be an integer"):
        config_from_dict(base_config_dict)


def test_bool_is_not_accepted_as_int(base_config_dict):
    """``bool`` subclasses ``int``; accepting it would hide a real mistake."""
    base_config_dict["loader"]["batch_size"] = True
    with pytest.raises(ConfigError, match="must be an integer"):
        config_from_dict(base_config_dict)


def test_int_is_not_accepted_as_bool(base_config_dict):
    base_config_dict["loader"]["shuffle"] = 1
    with pytest.raises(ConfigError, match="must be true or false"):
        config_from_dict(base_config_dict)


@pytest.mark.parametrize(
    ("section", "option", "value"),
    [
        ("run", "measured_batches", 0),
        ("run", "warmup_batches", -1),
        ("loader", "batch_size", 0),
        ("loader", "num_workers", -1),
        ("loader", "prefetch_factor", 0),
        ("loader", "compute_steps", -1),
    ],
)
def test_out_of_range_values_are_rejected(base_config_dict, section, option, value):
    base_config_dict[section][option] = value
    with pytest.raises(ConfigError, match="must be"):
        config_from_dict(base_config_dict)


def test_unknown_layout_is_rejected(base_config_dict):
    base_config_dict["dataset"]["layout"] = "parquet"
    with pytest.raises(ConfigError, match="dataset.layout must be one of"):
        config_from_dict(base_config_dict)


def test_persistent_workers_requires_workers(base_config_dict):
    base_config_dict["loader"]["num_workers"] = 0
    base_config_dict["loader"]["persistent_workers"] = True
    with pytest.raises(ConfigError, match="persistent_workers requires"):
        config_from_dict(base_config_dict)


def test_stage_to_tmp_requires_tmp_location(base_config_dict):
    base_config_dict["storage"] = {"location": "scratch", "stage_to_tmp": True}
    with pytest.raises(ConfigError, match="only valid with storage.location: tmp"):
        config_from_dict(base_config_dict)


def test_environment_variables_are_expanded(base_config_dict):
    base_config_dict["dataset"]["root"] = "${MY_ROOT}/source"
    base_config_dict["dataset"]["manifest"] = "$MY_ROOT/m.jsonl"
    config = config_from_dict(base_config_dict, environ={"MY_ROOT": "/scratch/x"})
    assert str(config.dataset_root) == "/scratch/x/source"
    assert str(config.manifest_path) == "/scratch/x/m.jsonl"


def test_unset_environment_variable_is_an_error(base_config_dict):
    base_config_dict["dataset"]["root"] = "${DEFINITELY_NOT_SET_12345}/source"
    with pytest.raises(ConfigError, match="undefined environment variable"):
        config_from_dict(base_config_dict, environ={})


def test_expand_vars_reports_every_missing_name():
    with pytest.raises(ConfigError) as excinfo:
        expand_vars("${A}/${B}", environ={})
    assert "A, B" in str(excinfo.value)


def test_config_hash_ignores_output_directory(base_config_dict):
    first = config_from_dict(base_config_dict)
    base_config_dict["output"]["directory"] = "/somewhere/else"
    second = config_from_dict(base_config_dict)
    assert first.config_hash() == second.config_hash()


def test_config_hash_changes_with_measured_settings(base_config_dict):
    first = config_from_dict(base_config_dict)
    base_config_dict["loader"]["num_workers"] = 4
    base_config_dict["loader"]["persistent_workers"] = True
    second = config_from_dict(base_config_dict)
    assert first.config_hash() != second.config_hash()


def test_overrides_apply(tmp_path, base_config_dict):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base_config_dict))
    config = load_config(path, overrides={"loader.batch_size": 4})
    assert config.loader.batch_size == 4


def test_override_requires_dotted_key(tmp_path, base_config_dict):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base_config_dict))
    with pytest.raises(ConfigError, match="section.option"):
        load_config(path, overrides={"batch_size": 4})


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_is_reported_clearly(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


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
