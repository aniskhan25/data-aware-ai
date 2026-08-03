"""Strict YAML configuration for tutorial experiments.

Two rules drive the design:

* An unknown or misspelled field is an error, never a silently ignored value.
  A typo in ``num_workers`` would otherwise produce a confident measurement of
  the wrong configuration.
* No committed configuration contains a project identifier. Paths use
  environment variables such as ``${TUTORIAL_ROOT}``, expanded at load time with
  a clear error when the variable is unset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: Dataset representations compared by the tutorial. Only ``loose-files`` is
#: implemented in the current release; the others are accepted by the
#: configuration and rejected by the runner with a pointer to the relevant part.
LAYOUTS = ("loose-files", "squashfs", "webdataset")

#: Storage placements compared in Part V.
STORAGE_LOCATIONS = ("scratch", "flash", "tmp", "local")

#: How a SquashFS image is made readable. 'prebound' expects it already mounted or
#: bound at dataset.root, which is what a container bind produces on LUMI.
SQUASHFS_MODES = ("prebound", "squashfuse")

_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


class ConfigError(ValueError):
    """Raised for any invalid, unknown, or unresolvable configuration value."""


@dataclass(frozen=True)
class RunSection:
    name: str
    seed: int = 1234
    warmup_batches: int = 20
    measured_batches: int = 200


@dataclass(frozen=True)
class DatasetSection:
    layout: str
    #: Where samples are read from. For ``loose-files`` the dataset tree; for
    #: ``squashfs`` the mount point the image appears at; for ``webdataset`` the
    #: directory holding the shards.
    root: str
    manifest: str
    #: ``module:Class`` path to a :class:`DatasetAdapter` for user datasets.
    adapter: str = ""
    #: SquashFS only: the image file. Required when squashfs_mode is 'squashfuse'.
    image: str = ""
    #: SquashFS only. 'prebound' reads an image already mounted or bound at root,
    #: which is how a container presents one. 'squashfuse' mounts the image with
    #: squashfuse and unmounts it afterwards.
    squashfs_mode: str = "prebound"
    #: WebDataset only: the shard index. Defaults to <root>/shard_index.json.
    shard_index: str = ""


@dataclass(frozen=True)
class LoaderSection:
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    shuffle: bool = True
    drop_last: bool = True
    #: Size of the synthetic per-batch compute step. Keeps the benchmark from
    #: degenerating into a pure storage microbenchmark. 0 disables it.
    compute_steps: int = 1
    #: Streaming layouts only: samples held back and drawn from at random, which is
    #: how a sequential shard reader approximates shuffling. 0 disables it.
    shuffle_buffer: int = 0


@dataclass(frozen=True)
class DistributedSection:
    enabled: bool = False
    validate_unique_samples: bool = True


@dataclass(frozen=True)
class StorageSection:
    location: str = "scratch"
    stage_to_tmp: bool = False


@dataclass(frozen=True)
class OutputSection:
    directory: str


@dataclass(frozen=True)
class Config:
    run: RunSection
    dataset: DatasetSection
    loader: LoaderSection
    distributed: DistributedSection
    storage: StorageSection
    output: OutputSection
    #: The fully resolved configuration, recorded verbatim in every run summary.
    resolved: dict[str, Any] = field(default_factory=dict)
    #: Path the configuration was loaded from, for provenance.
    source_path: str = ""

    @property
    def dataset_root(self) -> Path:
        return Path(self.dataset.root)

    @property
    def manifest_path(self) -> Path:
        return Path(self.dataset.manifest)

    @property
    def output_directory(self) -> Path:
        return Path(self.output.directory)

    def config_hash(self) -> str:
        """Stable hash of everything that affects what is measured.

        The ``output`` section is excluded: writing results to a different
        directory does not make it a different experiment.
        """
        payload = {k: v for k, v in self.resolved.items() if k != "output"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


_SECTIONS = {
    "run": RunSection,
    "dataset": DatasetSection,
    "loader": LoaderSection,
    "distributed": DistributedSection,
    "storage": StorageSection,
    "output": OutputSection,
}


def expand_vars(value: str, environ: dict[str, str] | None = None) -> str:
    """Expand ``$VAR`` and ``${VAR}`` references, failing loudly when unset.

    ``os.path.expandvars`` leaves unset variables in place, which turns a missing
    ``TUTORIAL_ROOT`` into a job that writes to a literal ``$TUTORIAL_ROOT``
    directory. That failure mode is worse than an exception.
    """
    env = os.environ if environ is None else environ
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in env:
            missing.append(name)
            return ""
        return env[name]

    expanded = _VAR_PATTERN.sub(replace, value)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise ConfigError(
            f"undefined environment variable(s) {names} in {value!r}; "
            "set them in your site configuration (see env.example.sh)"
        )
    return os.path.expanduser(expanded)


def load_config(
    path: str | os.PathLike[str],
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load, validate, and resolve an experiment configuration.

    ``overrides`` applies ``{"loader.num_workers": 8}`` style keys after the file
    is read, so that a worker ladder can reuse one configuration in tests.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None

    if raw is None:
        raise ConfigError(f"{path}: configuration is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

    for key, value in (overrides or {}).items():
        _apply_override(raw, key, value)

    return _build(raw, source_path=str(path), environ=environ)


def config_from_dict(
    raw: dict[str, Any],
    source_path: str = "<dict>",
    environ: dict[str, str] | None = None,
) -> Config:
    """Validate an in-memory configuration. Used by tests and adapters."""
    return _build(dict(raw), source_path=source_path, environ=environ)


def _apply_override(raw: dict[str, Any], dotted: str, value: Any) -> None:
    section, _, option = dotted.partition(".")
    if not option:
        raise ConfigError(f"override {dotted!r} must use the form section.option")
    raw.setdefault(section, {})[option] = value


def _build(
    raw: dict[str, Any],
    source_path: str,
    environ: dict[str, str] | None,
) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(
            f"{source_path}: unknown section(s) {sorted(unknown)}; "
            f"known sections are {sorted(_SECTIONS)}"
        )

    built: dict[str, Any] = {}
    for name, section_type in _SECTIONS.items():
        built[name] = _build_section(section_type, name, raw.get(name), source_path)

    config = Config(
        run=built["run"],
        dataset=built["dataset"],
        loader=built["loader"],
        distributed=built["distributed"],
        storage=built["storage"],
        output=built["output"],
        resolved={name: _as_dict(value) for name, value in built.items()},
        source_path=source_path,
    )
    _validate_values(config, source_path)
    return _expand_paths(config, environ)


def _build_section(section_type: type, name: str, data: Any, source_path: str):
    spec = {f.name: f for f in fields(section_type)}
    required = [f.name for f in fields(section_type) if f.default is MISSING]

    if data is None:
        if required:
            raise ConfigError(
                f"{source_path}: section '{name}' is required and must define "
                f"{required}"
            )
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{source_path}: section '{name}' must be a mapping, got {type(data).__name__}"
        )

    unknown = set(data) - set(spec)
    if unknown:
        raise ConfigError(
            f"{source_path}: unknown option(s) {sorted(unknown)} in section '{name}'; "
            f"known options are {sorted(spec)}"
        )
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"{source_path}: section '{name}' is missing {missing}")

    coerced = {
        key: _coerce(value, spec[key].type, f"{name}.{key}", source_path)
        for key, value in data.items()
    }
    return section_type(**coerced)


def _coerce(value: Any, declared: Any, where: str, source_path: str) -> Any:
    """Coerce a YAML scalar to the declared field type.

    Field types arrive as strings because of ``from __future__ import
    annotations``, so they are compared by name.
    """
    name = declared if isinstance(declared, str) else getattr(declared, "__name__", str(declared))

    if name == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{source_path}: {where} must be true or false, got {value!r}")
        return value
    if name == "int":
        # bool is a subclass of int; accepting it here would hide typos.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{source_path}: {where} must be an integer, got {value!r}")
        return value
    if name == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{source_path}: {where} must be a string, got {value!r}")
        return value
    return value


def _validate_values(config: Config, source_path: str) -> None:
    def fail(message: str) -> None:
        raise ConfigError(f"{source_path}: {message}")

    if not config.run.name.strip():
        fail("run.name must not be empty")
    if config.run.warmup_batches < 0:
        fail("run.warmup_batches must be >= 0")
    if config.run.measured_batches < 1:
        fail("run.measured_batches must be >= 1")

    if config.dataset.layout not in LAYOUTS:
        fail(f"dataset.layout must be one of {list(LAYOUTS)}, got {config.dataset.layout!r}")

    if config.dataset.squashfs_mode not in SQUASHFS_MODES:
        fail(
            f"dataset.squashfs_mode must be one of {list(SQUASHFS_MODES)}, "
            f"got {config.dataset.squashfs_mode!r}"
        )
    if config.dataset.layout == "squashfs":
        if config.dataset.squashfs_mode == "squashfuse" and not config.dataset.image:
            fail("dataset.image is required when dataset.squashfs_mode is 'squashfuse'")
        if config.dataset.squashfs_mode == "prebound" and not config.dataset.root:
            fail("dataset.root is required when dataset.squashfs_mode is 'prebound'")
    elif config.dataset.image:
        fail(f"dataset.image is only meaningful for layout 'squashfs', not {config.dataset.layout!r}")

    if config.dataset.shard_index and config.dataset.layout != "webdataset":
        fail(
            "dataset.shard_index is only meaningful for layout 'webdataset', "
            f"not {config.dataset.layout!r}"
        )
    if config.dataset.layout == "webdataset" and config.loader.shuffle:
        # A streaming dataset has no index to shuffle. Order comes from shard
        # assignment and the shuffle buffer instead, so accepting shuffle: true here
        # would promise something the layout cannot deliver.
        fail(
            "loader.shuffle must be false for layout 'webdataset'; shard streaming "
            "shuffles by shard order and shuffle buffer, not by index"
        )

    if config.loader.batch_size < 1:
        fail("loader.batch_size must be >= 1")
    if config.loader.num_workers < 0:
        fail("loader.num_workers must be >= 0")
    if config.loader.prefetch_factor < 1:
        fail("loader.prefetch_factor must be >= 1")
    if config.loader.compute_steps < 0:
        fail("loader.compute_steps must be >= 0")
    if config.loader.shuffle_buffer < 0:
        fail("loader.shuffle_buffer must be >= 0")
    if config.loader.num_workers == 0 and config.loader.persistent_workers:
        fail("loader.persistent_workers requires loader.num_workers >= 1")

    if config.storage.location not in STORAGE_LOCATIONS:
        fail(
            f"storage.location must be one of {list(STORAGE_LOCATIONS)}, "
            f"got {config.storage.location!r}"
        )
    if config.storage.stage_to_tmp and config.storage.location != "tmp":
        fail("storage.stage_to_tmp is only valid with storage.location: tmp")


def _expand_paths(config: Config, environ: dict[str, str] | None) -> Config:
    dataset = DatasetSection(
        layout=config.dataset.layout,
        root=expand_vars(config.dataset.root, environ),
        manifest=expand_vars(config.dataset.manifest, environ),
        adapter=config.dataset.adapter,
        image=expand_vars(config.dataset.image, environ) if config.dataset.image else "",
        squashfs_mode=config.dataset.squashfs_mode,
        shard_index=(
            expand_vars(config.dataset.shard_index, environ)
            if config.dataset.shard_index
            else ""
        ),
    )
    output = OutputSection(directory=expand_vars(config.output.directory, environ))
    resolved = dict(config.resolved)
    resolved["dataset"] = _as_dict(dataset)
    resolved["output"] = _as_dict(output)
    return Config(
        run=config.run,
        dataset=dataset,
        loader=config.loader,
        distributed=config.distributed,
        storage=config.storage,
        output=output,
        resolved=resolved,
        source_path=config.source_path,
    )


def _as_dict(section: Any) -> dict[str, Any]:
    return {f.name: getattr(section, f.name) for f in fields(section)}
