"""Strict YAML configuration for experiments.

One rule drives the strictness: **an unknown or misspelled key is an error.** A typo in
`num_workers` would otherwise produce a confident measurement of a configuration nobody
chose, and you would never know. That is the failure this module exists to prevent, and
it is the only kind of validation here that earns its lines.

Values are not type checked. A `batch_size: "64"` fails loudly a moment later when
PyTorch is handed a string, which is soon enough.

Paths use environment variables like `${TUTORIAL_ROOT}`, expanded at load time. No
committed configuration contains a project identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

#: Dataset representations. The first three are the core layouts of Part III; `adapter`
#: delegates to a DatasetAdapter, which is how optional tracks and your own data plug in.
LAYOUTS = ("loose-files", "squashfs", "webdataset", "adapter")

STORAGE_LOCATIONS = ("scratch", "flash", "tmp", "local")
SQUASHFS_MODES = ("prebound", "squashfuse")

_VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


@dataclass(frozen=True)
class Run:
    name: str
    seed: int = 1234
    warmup_batches: int = 20
    measured_batches: int = 200
    #: Measure whole passes instead of a batch count. Needed to check coverage: with a
    #: fixed batch count, readers holding fewer shards cycle into a second epoch and
    #: register as duplicate reads even though partitioning is correct.
    measured_epochs: int = 0


@dataclass(frozen=True)
class DatasetSpec:
    layout: str
    #: Where samples are read from: the tree, the mount point, or the shard directory.
    root: str
    manifest: str
    #: `module:Class` for layout `adapter`.
    adapter: str = ""
    #: SquashFS image, required for squashfs_mode `squashfuse`.
    image: str = ""
    squashfs_mode: str = "prebound"
    shard_index: str = ""


@dataclass(frozen=True)
class Loader:
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    shuffle: bool = True
    drop_last: bool = True
    #: Small fixed per-batch computation, so the benchmark is not a pure storage
    #: microbenchmark and the data-wait fraction has a denominator.
    compute_steps: int = 1
    #: Streaming layouts only: how many samples are held back and drawn from at random.
    shuffle_buffer: int = 0


@dataclass(frozen=True)
class Distributed:
    enabled: bool = False
    validate_unique_samples: bool = True
    #: Setting this false makes every rank read everything, which is the duplicate-sample
    #: failure mode. Exposed on purpose for Part VI. Never turn it off in real work.
    partition_by_rank: bool = True
    backend: str = "gloo"


@dataclass(frozen=True)
class Storage:
    location: str = "scratch"
    stage_to_tmp: bool = False
    tmp_dir: str = ""
    #: Largest share of allocated memory a staged dataset may occupy. Node-local /tmp is
    #: memory, so the rest of the allocation still has to hold the workload.
    safety_fraction: float = 0.5
    validate_staged: bool = True
    memory_bytes: int = 0


@dataclass(frozen=True)
class Output:
    directory: str


SECTIONS = {
    "run": Run,
    "dataset": DatasetSpec,
    "loader": Loader,
    "distributed": Distributed,
    "storage": Storage,
    "output": Output,
}


@dataclass(frozen=True)
class Config:
    run: Run
    dataset: DatasetSpec
    loader: Loader
    distributed: Distributed
    storage: Storage
    output: Output
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

    @property
    def resolved(self) -> dict[str, Any]:
        """The whole configuration as plain data, recorded in every run summary."""
        return {
            name: {f.name: getattr(getattr(self, name), f.name) for f in fields(section)}
            for name, section in SECTIONS.items()
        }

    def config_hash(self) -> str:
        """Hash of everything that affects what is measured.

        `output` is excluded: writing results elsewhere is not a different experiment.
        """
        payload = {k: v for k, v in self.resolved.items() if k != "output"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def with_root(self, root: Path) -> Config:
        return replace(self, dataset=replace(self.dataset, root=str(root)))

    def with_image(self, image: Path) -> Config:
        return replace(self, dataset=replace(self.dataset, image=str(image)))


def expand(value: str, environ: dict[str, str] | None = None) -> str:
    """Expand `$VAR` and `${VAR}`, failing when one is unset.

    `os.path.expandvars` leaves unset variables in place, which turns a missing
    TUTORIAL_ROOT into a job that writes to a directory literally named
    `$TUTORIAL_ROOT`. An exception is the better failure.
    """
    env = os.environ if environ is None else environ
    missing = []

    def substitute(match: re.Match) -> str:
        name = match.group(1) or match.group(2)
        if name not in env:
            missing.append(name)
            return ""
        return env[name]

    expanded = _VARIABLE.sub(substitute, value)
    if missing:
        raise ConfigError(
            f"undefined environment variable(s) {', '.join(sorted(set(missing)))} in "
            f"{value!r}; set them in your site configuration (see env.example.sh)"
        )
    return os.path.expanduser(expanded)


def load_config(
    path: str | os.PathLike[str],
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load and validate a configuration.

    `overrides` takes `{"loader.num_workers": 8}` style keys, so one file can drive a
    ladder of runs.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping of sections, got {type(raw).__name__}")

    for dotted, value in (overrides or {}).items():
        section, _, option = dotted.partition(".")
        if not option:
            raise ConfigError(f"override {dotted!r} must look like section.option")
        raw.setdefault(section, {})[option] = value

    return build_config(raw, source_path=str(path), environ=environ)


def build_config(
    raw: dict[str, Any],
    source_path: str = "<dict>",
    environ: dict[str, str] | None = None,
) -> Config:
    """Validate an in-memory configuration."""
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ConfigError(f"{source_path}: unknown section(s) {sorted(unknown)}")

    built = {}
    for name, section in SECTIONS.items():
        built[name] = _section(section, name, raw.get(name) or {}, source_path)

    config = Config(**built, source_path=source_path)
    _check(config, source_path)
    return _expand_paths(config, environ)


def _section(section: type, name: str, data: Any, source_path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{source_path}: section '{name}' must be a mapping")

    known = {f.name for f in fields(section)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{source_path}: unknown option(s) {unknown} in section '{name}'; "
            f"known options are {sorted(known)}"
        )
    try:
        return section(**data)
    except TypeError as exc:
        raise ConfigError(f"{source_path}: section '{name}' is incomplete: {exc}") from None


def _check(config: Config, source_path: str) -> None:
    """Reject combinations that would measure something other than what was asked for."""

    def fail(message: str) -> None:
        raise ConfigError(f"{source_path}: {message}")

    if config.dataset.layout not in LAYOUTS:
        fail(f"dataset.layout must be one of {list(LAYOUTS)}")
    if config.storage.location not in STORAGE_LOCATIONS:
        fail(f"storage.location must be one of {list(STORAGE_LOCATIONS)}")
    if config.dataset.squashfs_mode not in SQUASHFS_MODES:
        fail(f"dataset.squashfs_mode must be one of {list(SQUASHFS_MODES)}")

    if config.loader.num_workers == 0 and config.loader.persistent_workers:
        fail("loader.persistent_workers requires loader.num_workers >= 1")

    # A streaming layout has no index to permute, so accepting shuffle: true would
    # promise an ordering it cannot deliver.
    if config.dataset.layout == "webdataset" and config.loader.shuffle:
        fail(
            "loader.shuffle must be false for layout 'webdataset'; order comes from "
            "shard assignment and loader.shuffle_buffer instead"
        )
    if config.dataset.layout == "adapter" and not config.dataset.adapter:
        fail("dataset.layout: adapter requires dataset.adapter, e.g. 'module:Class'")
    if config.dataset.layout == "squashfs" and config.dataset.squashfs_mode == "squashfuse":
        if not config.dataset.image:
            fail("dataset.image is required when squashfs_mode is 'squashfuse'")

    if config.storage.location == "tmp" and not config.storage.stage_to_tmp:
        fail("storage.location: tmp requires storage.stage_to_tmp: true")
    if config.storage.stage_to_tmp and config.storage.location != "tmp":
        fail("storage.stage_to_tmp is only valid with storage.location: tmp")
    if not 0.0 < config.storage.safety_fraction <= 1.0:
        fail("storage.safety_fraction must be in (0, 1]")


def _expand_paths(config: Config, environ: dict[str, str] | None) -> Config:
    dataset = replace(
        config.dataset,
        root=expand(config.dataset.root, environ),
        manifest=expand(config.dataset.manifest, environ),
        image=expand(config.dataset.image, environ) if config.dataset.image else "",
        shard_index=(
            expand(config.dataset.shard_index, environ) if config.dataset.shard_index else ""
        ),
    )
    storage = replace(
        config.storage,
        tmp_dir=expand(config.storage.tmp_dir, environ) if config.storage.tmp_dir else "",
    )
    output = Output(directory=expand(config.output.directory, environ))
    return replace(config, dataset=dataset, storage=storage, output=output)
