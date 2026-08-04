"""Application-level data-loading benchmark.

What is measured is *useful sample throughput*: the rate at which decoded,
batched samples reach the training step. Raw storage bandwidth is reported too,
but it is not the headline metric, because it hides decode bottlenecks, worker
stalls, and duplicated samples.

Requires PyTorch. Configuration, manifest, generation, and reporting do not, so
that those tools remain usable in a minimal environment.
"""

from __future__ import annotations

import io
import random
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from . import env, metrics, squashfs
from .config import Config
from .manifest import Sample, checksum_bytes, manifest_hash, read_manifest
from .schema import new_run_summary
from .staging import staged_artifact
from .shards import (
    assign_shards,
    iter_shard_samples,
    read_shard_index,
    shard_statistics,
)


class WorkerFailure(RuntimeError):
    """Raised when a DataLoader worker process dies during a measurement.

    Distinguished from other errors because the cause and the fix are specific, and
    PyTorch's own message is a long traceback that buries both.
    """


class LooseFileDataset(Dataset):
    """Reads one file per sample from an ordinary directory tree.

    This is the baseline every other layout is compared against, and the layout
    most datasets arrive in. Each ``__getitem__`` performs one path lookup, one
    open, one read, and one decode.
    """

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[Sample],
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root)
        self.samples = samples
        self.verify_checksums = verify_checksums

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            payload = (self.root / sample.relative_path).read_bytes()
            if self.verify_checksums and checksum_bytes(payload) != sample.checksum:
                raise ValueError(
                    f"checksum mismatch for {sample.sample_id}: the file on disk "
                    "does not match the manifest"
                )
            with Image.open(io.BytesIO(payload)) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            failed = 0
        except Exception as exc:  # noqa: BLE001 - a failed sample is data, not a crash
            # A single unreadable sample must not abort a measurement, but it must
            # be counted. A run with failed samples is not a valid comparison.
            print(f"WARNING failed sample {sample.sample_id}: {exc}", flush=True)
            pixels = np.zeros((sample.height, sample.width, 3), dtype=np.uint8)
            payload = b""
            failed = 1

        return {
            "image": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
            "class_id": sample.class_id,
            "sample_index": index,
            "byte_size": len(payload),
            "failed": failed,
        }


def coverage_expectation(
    total_samples: int,
    batch_size: int,
    drop_last: bool,
    num_workers: int,
    streaming: bool,
) -> tuple[int, int]:
    """How many samples a correct epoch should cover, and how many may be dropped.

    ``drop_last`` complicates this differently for the two dataset styles:

    * A map-style dataset batches one index stream, so exactly
      ``total_samples % batch_size`` samples are dropped. That is knowable, so the
      expectation is lowered precisely and nothing is tolerated beyond it.
    * A streaming dataset gives each worker its own stream, and each worker batches
      independently. Every worker may therefore drop up to ``batch_size - 1``
      samples. Which ones is not knowable in advance, so the shortfall is expressed
      as an allowance rather than a lower expectation.

    Returns ``(expected_coverage, drop_allowance)``.
    """
    if not drop_last:
        return total_samples, 0
    if streaming:
        return total_samples, (batch_size - 1) * max(1, num_workers)
    return (total_samples // batch_size) * batch_size, 0


@dataclass
class SampleAccounting:
    """Tracks which samples a run actually read.

    Duplicates are counted per epoch, because a correct sampler visits each
    sample at most once per epoch. Missing samples are only counted for epochs
    that were fully traversed inside the measured window: a partial epoch says
    nothing about coverage, and reporting it as missing data would be wrong.
    """

    expected_coverage: int
    drop_allowance: int = 0

    def __post_init__(self) -> None:
        self._epoch_seen: dict[int, int] = {}
        self._all_seen: set[int] = set()
        self.duplicate_samples = 0
        self.missing_samples = 0
        self.total_observed = 0
        self.complete_epochs = 0
        self.partial_epoch = False

    def observe(self, indices: Sequence[int]) -> None:
        for index in indices:
            self._epoch_seen[index] = self._epoch_seen.get(index, 0) + 1
            self._all_seen.add(index)
            self.total_observed += 1

    def end_epoch(self, complete: bool) -> None:
        observed = sum(self._epoch_seen.values())
        unique = len(self._epoch_seen)
        self.duplicate_samples += observed - unique
        if complete:
            self.complete_epochs += 1
            shortfall = self.expected_coverage - unique - self.drop_allowance
            self.missing_samples += max(0, shortfall)
        else:
            self.partial_epoch = True
        self._epoch_seen = {}

    @property
    def unique_samples(self) -> int:
        return len(self._all_seen)

    @property
    def observed_indices(self) -> list[int]:
        """Manifest positions this reader touched, sorted.

        Cross-rank duplicate detection needs identities, not counts: two ranks each
        reporting 100 samples could be covering 200 or the same 100 twice.
        """
        return sorted(self._all_seen)


class ShardStreamDataset(IterableDataset):
    """Streams samples from tar shards, one sequential pass per shard.

    Shards are assigned to workers round-robin. That assignment is the whole point
    of the layout: it is what lets many readers cover a dataset without any of them
    reading the same bytes. Nothing here corrects for having fewer shards than
    workers — that failure mode stays visible.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        shard_names: Sequence[str],
        index_by_sample_id: dict[str, int],
        class_by_sample_id: dict[str, int],
        seed: int = 1234,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.shard_names = list(shard_names)
        self.index_by_sample_id = index_by_sample_id
        self.class_by_sample_id = class_by_sample_id
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)

        # Readers are numbered across the whole job, not within a rank. Rank 1's first
        # worker must not be given the same shards as rank 0's first worker, so the
        # reader identity has to combine both.
        reader = self.rank * num_workers + worker_id
        readers = self.world_size * num_workers

        names = list(self.shard_names)
        if self.shuffle_shards:
            # Shard order is fixed by the seed rather than varied per epoch. That
            # keeps repeated epochs comparable, which is what this benchmark needs;
            # epoch-varying shard order belongs with the distributed sampler.
            random.Random(self.seed).shuffle(names)
        mine = assign_shards(names, reader, readers)

        stream = self._read_shards(mine)
        if self.shuffle_buffer > 1:
            stream = _shuffled(stream, self.shuffle_buffer, self.seed + worker_id)
        yield from stream

    def _read_shards(self, names: Sequence[str]) -> Iterator[dict[str, Any]]:
        for name in names:
            path = self.shard_dir / name
            opened = time.perf_counter()
            first = True
            try:
                sample_stream = iter_shard_samples(path)
                for sample_id, payload, class_id in sample_stream:
                    open_cost = time.perf_counter() - opened if first else 0.0
                    yield self._decode(sample_id, payload, class_id, first, open_cost)
                    first = False
            except (OSError, tarfile.TarError) as exc:
                print(f"WARNING failed shard {name}: {exc}", flush=True)

    def _decode(
        self,
        sample_id: str,
        payload: bytes,
        class_id: int,
        first_in_shard: bool,
        open_cost: float,
    ) -> dict[str, Any]:
        index = self.index_by_sample_id.get(sample_id, -1)
        try:
            if index < 0:
                raise KeyError(
                    f"sample {sample_id} is in a shard but not in the manifest"
                )
            with Image.open(io.BytesIO(payload)) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            failed = 0
            nbytes = len(payload)
        except Exception as exc:  # noqa: BLE001 - a failed sample is data, not a crash
            print(f"WARNING failed sample {sample_id}: {exc}", flush=True)
            pixels = np.zeros((1, 1, 3), dtype=np.uint8)
            failed = 1
            nbytes = 0

        return {
            "image": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
            "class_id": class_id,
            "sample_index": index,
            "byte_size": nbytes,
            "failed": failed,
            "shard_opened": 1 if first_in_shard else 0,
            "shard_open_seconds": open_cost,
        }


def _shuffled(
    stream: Iterator[dict[str, Any]], buffer_size: int, seed: int
) -> Iterator[dict[str, Any]]:
    """Approximate shuffling for a sequential stream.

    A streaming layout has no index to permute, so samples are held back in a
    buffer and drawn from at random. The window is the buffer, not the dataset: this
    is weaker than a full shuffle, and that trade-off is part of choosing shards.
    """
    rng = random.Random(seed)
    buffer: list[dict[str, Any]] = []
    for item in stream:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            position = rng.randrange(len(buffer))
            buffer[position], buffer[-1] = buffer[-1], buffer[position]
            yield buffer.pop()
    rng.shuffle(buffer)
    yield from buffer


def build_dataset(
    config: Config,
    samples: Sequence[Sample],
    rank: int = 0,
    world_size: int = 1,
) -> Dataset:
    """Construct the dataset for the configured layout.

    ``loose-files`` and ``squashfs`` share an implementation on purpose: a mounted
    SquashFS image presents ordinary paths, so the reader is identical and only the
    root differs. That is the property the packaging comparison is testing.
    """
    if config.dataset.layout in ("loose-files", "squashfs"):
        return LooseFileDataset(config.dataset_root, samples)
    # A rank that is told to behave as rank 0 reads the same shards as every other
    # rank. That is the duplicate-sample failure mode, reachable on purpose through
    # distributed.partition_by_rank so that Part VI can show what it looks like.
    if config.dataset.layout == "webdataset":
        index = read_shard_index(_shard_index_path(config))
        return ShardStreamDataset(
            shard_dir=config.dataset_root,
            shard_names=[record["shard"] for record in index["shards"]],
            index_by_sample_id={
                sample.sample_id: position for position, sample in enumerate(samples)
            },
            class_by_sample_id={sample.sample_id: sample.class_id for sample in samples},
            seed=config.run.seed,
            shuffle_shards=True,
            shuffle_buffer=config.loader.shuffle_buffer,
            rank=rank,
            world_size=world_size,
        )
    raise NotImplementedError(f"dataset.layout '{config.dataset.layout}' is not implemented")


def _shard_index_path(config: Config) -> Path:
    if config.dataset.shard_index:
        return Path(config.dataset.shard_index)
    return config.dataset_root / "shard_index.json"


def build_dataloader(config: Config, dataset: Dataset) -> DataLoader:
    """Construct a DataLoader that is reproducible for a fixed seed."""
    generator = torch.Generator()
    generator.manual_seed(config.run.seed)

    kwargs: dict[str, Any] = {
        "batch_size": config.loader.batch_size,
        "num_workers": config.loader.num_workers,
        "shuffle": config.loader.shuffle,
        "drop_last": config.loader.drop_last,
        "generator": generator,
        "worker_init_fn": _seed_worker,
    }
    if config.loader.num_workers > 0:
        kwargs["prefetch_factor"] = config.loader.prefetch_factor
        kwargs["persistent_workers"] = config.loader.persistent_workers
    return DataLoader(dataset, **kwargs)


def _seed_worker(worker_id: int) -> None:
    """Give each worker a distinct but reproducible seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    # One thread per worker. Without this, each worker would try to use every
    # allocated core, and the resulting oversubscription would be mistaken for a
    # storage problem.
    torch.set_num_threads(1)


def synthetic_compute(batch: torch.Tensor, steps: int) -> torch.Tensor:
    """Small fixed per-batch computation.

    Its purpose is to stop the benchmark from being a pure storage
    microbenchmark, and to give the data-wait fraction a denominator. It is
    deliberately far cheaper than real training: it does not model GPU cost, and
    its absolute time should not be compared against a training step.
    """
    features = batch.to(torch.float32).div_(255.0).mean(dim=(2, 3))
    for _ in range(steps):
        features = torch.tanh(features @ features.new_ones(3, 3) * 0.5)
    return features


def run_loader_benchmark(
    config: Config,
    repo_root: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
    collect_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Run one measured loading experiment and return this rank's run summary.

    With ``world_size`` above 1 this is one rank's share of the work; combining the
    ranks into a correctness verdict is :func:`dataaware.distributed.aggregate`.
    """
    samples = read_manifest(config.manifest_path)
    if not samples:
        raise ValueError(f"{config.manifest_path} contains no samples")
    if config.loader.drop_last and len(samples) < config.loader.batch_size:
        raise ValueError(
            f"{len(samples)} samples is fewer than loader.batch_size "
            f"{config.loader.batch_size} with drop_last enabled: every batch would "
            "be dropped"
        )

    torch.manual_seed(config.run.seed)
    if config.loader.num_workers > 0:
        # See _seed_worker: keep the main process from competing with its workers.
        torch.set_num_threads(1)

    job_started = time.perf_counter()
    with _staged_if_requested(config, samples) as (staged_config, staging_metrics):
        with prepared_layout(staged_config) as (resolved_config, layout_metrics):
            summary = _measure(
                resolved_config,
                samples,
                {**layout_metrics, **staging_metrics},
                repo_root=repo_root,
                rank=rank,
                world_size=world_size,
                collect_indices=collect_indices,
            )
    # Measured after staging is cleaned up, so it is the whole job's cost: staging,
    # startup, measurement, and teardown. This is the number that decides whether
    # staging was worth it, not the steady-state throughput above it.
    summary["total_job_seconds"] = time.perf_counter() - job_started
    return summary


@contextmanager
def _staged_if_requested(
    config: Config, samples: Sequence[Sample]
) -> Iterator[tuple[Config, dict[str, Any]]]:
    """Stage the dataset to node-local storage when the configuration asks for it.

    The artifact staged is the one the layout actually reads: the tree for loose
    files, the shard directory for a streaming layout, the image for SquashFS. Staging
    a packaged form moves far fewer files, which is usually the difference between a
    copy that pays for itself and one that does not.
    """
    if not config.storage.stage_to_tmp:
        yield config, {}
        return

    if config.dataset.layout == "squashfs":
        if not config.dataset.image:
            raise ValueError(
                "staging a squashfs layout needs dataset.image: the image is what "
                "gets copied, not the mount point"
            )
        source = Path(config.dataset.image)
    else:
        source = config.dataset_root

    with staged_artifact(
        source=source,
        tmp_dir=config.storage.tmp_dir,
        samples=samples if config.dataset.layout == "loose-files" else None,
        validate=config.storage.validate_staged,
        safety_fraction=config.storage.safety_fraction,
        memory_bytes=config.storage.memory_bytes or None,
    ) as staged:
        staged_path = Path(staged.pop("staged_path"))
        if config.dataset.layout == "squashfs":
            adjusted = _with_image(config, staged_path)
        else:
            adjusted = _with_root(config, staged_path)
        yield adjusted, staged


def _with_image(config: Config, image: Path) -> Config:
    """Copy a configuration with ``dataset.image`` replaced by the staged image."""
    from dataclasses import replace

    dataset = replace(config.dataset, image=str(image))
    resolved = dict(config.resolved)
    resolved["dataset"] = {**resolved["dataset"], "image": str(image)}
    return replace(config, dataset=dataset, resolved=resolved)


@contextmanager
def prepared_layout(config: Config) -> Iterator[tuple[Config, dict[str, Any]]]:
    """Make the configured layout readable, and clean up afterwards.

    Yields a configuration whose ``dataset.root`` points at readable data, together
    with layout-specific metrics for the run summary. For ``squashfuse`` mode the
    image is mounted here and unmounted on the way out, including on failure.
    """
    layout = config.dataset.layout

    if layout == "loose-files":
        yield config, {"filesystem_objects": _count_files(config.dataset_root)}
        return

    if layout == "webdataset":
        index = read_shard_index(_shard_index_path(config))
        statistics = shard_statistics(index)
        # The shards plus their index. This is the number the packaging comparison
        # puts against a loose tree's file count.
        statistics["filesystem_objects"] = statistics["num_shards"] + 1
        yield config, statistics
        return

    if layout == "squashfs":
        image_metrics: dict[str, Any] = {"filesystem_objects": 1}
        if config.dataset.image:
            image_metrics["image_bytes"] = Path(config.dataset.image).stat().st_size

        if config.dataset.squashfs_mode == "prebound":
            root = config.dataset_root
            if not root.is_dir():
                raise ValueError(
                    f"squashfs_mode is 'prebound' but {root} is not a directory. "
                    "Mount or bind the image there first, or use "
                    "dataset.squashfs_mode: squashfuse."
                )
            image_metrics["mount_seconds"] = 0.0
            yield config, image_metrics
            return

        with squashfs.mounted_image(config.dataset.image) as (mount_point, seconds):
            image_metrics["mount_seconds"] = seconds
            yield _with_root(config, mount_point), image_metrics
        return

    raise NotImplementedError(f"dataset.layout '{layout}' is not implemented")


def _with_root(config: Config, root: Path) -> Config:
    """Copy a configuration with ``dataset.root`` replaced.

    The resolved configuration recorded in the summary is updated too, so a summary
    always says where its data was actually read from.
    """
    from dataclasses import replace

    dataset = replace(config.dataset, root=str(root))
    resolved = dict(config.resolved)
    resolved["dataset"] = {**resolved["dataset"], "root": str(root)}
    return replace(config, dataset=dataset, resolved=resolved)


def _count_files(root: Path) -> int:
    """Files present under a loose-file root, for the object-count comparison."""
    import os

    total = 0
    for _, _, names in os.walk(root):
        total += len(names)
    return total


def _measure(
    config: Config,
    samples: Sequence[Sample],
    layout_metrics: dict[str, Any],
    repo_root: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
    collect_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Run the measured loop. The layout is already readable by this point."""
    streaming = config.dataset.layout == "webdataset"
    # partition_by_rank: false makes every rank claim to be rank 0, so all of them
    # read the same shards. That is the duplicate-sample break, on purpose.
    reader_rank = rank if config.distributed.partition_by_rank else 0
    reader_world = world_size if config.distributed.partition_by_rank else 1
    dataset = build_dataset(config, samples, rank=reader_rank, world_size=reader_world)
    loader = build_dataloader(config, dataset)

    expected_coverage, drop_allowance = coverage_expectation(
        total_samples=len(samples),
        batch_size=config.loader.batch_size,
        drop_last=config.loader.drop_last,
        num_workers=config.loader.num_workers,
        streaming=streaming,
    )
    accounting = SampleAccounting(
        expected_coverage=expected_coverage, drop_allowance=drop_allowance
    )

    batches = _epoch_cycling_batches(loader)

    # Startup covers worker creation and the first batch, which is then discarded.
    # It is reported separately rather than folded into throughput, because a
    # layout can trade startup cost against steady-state speed.
    # In epoch mode nothing may be consumed before measuring: a discarded first batch
    # would make the first pass incomplete and coverage unverifiable. Startup cost is
    # then folded into the first epoch's wait time instead of reported separately.
    startup_start = time.perf_counter()
    if config.run.measured_epochs == 0:
        if next(batches, None) is None:
            raise ValueError("the DataLoader produced no batches")
    startup_seconds = time.perf_counter() - startup_start

    warmup_start = time.perf_counter()
    for _ in range(0 if config.run.measured_epochs else config.run.warmup_batches):
        item = next(batches, None)
        if item is None:
            break
        synthetic_compute(item[1]["image"], config.loader.compute_steps)
    warmup_seconds = time.perf_counter() - warmup_start

    wait_times: list[float] = []
    compute_seconds = 0.0
    bytes_read = 0
    failed_samples = 0
    batches_measured = 0
    shard_opens = 0
    shard_open_seconds = 0.0
    # The epoch in progress when measurement began was partly consumed by startup
    # and warm-up, so its coverage cannot be judged.
    entry_epoch: int | None = None
    current_epoch: int | None = None

    # CPU counters are sampled as deltas across the measured window only, so that
    # startup and warm-up work is not attributed to steady-state utilisation.
    cpu_before = env.cpu_seconds()
    switches_before = env.context_switches()

    # Either a fixed number of batches, or a whole number of passes over this
    # reader's share. Epoch mode exists for coverage validation: see
    # RunSection.measured_epochs.
    epoch_mode = config.run.measured_epochs > 0
    budget = (
        config.run.measured_epochs if epoch_mode else config.run.measured_batches
    )

    measured_start = time.perf_counter()
    while True:
        if epoch_mode:
            if accounting.complete_epochs >= budget:
                break
        elif batches_measured >= budget:
            break

        wait_start = time.perf_counter()
        item = next(batches, None)
        wait = time.perf_counter() - wait_start
        if item is None:
            break
        epoch, batch = item

        if current_epoch is None:
            entry_epoch = current_epoch = epoch
        elif epoch != current_epoch:
            # The previous epoch finished. Coverage can only be judged for an
            # epoch measurement saw from its very first batch, which excludes the
            # one already in progress when the measured window opened — unless we are
            # in epoch mode, where warm-up is skipped so the first epoch is complete.
            accounting.end_epoch(
                complete=epoch_mode or current_epoch != entry_epoch
            )
            current_epoch = epoch
            if epoch_mode and accounting.complete_epochs >= budget:
                # An epoch boundary is only visible once a batch of the *next* pass
                # arrives. That batch must be discarded, not counted: including it
                # would read a handful of samples a second time and register as
                # duplicates in a run whose partitioning is perfectly correct.
                break

        wait_times.append(wait)

        compute_start = time.perf_counter()
        synthetic_compute(batch["image"], config.loader.compute_steps)
        compute_seconds += time.perf_counter() - compute_start

        accounting.observe(batch["sample_index"].tolist())
        bytes_read += int(batch["byte_size"].sum())
        failed_samples += int(batch["failed"].sum())
        batches_measured += 1
        if "shard_opened" in batch:
            shard_opens += int(batch["shard_opened"].sum())
            shard_open_seconds += float(batch["shard_open_seconds"].sum())
    measured_seconds = time.perf_counter() - measured_start

    # Child processes are counted while the workers are still alive.
    child_processes = env.child_process_count()
    user_cpu = env.cpu_seconds()[0] - cpu_before[0]
    system_cpu = env.cpu_seconds()[1] - cpu_before[1]
    switches_after = env.context_switches()
    voluntary = switches_after[0] - switches_before[0]
    involuntary = switches_after[1] - switches_before[1]

    # Whatever epoch measurement stopped inside is partial by construction, except in
    # epoch mode where the loop stops exactly on a boundary.
    if not epoch_mode:
        accounting.end_epoch(complete=False)
    samples_measured = accounting.total_observed
    total_wait = sum(wait_times)

    if collect_indices is not None:
        collect_indices.extend(accounting.observed_indices)

    cpus = env.cpus_available()
    # Fraction of the allocated CPUs actually kept busy. Above ~1.0 means the
    # process tree used more CPU than was allocated, which on a shared node means
    # it was competing with itself.
    utilization = (
        (user_cpu + system_cpu) / (measured_seconds * cpus)
        if measured_seconds > 0 and cpus > 0
        else 0.0
    )
    # Processes wanting CPU per allocated core. Above 1.0 the pipeline cannot all
    # run at once, whatever num_workers claims.
    oversubscription = (config.loader.num_workers + 1) / cpus if cpus > 0 else 0.0

    return new_run_summary(
        run_name=config.run.name,
        timestamp_utc=env.timestamp_utc(),
        git_commit=env.git_commit(repo_root),
        config_path=config.source_path,
        config_hash=config.config_hash(),
        hostname=env.hostname(),
        slurm_job_id=env.slurm_context()["SLURM_JOB_ID"],
        layout=config.dataset.layout,
        storage=config.storage.location,
        world_size=world_size,
        rank=rank,
        num_workers=config.loader.num_workers,
        warmup_batches=config.run.warmup_batches,
        measured_batches=config.run.measured_batches,
        samples_measured=samples_measured,
        bytes_read=bytes_read,
        startup_seconds=startup_seconds,
        staging_seconds=float(layout_metrics.get("staging_seconds", 0.0)),
        measured_seconds=measured_seconds,
        samples_per_second=metrics.per_second(samples_measured, measured_seconds),
        mib_per_second=metrics.mib_per_second(bytes_read, measured_seconds),
        mean_batch_wait_seconds=metrics.mean(wait_times),
        p95_batch_wait_seconds=metrics.percentile(wait_times, 95.0),
        mean_data_wait_fraction=(
            total_wait / measured_seconds if measured_seconds > 0 else 0.0
        ),
        peak_memory_bytes=env.peak_memory_bytes(),
        failed_samples=failed_samples,
        duplicate_samples=accounting.duplicate_samples,
        missing_samples=accounting.missing_samples,
        # Optional detail.
        manifest_path=str(config.manifest_path),
        manifest_hash=manifest_hash(config.manifest_path),
        resolved_config=config.resolved,
        slurm_context=env.slurm_context(),
        cpus_available=env.cpus_available(),
        batch_size=config.loader.batch_size,
        prefetch_factor=config.loader.prefetch_factor,
        persistent_workers=config.loader.persistent_workers,
        shuffle=config.loader.shuffle,
        seed=config.run.seed,
        compute_steps=config.loader.compute_steps,
        batches_measured=batches_measured,
        # What the layout actually opened to serve those samples: one file per
        # sample for a loose tree, one file per shard for a streaming layout. This
        # is the difference the packaging comparison is about.
        files_opened=shard_opens if streaming else samples_measured,
        batches_per_second=metrics.per_second(batches_measured, measured_seconds),
        median_batch_wait_seconds=metrics.percentile(wait_times, 50.0),
        max_batch_wait_seconds=max(wait_times) if wait_times else 0.0,
        compute_seconds=compute_seconds,
        warmup_seconds=warmup_seconds,
        measured_epochs=accounting.complete_epochs,
        unique_samples=accounting.unique_samples,
        notes=_accounting_note(accounting),
        total_samples=len(samples),
        # Derived from steady-state throughput rather than timed directly: the
        # measured window is a fixed number of batches, which rarely lands on an
        # epoch boundary. Break-even arithmetic needs a per-epoch cost, and this is
        # the honest way to get one from a partial window.
        estimated_epoch_seconds=(
            len(samples) / metrics.per_second(samples_measured, measured_seconds)
            if samples_measured and measured_seconds > 0
            else 0.0
        ),
        shuffle_buffer=config.loader.shuffle_buffer,
        shard_opens=shard_opens,
        shard_open_seconds=shard_open_seconds,
        # CPU and scheduling detail: what Part IV reads to tell a pipeline that
        # needs more workers from one that already has too many.
        user_cpu_seconds=user_cpu,
        system_cpu_seconds=system_cpu,
        cpu_utilization=utilization,
        voluntary_context_switches=voluntary,
        involuntary_context_switches=involuntary,
        involuntary_switches_per_second=metrics.per_second(involuntary, measured_seconds),
        child_processes=child_processes,
        oversubscription_ratio=oversubscription,
        # staging_seconds is passed explicitly above, so it must not arrive twice.
        **{k: v for k, v in layout_metrics.items() if k != "staging_seconds"},
    )


def _epoch_cycling_batches(loader: DataLoader) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(epoch_index, batch)`` indefinitely, restarting at each epoch end.

    A fixed number of measured batches is easier to compare than a fixed number
    of epochs, and small datasets would otherwise run out. The cost of restarting
    an epoch stays inside the batch wait time on purpose: it is a real cost the
    workload pays. The epoch index is what lets duplicate and missing samples be
    judged per epoch rather than over an arbitrary window.
    """
    epoch = 0
    while True:
        produced = False
        try:
            for batch in loader:
                produced = True
                yield epoch, batch
        except RuntimeError as exc:
            message = str(exc)
            if "worker" in message.lower() or "DataLoader" in message:
                raise WorkerFailure(
                    "A DataLoader worker process died during the measurement.\n"
                    f"PyTorch reported: {message}\n\n"
                    "Each worker is a separate process. The usual causes are the "
                    "job running out of its memory allocation (raise --mem, or "
                    "lower loader.num_workers and loader.prefetch_factor), or more "
                    "workers than allocated CPUs. This run produced no result."
                ) from exc
            raise
        if not produced:
            return
        epoch += 1


def _accounting_note(accounting: SampleAccounting) -> str:
    if accounting.complete_epochs == 0:
        return (
            "Measured window covered a partial epoch, so missing_samples is not "
            "evaluated. duplicate_samples still counts repeats within the window."
        )
    return (
        f"Measured window covered {accounting.complete_epochs} complete epoch(s); "
        "missing_samples is evaluated over those epochs only."
    )
