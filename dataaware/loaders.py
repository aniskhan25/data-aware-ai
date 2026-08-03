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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import env, metrics
from .config import Config
from .manifest import Sample, checksum_bytes, manifest_hash, read_manifest
from .schema import new_run_summary


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


@dataclass
class SampleAccounting:
    """Tracks which samples a run actually read.

    Duplicates are counted per epoch, because a correct sampler visits each
    sample at most once per epoch. Missing samples are only counted for epochs
    that were fully traversed inside the measured window: a partial epoch says
    nothing about coverage, and reporting it as missing data would be wrong.
    """

    dataset_size: int
    batch_size: int
    drop_last: bool

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
            self.missing_samples += max(0, self._expected_coverage() - unique)
        else:
            self.partial_epoch = True
        self._epoch_seen = {}

    def _expected_coverage(self) -> int:
        """Samples a correct epoch should cover, honouring ``drop_last``."""
        if self.drop_last:
            return (self.dataset_size // self.batch_size) * self.batch_size
        return self.dataset_size

    @property
    def unique_samples(self) -> int:
        return len(self._all_seen)


def build_dataset(config: Config, samples: Sequence[Sample]) -> Dataset:
    """Construct the dataset for the configured layout."""
    if config.dataset.layout == "loose-files":
        return LooseFileDataset(config.dataset_root, samples)
    raise NotImplementedError(
        f"dataset.layout '{config.dataset.layout}' is not implemented in this "
        "release. Only 'loose-files' is available; SquashFS and WebDataset "
        "layouts arrive with Part III of the tutorial."
    )


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


def run_loader_benchmark(config: Config, repo_root: Path | None = None) -> dict[str, Any]:
    """Run one measured loading experiment and return its run summary."""
    if config.distributed.enabled:
        raise NotImplementedError(
            "distributed.enabled requires the rank-aware loader from Part VI, "
            "which is not in this release"
        )

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

    dataset = build_dataset(config, samples)
    loader = build_dataloader(config, dataset)
    accounting = SampleAccounting(
        dataset_size=len(samples),
        batch_size=config.loader.batch_size,
        drop_last=config.loader.drop_last,
    )

    batches = _epoch_cycling_batches(loader)

    # Startup covers worker creation and the first batch, which is then discarded.
    # It is reported separately rather than folded into throughput, because a
    # layout can trade startup cost against steady-state speed.
    startup_start = time.perf_counter()
    if next(batches, None) is None:
        raise ValueError("the DataLoader produced no batches")
    startup_seconds = time.perf_counter() - startup_start

    warmup_start = time.perf_counter()
    for _ in range(config.run.warmup_batches):
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
    # The epoch in progress when measurement began was partly consumed by startup
    # and warm-up, so its coverage cannot be judged.
    entry_epoch: int | None = None
    current_epoch: int | None = None

    measured_start = time.perf_counter()
    for _ in range(config.run.measured_batches):
        wait_start = time.perf_counter()
        item = next(batches, None)
        wait = time.perf_counter() - wait_start
        if item is None:
            break
        epoch, batch = item
        wait_times.append(wait)

        if current_epoch is None:
            entry_epoch = current_epoch = epoch
        elif epoch != current_epoch:
            # The previous epoch finished. Coverage can only be judged for an
            # epoch measurement saw from its very first batch, which excludes the
            # one already in progress when the measured window opened.
            accounting.end_epoch(complete=current_epoch != entry_epoch)
            current_epoch = epoch

        compute_start = time.perf_counter()
        synthetic_compute(batch["image"], config.loader.compute_steps)
        compute_seconds += time.perf_counter() - compute_start

        accounting.observe(batch["sample_index"].tolist())
        bytes_read += int(batch["byte_size"].sum())
        failed_samples += int(batch["failed"].sum())
        batches_measured += 1
    measured_seconds = time.perf_counter() - measured_start

    # Whatever epoch measurement stopped inside is partial by construction.
    accounting.end_epoch(complete=False)
    samples_measured = accounting.total_observed
    total_wait = sum(wait_times)

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
        world_size=1,
        num_workers=config.loader.num_workers,
        warmup_batches=config.run.warmup_batches,
        measured_batches=config.run.measured_batches,
        samples_measured=samples_measured,
        bytes_read=bytes_read,
        startup_seconds=startup_seconds,
        staging_seconds=0.0,
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
        files_opened=samples_measured,
        batches_per_second=metrics.per_second(batches_measured, measured_seconds),
        median_batch_wait_seconds=metrics.percentile(wait_times, 50.0),
        max_batch_wait_seconds=max(wait_times) if wait_times else 0.0,
        compute_seconds=compute_seconds,
        warmup_seconds=warmup_seconds,
        unique_samples=accounting.unique_samples,
        notes=_accounting_note(accounting),
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
        for batch in loader:
            produced = True
            yield epoch, batch
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
