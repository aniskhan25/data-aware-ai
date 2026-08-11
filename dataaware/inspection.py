"""Characterise a dataset on disk before allocating any GPUs.

This is the cheapest experiment in the tutorial: it reads metadata, never file
contents, and needs no GPU, no PyTorch, and no job allocation beyond a login-node
shell for small trees.

What it does
------------
Walks a directory tree once and reports file counts, the file-size distribution,
extension mix, directory shape, and packaging arithmetic. From those it proposes
*candidate next experiments*.

What it deliberately does not do
--------------------------------
It does not choose a format. It cannot see application semantics: whether random
access is required, whether sample order matters, whether records are mutable,
whether an ecosystem mandates a format, or how expensive a record is to decode.
Every suggestion it makes is a hypothesis to be measured, not a decision.

Determinism
-----------
Every field outside the ``provenance`` block is a deterministic function of the
tree being inspected and the configured thresholds. ``provenance`` holds the
timestamp, hostname, and walk duration, which are not.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import env, metrics
from .errors import DataError

INSPECTION_SCHEMA_VERSION = "1.0"

GIB = 1024**3

#: Files below this size pay a large share of their cost in per-file overhead
#: rather than in reading bytes. Configurable; this is a starting point, not a
#: property of the filesystem.
DEFAULT_SMALL_FILE_BYTES = 64 * 1024

#: Size thresholds reported as a coarse histogram.
DEFAULT_THRESHOLDS = (4 * 1024, 64 * 1024, 1024 * 1024)

#: A tree needs to be large enough for per-file overhead to matter in aggregate
#: before packaging is worth measuring.
MANY_FILES_TRIGGER = 10_000

#: Fraction of files below the small-file threshold that makes a tree
#: "metadata-heavy" enough to be worth a packaging experiment.
SMALL_FILE_FRACTION_TRIGGER = 0.5

#: Ratio of the 95th percentile file size to the median, above which equal sample
#: counts per shard would not mean equal work per shard.
#:
#: A ratio is used rather than the coefficient of variation because CV is dominated
#: by a handful of outliers: one stray manifest or checkpoint among uniform samples
#: pushes CV above any fixed threshold while the bulk of the distribution is tight.
#: The CV is still reported, it just does not drive the suggestion.
SIZE_RATIO_TRIGGER = 4.0

#: Node-local /tmp is memory. Staging is only proposed when the dataset fits well
#: inside this fraction of the allocation, leaving room for the workload itself.
DEFAULT_TMP_SAFETY_FRACTION = 0.5

#: Target size for a tar shard. Large enough that sequential reads dominate,
#: small enough that many shards exist for many readers.
DEFAULT_TARGET_SHARD_BYTES = 512 * 1024 * 1024

#: Fewest shards worth suggesting, whatever the target size implies.
#:
#: A size-based target alone gives bad advice for small datasets: the 512 MiB default
#: target on a 143 MiB dataset works out to one shard, and one shard cannot feed more
#: than one reader. A full LUMI-G node has eight GCDs, so a dataset that cannot be
#: split at least eight ways cannot keep one node busy however fast the storage is.
#:
#: This is a floor for the *suggestion*, not a recommendation to stop there: shards
#: must be at least as numerous as the readers you actually intend to run.
DEFAULT_MIN_SHARDS = 8

#: Extensions whose bytes are already compressed, so packaging or compressing
#: them again saves little.
_COMPRESSED_EXTENSIONS = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".heic",
        ".mp3", ".flac", ".ogg", ".opus", ".m4a",
        ".mp4", ".mkv", ".webm", ".mov",
        ".gz", ".bz2", ".xz", ".zst", ".zip", ".7z", ".lz4",
        ".parquet", ".orc",
    }
)

#: Extension -> optional format track worth reading about. These are hints from
#: file names only; the tool cannot confirm how the data is actually used.
_FORMAT_HINTS = {
    ".parquet": "Tabular or record-oriented data. See the Parquet track.",
    ".arrow": "Arrow-backed data. See the Hugging Face Datasets track.",
    ".h5": "Dense arrays. See the HDF5 track.",
    ".hdf5": "Dense arrays. See the HDF5 track.",
    ".nc": "Scientific multidimensional data (NetCDF).",
    ".zarr": "Chunked arrays (Zarr). Check chunk sizing before anything else.",
    ".npy": "NumPy intermediates. Fine when small, hard to manage at scale.",
    ".npz": "NumPy intermediates. Fine when small, hard to manage at scale.",
    ".tar": "Already tar-packaged. A streaming layout may already be available.",
    ".squashfs": "Already a SquashFS image.",
    ".sqfs": "Already a SquashFS image.",
    ".tfrecord": "TFRecord shards. A streaming layout is already in use.",
    ".sqlite": "Indexed local access. Test concurrency before scaling readers.",
    ".mdb": "LMDB. Test concurrency before scaling readers.",
}

_MAX_EXTENSIONS_REPORTED = 12


@dataclass
class WalkResult:
    """Raw counts from one pass over the tree."""

    total_files: int = 0
    total_bytes: int = 0
    directories: int = 0
    symlinks: int = 0
    other_entries: int = 0
    hardlinked_files: int = 0
    unreadable_directories: int = 0
    unreadable_files: int = 0
    max_depth: int = 0
    max_files_in_one_directory: int = 0
    max_files_directory: str = ""
    file_sizes: list[int] = field(default_factory=list)
    extension_files: dict[str, int] = field(default_factory=dict)
    extension_bytes: dict[str, int] = field(default_factory=dict)


def walk_tree(
    root: Path,
    progress_every: int = 0,
) -> WalkResult:
    """Walk ``root`` once, collecting metadata only.

    Directory symlinks are never followed, so a cyclic tree cannot trap the walk.
    Unreadable directories and files are counted and skipped: a permission problem
    somewhere in a large tree should not discard the rest of the measurement.
    """
    result = WalkResult()
    # An explicit stack rather than recursion: dataset trees can be deep, and this
    # keeps the traversal order and the depth bookkeeping obvious.
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        directory, depth = stack.pop()
        result.max_depth = max(result.max_depth, depth)
        # Counted on visit, so the root is included and the count matches the
        # number of directories the walk actually accounted for.
        result.directories += 1
        try:
            with os.scandir(directory) as entries:
                listing = list(entries)
        except OSError:
            result.unreadable_directories += 1
            continue

        files_here = 0
        for entry in listing:
            try:
                if entry.is_symlink():
                    # Counted but not followed, in either direction.
                    result.symlinks += 1
                elif entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
                elif entry.is_file(follow_symlinks=False):
                    info = entry.stat(follow_symlinks=False)
                    files_here += 1
                    result.total_files += 1
                    result.total_bytes += info.st_size
                    result.file_sizes.append(info.st_size)
                    if info.st_nlink > 1:
                        result.hardlinked_files += 1
                    extension = _extension_of(entry.name)
                    result.extension_files[extension] = (
                        result.extension_files.get(extension, 0) + 1
                    )
                    result.extension_bytes[extension] = (
                        result.extension_bytes.get(extension, 0) + info.st_size
                    )
                    if progress_every and result.total_files % progress_every == 0:
                        print(f"scanned {result.total_files} files", flush=True)
                else:
                    result.other_entries += 1
            except OSError:
                result.unreadable_files += 1

        if files_here > result.max_files_in_one_directory:
            result.max_files_in_one_directory = files_here
            result.max_files_directory = str(directory)

    return result


def inspect_path(
    path: str | os.PathLike[str],
    small_file_bytes: int = DEFAULT_SMALL_FILE_BYTES,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
    memory_bytes: int | None = None,
    tmp_safety_fraction: float = DEFAULT_TMP_SAFETY_FRACTION,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Inspect a dataset directory and return a report dictionary."""
    root = Path(path)
    if not root.exists():
        raise DataError(f"path does not exist: {root}")

    started = time.perf_counter()
    if root.is_dir():
        walk = walk_tree(root, progress_every=progress_every)
    else:
        # Inspecting a single file is legitimate: it is how you check an archive
        # or a packaged image that someone handed you.
        walk = _single_file_walk(root)
    walk_seconds = time.perf_counter() - started

    if memory_bytes is None:
        memory_bytes, memory_source = detect_allocated_memory()
    else:
        memory_source = "explicit"

    report = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "dataset_path": str(root),
        "settings": {
            "small_file_bytes": small_file_bytes,
            "thresholds_bytes": sorted(set(int(t) for t in thresholds)),
            "tmp_safety_fraction": tmp_safety_fraction,
            "target_shard_bytes": target_shard_bytes,
        },
        "tree": _tree_section(walk),
        "file_sizes": _size_section(walk),
        "size_thresholds": _threshold_section(walk, thresholds),
        "small_files": _small_file_section(walk, small_file_bytes),
        "extensions": _extension_section(walk),
        "directories": _directory_section(walk),
        "packaging": _packaging_section(walk, target_shard_bytes),
        "memory": _memory_section(walk, memory_bytes, memory_source, tmp_safety_fraction),
        "format_hints": _format_hints(walk),
        "limitations": list(LIMITATIONS),
        "provenance": {
            "generated_utc": env.timestamp_utc(),
            "hostname": env.hostname(),
            "walk_seconds": walk_seconds,
            "slurm_job_id": env.slurm_context()["SLURM_JOB_ID"],
        },
    }
    report["candidates"] = _candidates(report)
    return report


#: Stated in every report, so a reader cannot mistake a suggestion for a finding.
LIMITATIONS = (
    "This tool reads metadata only. It never opens a file, so it cannot know how "
    "expensive a record is to decode.",
    "It cannot tell whether the workload needs random access by path, or whether "
    "a stream of samples would do.",
    "It cannot tell whether sample order matters, or whether records are mutable.",
    "It cannot tell whether an ecosystem or library mandates a particular format.",
    "Extension-based hints describe file names, not how the data is actually used.",
    "Every candidate below is a hypothesis to be measured, not a decision.",
)


def _single_file_walk(root: Path) -> WalkResult:
    info = root.stat()
    extension = _extension_of(root.name)
    return WalkResult(
        total_files=1,
        total_bytes=info.st_size,
        file_sizes=[info.st_size],
        extension_files={extension: 1},
        extension_bytes={extension: info.st_size},
        max_files_in_one_directory=1,
        max_files_directory=str(root.parent),
    )


def _extension_of(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return suffix if suffix else "(none)"


def _tree_section(walk: WalkResult) -> dict[str, Any]:
    return {
        "total_files": walk.total_files,
        "total_bytes": walk.total_bytes,
        "total_gib": walk.total_bytes / GIB,
        "directories": walk.directories,
        "symlinks": walk.symlinks,
        "other_entries": walk.other_entries,
        "hardlinked_files": walk.hardlinked_files,
        "unreadable_directories": walk.unreadable_directories,
        "unreadable_files": walk.unreadable_files,
        # One object per file plus one per directory. This is the number packaging
        # would collapse, and the reason small-file trees pressure metadata.
        "filesystem_objects": walk.total_files + walk.directories,
    }


def _size_section(walk: WalkResult) -> dict[str, Any]:
    sizes = walk.file_sizes
    if not sizes:
        return {
            "min_bytes": 0,
            "p5_bytes": 0.0,
            "median_bytes": 0.0,
            "mean_bytes": 0.0,
            "p95_bytes": 0.0,
            "max_bytes": 0,
            "coefficient_of_variation": 0.0,
            "p95_to_median_ratio": 0.0,
        }
    median = metrics.percentile(sizes, 50.0)
    p95 = metrics.percentile(sizes, 95.0)
    return {
        "min_bytes": min(sizes),
        "p5_bytes": metrics.percentile(sizes, 5.0),
        "median_bytes": median,
        "mean_bytes": metrics.mean(sizes),
        "p95_bytes": p95,
        "max_bytes": max(sizes),
        # Reported for completeness, but outlier-sensitive; see SIZE_RATIO_TRIGGER.
        "coefficient_of_variation": metrics.coefficient_of_variation(sizes),
        # Robust spread: unaffected by a few very large or very small files.
        "p95_to_median_ratio": (p95 / median) if median > 0 else 0.0,
    }


def _threshold_section(walk: WalkResult, thresholds: Iterable[int]) -> list[dict[str, Any]]:
    sizes = walk.file_sizes
    rows = []
    for threshold in sorted(set(int(t) for t in thresholds)):
        count = sum(1 for size in sizes if size < threshold)
        rows.append(
            {
                "bytes": threshold,
                "files": count,
                "fraction": count / len(sizes) if sizes else 0.0,
            }
        )
    return rows


def _small_file_section(walk: WalkResult, small_file_bytes: int) -> dict[str, Any]:
    count = sum(1 for size in walk.file_sizes if size < small_file_bytes)
    fraction = count / walk.total_files if walk.total_files else 0.0
    return {
        "threshold_bytes": small_file_bytes,
        "files": count,
        "fraction": fraction,
    }


def _extension_section(walk: WalkResult) -> list[dict[str, Any]]:
    """Extension mix, ordered by file count then name so output is stable."""
    total = walk.total_files or 1
    ordered = sorted(
        walk.extension_files.items(), key=lambda item: (-item[1], item[0])
    )
    rows = [
        {
            "extension": extension,
            "files": count,
            "bytes": walk.extension_bytes.get(extension, 0),
            "fraction": count / total,
        }
        for extension, count in ordered[:_MAX_EXTENSIONS_REPORTED]
    ]
    remaining = ordered[_MAX_EXTENSIONS_REPORTED:]
    if remaining:
        rows.append(
            {
                "extension": "(other)",
                "files": sum(count for _, count in remaining),
                "bytes": sum(walk.extension_bytes.get(ext, 0) for ext, _ in remaining),
                "fraction": sum(count for _, count in remaining) / total,
            }
        )
    return rows


def _directory_section(walk: WalkResult) -> dict[str, Any]:
    return {
        "count": walk.directories,
        "max_depth": walk.max_depth,
        "max_files_in_one_directory": walk.max_files_in_one_directory,
        "max_files_directory": walk.max_files_directory,
        "mean_files_per_directory": (
            walk.total_files / walk.directories if walk.directories else float(walk.total_files)
        ),
    }


def _packaging_section(walk: WalkResult, target_shard_bytes: int) -> dict[str, Any]:
    compressed_bytes = sum(
        byte_count
        for extension, byte_count in walk.extension_bytes.items()
        if extension in _COMPRESSED_EXTENSIONS
    )
    compressed_fraction = (
        compressed_bytes / walk.total_bytes if walk.total_bytes else 0.0
    )
    size_based_shards = (
        max(1, -(-walk.total_bytes // target_shard_bytes)) if walk.total_bytes else 0
    )
    # Never suggest fewer shards than could keep one node's readers busy, and never
    # more shards than there are samples to put in them.
    shards = (
        min(max(size_based_shards, DEFAULT_MIN_SHARDS), walk.total_files)
        if walk.total_files
        else 0
    )
    return {
        "filesystem_objects_now": walk.total_files + walk.directories,
        # A SquashFS image is one file on the parallel filesystem regardless of how
        # many files it contains. That is the operational argument for it.
        "filesystem_objects_as_squashfs": 1 if walk.total_files else 0,
        "already_compressed_byte_fraction": compressed_fraction,
        # Re-compressing already-compressed bytes buys little, so a packaged image
        # of a JPEG tree is roughly the size of the tree.
        "compression_likely_to_help": compressed_fraction < 0.5,
        "estimated_packaged_bytes": walk.total_bytes,
        "target_shard_bytes": target_shard_bytes,
        # What the target size alone implies, kept so the floor below is visible
        # rather than looking like arithmetic that does not add up.
        "size_based_shards": size_based_shards,
        "min_shards": DEFAULT_MIN_SHARDS,
        "suggested_shards": shards,
        "suggested_samples_per_shard": (
            walk.total_files // shards if shards else 0
        ),
    }


def detect_allocated_memory() -> tuple[int | None, str]:
    """Best effort at the memory this job may use, and where the number came from.

    Returns ``(None, reason)`` when it cannot be determined, which is the honest
    answer on a login node or laptop. Node-local ``/tmp`` staging advice depends on
    this number, so guessing would be worse than declining to answer.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node and per_node.isdigit():
        return int(per_node) * 1024 * 1024, "SLURM_MEM_PER_NODE"

    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus) * 1024 * 1024, "SLURM_MEM_PER_CPU x SLURM_CPUS_PER_TASK"

    return None, "unknown (not running under a Slurm allocation)"


def _memory_section(
    walk: WalkResult,
    memory_bytes: int | None,
    memory_source: str,
    safety_fraction: float,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "allocated_bytes": memory_bytes,
        "source": memory_source,
        "safety_fraction": safety_fraction,
        "dataset_fraction_of_memory": None,
        "tmp_staging_within_safety_margin": None,
        "note": "",
    }
    if memory_bytes is None:
        section["note"] = (
            "Allocated memory is unknown, so no node-local staging advice is given. "
            "Re-run inside the job allocation you intend to use, or pass "
            "--memory-bytes."
        )
        return section
    if memory_bytes <= 0:
        section["note"] = "Reported allocated memory was not positive; ignoring it."
        return section

    fraction = walk.total_bytes / memory_bytes
    section["dataset_fraction_of_memory"] = fraction
    section["tmp_staging_within_safety_margin"] = fraction <= safety_fraction
    section["note"] = (
        "Compute-node /tmp is memory and is charged against this allocation, so a "
        "staged dataset competes with the workload itself."
    )
    return section


def _format_hints(walk: WalkResult) -> list[dict[str, str]]:
    hints = []
    for extension in sorted(walk.extension_files):
        if extension in _FORMAT_HINTS:
            hints.append({"extension": extension, "hint": _FORMAT_HINTS[extension]})
    return hints


def _shard_floor_note(packaging: dict[str, Any]) -> str:
    """Explain the shard count when the size target was not what decided it."""
    if packaging["suggested_shards"] > packaging["size_based_shards"]:
        return (
            f"a {packaging['target_shard_bytes']}-byte target alone would give only "
            f"{packaging['size_based_shards']}, too few to feed one node"
        )
    return f"from a {packaging['target_shard_bytes']}-byte shard target"


def _candidates(report: dict[str, Any]) -> list[dict[str, str]]:
    """Propose next experiments, each with the observation that motivates it.

    Order matters: the baseline always comes first, because a comparison needs
    something to compare against.
    """
    tree = report["tree"]
    sizes = report["file_sizes"]
    small = report["small_files"]
    memory = report["memory"]
    packaging = report["packaging"]

    if tree["total_files"] == 0:
        return [
            {
                "experiment": "none",
                "reason": (
                    "No files were found. Check the path, and check whether the "
                    "dataset still needs to be downloaded or generated."
                ),
            }
        ]

    candidates = [
        {
            "experiment": "loose-file-baseline",
            "reason": (
                "Always measure the unmodified dataset first. Every later "
                "comparison needs this reference point (Part II)."
            ),
        }
    ]

    many_files = tree["total_files"] >= MANY_FILES_TRIGGER
    mostly_small = small["fraction"] >= SMALL_FILE_FRACTION_TRIGGER

    if many_files and mostly_small:
        candidates.append(
            {
                "experiment": "squashfs",
                "reason": (
                    f"{small['files']} of {tree['total_files']} files are under "
                    f"{small['threshold_bytes']} bytes "
                    f"({small['fraction']:.0%}). Packaging would present "
                    f"{packaging['filesystem_objects_now']} filesystem objects as 1 "
                    "while preserving ordinary paths (Part III)."
                ),
            }
        )
        candidates.append(
            {
                "experiment": "webdataset",
                "reason": (
                    "If the samples are independent and can be consumed as a "
                    "stream, tar shards also give explicit rank-aware assignment. "
                    f"Start near {packaging['suggested_shards']} shards of about "
                    f"{packaging['suggested_samples_per_shard']} samples "
                    f"({_shard_floor_note(packaging)}). Shards must be at least as "
                    "numerous as the readers you intend to run, or some sit idle "
                    "(Parts III and VI)."
                ),
            }
        )
    elif many_files:
        candidates.append(
            {
                "experiment": "webdataset",
                "reason": (
                    f"{tree['total_files']} files is enough for per-file overhead "
                    "to matter in aggregate, but they are not mostly small. Compare "
                    "streaming shards against the baseline (Part III)."
                ),
            }
        )
    else:
        candidates.append(
            {
                "experiment": "benchmark-native-representation",
                "reason": (
                    f"Only {tree['total_files']} files, median "
                    f"{sizes['median_bytes']:.0f} bytes. Packaging a small number of "
                    "files rarely pays; measure the dataset as it is."
                ),
            }
        )

    if sizes["p95_to_median_ratio"] >= SIZE_RATIO_TRIGGER:
        candidates.append(
            {
                "experiment": "shard-balancing",
                "reason": (
                    "File sizes vary widely: the 95th percentile is "
                    f"{sizes['p95_to_median_ratio']:.1f}x the median "
                    f"({sizes['p95_bytes']:.0f} against "
                    f"{sizes['median_bytes']:.0f} bytes), so equal sample counts "
                    "per shard would not mean equal work per shard (Part VI)."
                ),
            }
        )

    if memory["tmp_staging_within_safety_margin"]:
        candidates.append(
            {
                "experiment": "tmp-staging",
                "reason": (
                    f"The dataset is {memory['dataset_fraction_of_memory']:.0%} of "
                    "allocated memory, within the safety margin, so node-local "
                    "staging is worth testing. Include the copy cost and compute "
                    "break-even epochs (Part V)."
                ),
            }
        )
    elif memory["tmp_staging_within_safety_margin"] is False:
        candidates.append(
            {
                "experiment": "avoid-tmp-staging",
                "reason": (
                    f"The dataset is {memory['dataset_fraction_of_memory']:.0%} of "
                    "allocated memory. Staging it to node-local /tmp would spend "
                    "memory the workload needs (Part V)."
                ),
            }
        )

    if tree["unreadable_directories"] or tree["unreadable_files"]:
        candidates.append(
            {
                "experiment": "fix-permissions-first",
                "reason": (
                    f"{tree['unreadable_directories']} directory and "
                    f"{tree['unreadable_files']} file entries could not be read. "
                    "The numbers above cover only what was readable."
                ),
            }
        )

    return candidates


def format_keyvalue(report: dict[str, Any]) -> str:
    """Render the headline findings as ``KEY=VALUE`` lines.

    Deliberately short. These are the numbers that change what you do next; the
    JSON report holds the rest.
    """
    tree = report["tree"]
    sizes = report["file_sizes"]
    memory = report["memory"]

    lines = [
        f"DATASET_PATH={report['dataset_path']}",
        f"TOTAL_FILES={tree['total_files']}",
        f"TOTAL_GIB={tree['total_gib']:.4g}",
        f"MEDIAN_FILE_BYTES={sizes['median_bytes']:.0f}",
        f"P95_TO_MEDIAN_RATIO={sizes['p95_to_median_ratio']:.4g}",
        f"SMALL_FILE_FRACTION={report['small_files']['fraction']:.4g}",
        f"FILESYSTEM_OBJECTS={tree['filesystem_objects']}",
        f"MAX_FILES_IN_ONE_DIRECTORY={report['directories']['max_files_in_one_directory']}",
    ]
    if memory["dataset_fraction_of_memory"] is not None:
        lines.append(
            f"DATASET_FRACTION_OF_MEMORY={memory['dataset_fraction_of_memory']:.4g}"
        )
    # Only worth a line when there is something to act on.
    for key in ("symlinks", "unreadable_directories", "unreadable_files"):
        if tree[key]:
            lines.append(f"{key.upper()}={tree[key]}")
    lines.append(
        "CANDIDATE_EXPERIMENTS="
        + ",".join(candidate["experiment"] for candidate in report["candidates"])
    )
    return "\n".join(lines)
