"""SquashFS packaging: build an image, and make it readable.

The point of SquashFS in this tutorial is narrow and worth stating plainly: it
turns a read-only tree of many files into **one file** on the parallel filesystem,
while the application keeps using ordinary paths. Nothing in the loader changes —
only where ``dataset.root`` points and how many objects the filesystem has to
track.

Two ways to make an image readable, both supported:

``prebound``
    The image is already mounted or bound at ``dataset.root``. This is what a
    container bind produces, and it is the path LUMI documentation points to.
    Nothing here needs to run.

``squashfuse``
    Mount the image in userspace with ``squashfuse`` and unmount it afterwards.
    Useful outside a container. Requires ``squashfuse`` on PATH.

Building needs ``mksquashfs``. Neither tool is a Python dependency, so both are
probed for and reported clearly when absent rather than failing deep inside a job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

#: Default compression for the tutorial's image. The dataset is JPEG and PNG, whose
#: bytes are already compressed, so compressing them again spends CPU on every read
#: for almost no space saved.
#:
#: Both ``-noD`` and ``-noF`` are needed, and the second one is easy to miss.
#: ``-noD`` only disables compression of full *data blocks*; files smaller than the
#: block size are stored as *fragments*, which ``-noF`` covers. For a metadata-heavy
#: dataset almost every file is a fragment, so ``-noD`` alone leaves the image fully
#: compressed — measured on LUMI, a 50 000-file tree of 2.7 KB JPEGs still shrank to
#: 79 % of its source size and took five minutes to pack.
#:
#: The inode table is left compressed: it is small and is read once at mount, so it
#: costs nothing per sample.
#:
#: For uncompressed source data (.npy, .csv, .bin) prefer ``-comp zstd``, and expect
#: a smaller image at the cost of decompression work while reading.
DEFAULT_MKSQUASHFS_ARGS = ("-noD", "-noF", "-no-xattrs", "-no-progress")


class SquashFSError(RuntimeError):
    """Raised when an image cannot be built, mounted, or verified."""


def have_mksquashfs() -> bool:
    return shutil.which("mksquashfs") is not None


def have_squashfuse() -> bool:
    return shutil.which("squashfuse") is not None


def build_image(
    source_root: str | os.PathLike[str],
    image_path: str | os.PathLike[str],
    extra_args: tuple[str, ...] = DEFAULT_MKSQUASHFS_ARGS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build a SquashFS image from a directory tree.

    Returns build statistics including the resulting image size, which is what the
    packaging comparison reports against the size of the source tree.
    """
    source_root = Path(source_root)
    image_path = Path(image_path)

    if not source_root.is_dir():
        raise SquashFSError(f"source is not a directory: {source_root}")
    if not have_mksquashfs():
        raise SquashFSError(
            "mksquashfs was not found on PATH.\n"
            "On LUMI it is provided by the squashfs-tools module or inside a "
            "container; see https://docs.lumi-supercomputer.eu/storage/formats/FUSE/"
        )
    if image_path.exists():
        if not overwrite:
            raise SquashFSError(
                f"{image_path} already exists; pass --overwrite to rebuild it"
            )
        image_path.unlink()

    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Build to a partial name and rename, so a cancelled job cannot leave a
    # truncated image that a later run would mount and measure.
    partial = image_path.with_suffix(image_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    command = ["mksquashfs", str(source_root), str(partial), *extra_args]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    build_seconds = time.perf_counter() - started

    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        raise SquashFSError(
            f"mksquashfs failed with exit code {completed.returncode}\n"
            f"command: {' '.join(command)}\n{completed.stderr.strip()}"
        )
    partial.replace(image_path)

    source_bytes = _tree_bytes(source_root)
    image_bytes = image_path.stat().st_size
    return {
        "image_path": str(image_path),
        "image_bytes": image_bytes,
        "source_bytes": source_bytes,
        "size_ratio": image_bytes / source_bytes if source_bytes else 0.0,
        "build_seconds": build_seconds,
        "command": " ".join(command),
    }


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                continue
    return total


@contextmanager
def mounted_image(
    image_path: str | os.PathLike[str],
    mount_point: str | os.PathLike[str] | None = None,
) -> Iterator[tuple[Path, float]]:
    """Mount an image with ``squashfuse``, and always unmount it.

    Yields ``(mount_point, mount_seconds)``. The unmount runs in a ``finally``
    block, so it happens whether the measurement succeeds, fails, or raises. A
    leaked FUSE mount would otherwise outlive the job and confuse the next one.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise SquashFSError(f"image not found: {image_path}")
    if not have_squashfuse():
        raise SquashFSError(
            "squashfuse was not found on PATH.\n"
            "Either install it, or use dataset.squashfs_mode: prebound with the "
            "image bound into your container."
        )

    created_mount_point = mount_point is None
    if mount_point is None:
        import tempfile

        mount_point = Path(tempfile.mkdtemp(prefix="daai-squashfs-"))
    else:
        mount_point = Path(mount_point)
        mount_point.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    completed = subprocess.run(
        ["squashfuse", str(image_path), str(mount_point)],
        capture_output=True,
        text=True,
    )
    mount_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        if created_mount_point:
            _remove_quietly(mount_point)
        raise SquashFSError(
            f"squashfuse failed with exit code {completed.returncode}\n"
            f"{completed.stderr.strip()}"
        )

    try:
        yield mount_point, mount_seconds
    finally:
        _unmount(mount_point)
        if created_mount_point:
            _remove_quietly(mount_point)


def _unmount(mount_point: Path) -> None:
    """Unmount, trying the portable tool first.

    Failure is reported but not raised: an unmount problem during cleanup should not
    mask whatever the job was actually doing.
    """
    for command in (
        ["fusermount", "-u", str(mount_point)],
        ["fusermount3", "-u", str(mount_point)],
        ["umount", str(mount_point)],
    ):
        if shutil.which(command[0]) is None:
            continue
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            return
    print(
        f"WARNING could not unmount {mount_point}; it may need manual cleanup",
        flush=True,
    )


def _remove_quietly(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
