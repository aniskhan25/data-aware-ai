#!/usr/bin/env python3
"""Package a loose-file dataset into one SquashFS image.

    python scripts/build_squashfs.py \\
        --source "$TUTORIAL_ROOT/source" \\
        --image "$TUTORIAL_ROOT/source.squashfs"

The image holds the same tree with the same paths, so the application keeps reading
ordinary filenames. What changes is the number of objects the parallel filesystem
tracks: one, instead of one per sample.

Requires ``mksquashfs``. Needs no PyTorch.

Compression: the default stores sample bytes uncompressed (``-noD -noF``), because
the tutorial dataset is JPEG and PNG whose bytes are already compressed — compressing
them again spends CPU on every read for almost nothing. Both flags matter: ``-noD``
covers full data blocks, while files smaller than the block size are stored as
fragments and need ``-noF``. In a small-file dataset nearly every file is a fragment,
so ``-noD`` alone leaves the image fully compressed.

For uncompressed source data, pass ``--mksquashfs-args '-comp zstd'`` and expect a
smaller image at the cost of decompression while reading.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.squashfs import (  # noqa: E402
    DEFAULT_MKSQUASHFS_ARGS,
    SquashFSError,
    build_image,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True, help="dataset tree to pack")
    parser.add_argument("--image", type=Path, required=True, help="image file to write")
    parser.add_argument(
        "--mksquashfs-args",
        default=" ".join(DEFAULT_MKSQUASHFS_ARGS),
        help="arguments passed to mksquashfs (default: %(default)r)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = build_image(
            source_root=args.source,
            image_path=args.image,
            extra_args=tuple(shlex.split(args.mksquashfs_args)),
            overwrite=args.overwrite,
        )
    except SquashFSError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    print(f"IMAGE_PATH={result['image_path']}")
    print(f"IMAGE_BYTES={result['image_bytes']}")
    print(f"SOURCE_BYTES={result['source_bytes']}")
    print(f"SIZE_RATIO={result['size_ratio']:.4g}")
    print(f"BUILD_SECONDS={result['build_seconds']:.4g}")
    print(f"COMMAND={result['command']}")
    print()
    print(
        "The image is one filesystem object. To read it, either bind it into your "
        "container and use dataset.squashfs_mode: prebound, or set "
        "dataset.squashfs_mode: squashfuse with dataset.image pointing here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
