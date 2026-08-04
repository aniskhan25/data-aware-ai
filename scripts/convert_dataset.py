#!/usr/bin/env python3
"""Convert the generated dataset into an optional format.

    python3 scripts/convert_dataset.py --to parquet \
        --source "$TUTORIAL_ROOT/source" \
        --manifest "$TUTORIAL_ROOT/source/manifest.jsonl" \
        --output "$TUTORIAL_ROOT/parquet"

Each track reads the same manifest and stores the same sample bytes, so a run against
the converted artifact is directly comparable with the core layouts from Part III — the
comparison tools check the manifest hash and will say so if it differs.

Optional dependencies are imported only by the track that needs them:

    pip install '.[parquet]'      pyarrow
    pip install '.[hdf5]'         h5py
    pip install '.[huggingface]'  datasets

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataaware.adapters import AdapterError  # noqa: E402
from dataaware.manifest import ManifestError, read_manifest  # noqa: E402

TRACKS = ("parquet", "hdf5", "huggingface")

ADAPTERS = {
    "parquet": "examples.parquet_track:ParquetAdapter",
    "hdf5": "examples.hdf5_track:HDF5Adapter",
    "huggingface": "examples.huggingface_track:HuggingFaceAdapter",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--to", dest="track", choices=TRACKS, required=True)
    parser.add_argument("--source", type=Path, required=True, help="loose-file dataset root")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-size",
        type=int,
        default=1000,
        help=(
            "samples per row group (Parquet), chunk (HDF5), or writer batch (Arrow). "
            "This is the knob that decides how much has to be read to reach one sample "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--compression",
        default="none",
        help="'none' by default: the payloads are already-compressed JPEG",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.group_size < 1:
        parser.error("--group-size must be >= 1")

    try:
        samples = read_manifest(args.manifest)
    except ManifestError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    if not samples:
        print(f"ERROR {args.manifest} contains no samples", file=sys.stderr)
        return 2

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        print(
            f"ERROR {args.output} is not empty; pass --overwrite to reconvert",
            file=sys.stderr,
        )
        return 2

    progress = max(1, len(samples) // 10)
    try:
        if args.track == "parquet":
            from examples.parquet_track import convert

            result = convert(
                args.source, samples, args.output,
                row_group_size=args.group_size,
                compression=args.compression,
                progress_every=progress,
            )
        elif args.track == "hdf5":
            from examples.hdf5_track import convert

            result = convert(
                args.source, samples, args.output,
                chunk_size=args.group_size,
                compression=None if args.compression == "none" else args.compression,
                progress_every=progress,
            )
        else:
            from examples.huggingface_track import cache_dir_advice, convert

            result = convert(
                args.source, samples, args.output,
                writer_batch_size=args.group_size,
                progress_every=progress,
            )
            print("\n--- keep the cache off your home directory ---")
            for key, value in cache_dir_advice(args.output).items():
                print(f"export {key}={value}")
            print()
    except AdapterError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR conversion failed: {exc}", file=sys.stderr)
        return 2

    print(f"TRACK={args.track}")
    for key, value in result.items():
        print(f"{key.upper()}={value}")
    print(f"MANIFEST_SAMPLES={len(samples)}")
    print(f"ADAPTER={ADAPTERS[args.track]}")

    if result.get("rows") not in (None, len(samples)):
        print(
            f"WARNING wrote {result['rows']} rows for {len(samples)} manifest samples",
            file=sys.stderr,
        )
        return 1

    (args.output / "conversion.json").write_text(
        json.dumps({**result, "track": args.track, "manifest": str(args.manifest)},
                   indent=2, sort_keys=True) + "\n"
    )
    print()
    print("Measure it with the same loader the core layouts use:")
    print("  sbatch jobs/run_loader.sh configs/formats/%s.yaml" % args.track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
