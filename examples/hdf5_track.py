"""HDF5: samples as rows in one chunked dataset.

HDF5 suits dense arrays and scientific data. The tutorial's images go in as variable-
length byte rows, which is a legitimate but not idiomatic use - the point is to put the
format on the same measurement footing as the core layouts.

Two things decide performance here, and both are exposed:

**Chunking.** HDF5 reads and writes whole chunks. A chunk spanning many samples turns
sequential access into few large reads; a chunk per sample makes random access cheap and
multiplies overhead. This is the same trade as a Parquet row group or a tar shard.

**Concurrency.** A single HDF5 file handle is not safe to share across processes. Handles
are opened lazily, once per process, by the adapter base class - a handle created before
a DataLoader fork and used in several workers gives corrupt reads or crashes, which is
the classic HDF5-with-workers failure.

Requires h5py:  pip install '.[hdf5]'
Note that LUMI's PyTorch containers do not currently include h5py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from dataaware.adapters import DatasetAdapter
from dataaware.errors import DataError
from dataaware.manifest import Sample

#: Samples per chunk. Matches the tutorial's default shard size so that HDF5, Parquet,
#: and tar shards group the same number of samples per unit of I/O.
DEFAULT_CHUNK = 1000

ARTIFACT = "dataset.h5"
PAYLOAD = "payload"
CLASS_IDS = "class_id"


def convert(
    source_root: Path,
    samples: Sequence[Sample],
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK,
    compression: str | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Write the dataset as one HDF5 file, preserving manifest order.

    ``compression`` defaults to none, for the same reason as everywhere else in this
    tutorial: the payloads are already-compressed JPEG.
    """
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise DataError(f"h5py is required for the HDF5 track: {exc}") from None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ARTIFACT

    variable_bytes = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(target, "w") as handle:
        payloads = handle.create_dataset(
            PAYLOAD,
            shape=(len(samples),),
            dtype=variable_bytes,
            chunks=(min(chunk_size, len(samples)),),
            compression=compression,
        )
        classes = handle.create_dataset(
            CLASS_IDS, shape=(len(samples),), dtype="int32"
        )
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            payloads[start : start + len(chunk)] = [
                np.frombuffer((source_root / s.relative_path).read_bytes(), dtype="uint8")
                for s in chunk
            ]
            classes[start : start + len(chunk)] = [s.class_id for s in chunk]
            if progress_every and (start + len(chunk)) % progress_every < chunk_size:
                print(f"wrote {start + len(chunk)}/{len(samples)} rows", flush=True)
        handle.attrs["total_samples"] = len(samples)

    return {
        "artifact": str(target),
        "artifact_bytes": target.stat().st_size,
        "rows": len(samples),
        "chunk_size": chunk_size,
        "compression": compression or "none",
    }


class HDF5Adapter(DatasetAdapter):
    """Reads sample bytes from one HDF5 dataset by manifest position."""

    name = "hdf5"

    def open_resource(self) -> Any:
        try:
            import h5py
        except ImportError as exc:
            raise DataError(f"h5py is required for the HDF5 track: {exc}") from None

        path = self.root if self.root.is_file() else self.root / ARTIFACT
        if not path.is_file():
            raise DataError(
                f"HDF5 artifact not found: {path}\n"
                "Convert it first: python3 scripts/convert_dataset.py --to hdf5"
            )
        # Read-only, and no file locking: on a parallel filesystem, locks across many
        # readers are both unnecessary for read-only access and a common source of
        # hangs.
        handle = h5py.File(path, "r", locking=False)
        stored = int(handle.attrs.get("total_samples", handle[PAYLOAD].shape[0]))
        if stored != len(self.samples):
            handle.close()
            raise DataError(
                f"{path} holds {stored} rows but the manifest has {len(self.samples)}. "
                "Reconvert from the same manifest."
            )
        return handle

    def read_payload(self, index: int) -> bytes:
        return self.resource()[PAYLOAD][index].tobytes()

    def describe(self) -> dict[str, Any]:
        path = self.root if self.root.is_file() else self.root / ARTIFACT
        handle = self.resource()
        chunks = handle[PAYLOAD].chunks
        return {
            "artifact_bytes": path.stat().st_size,
            "chunk_size": int(chunks[0]) if chunks else 0,
            "filesystem_objects": 1,
        }
