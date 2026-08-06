"""The sample manifest.

Every dataset representation in this tutorial reads the same manifest. That is
what makes a layout comparison meaningful: loose files, a SquashFS image, and
tar shards are then measured over identical records in an identical order.

The manifest is JSON Lines, one record per sample, sorted by ``sample_id`` and
written with sorted keys so that regenerating a dataset produces a
byte-identical file.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator
from .errors import DataError

#: Fields required in every manifest record.
MANIFEST_FIELDS = (
    "sample_id",
    "relative_path",
    "class_id",
    "byte_size",
    "width",
    "height",
    "checksum",
    "estimated_decode_cost",
)


@dataclass(frozen=True)
class Sample:
    """One sample, independent of how the bytes are stored."""

    sample_id: str
    relative_path: str
    class_id: int
    byte_size: int
    width: int
    height: int
    #: SHA-256 of the stored bytes, truncated to 16 hex characters. Used to prove
    #: that different layouts really do return the same sample.
    checksum: str
    #: Work proxy for shard balancing. Pixel count, not measured seconds.
    estimated_decode_cost: int


def stable_sample_id(relative_path: str) -> str:
    """Deterministic sample ID derived from a path.

    Used for user-supplied datasets, where no generator assigned an index. Stable
    across machines and runs, unlike enumeration order.
    """
    digest = hashlib.sha256(relative_path.encode()).hexdigest()
    return f"s{digest[:15]}"


def checksum_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def write_manifest(path: str | os.PathLike[str], samples: Iterable[Sample]) -> Path:
    """Write records sorted by ``sample_id``, rejecting duplicates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(samples, key=lambda s: s.sample_id)
    seen: set[str] = set()
    for sample in ordered:
        if sample.sample_id in seen:
            raise DataError(f"duplicate sample_id in manifest: {sample.sample_id}")
        seen.add(sample.sample_id)

    with path.open("w") as handle:
        for sample in ordered:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
    return path


def read_manifest(path: str | os.PathLike[str]) -> list[Sample]:
    """Read and validate a manifest. Order is the file order."""
    return list(iter_manifest(path))


def iter_manifest(path: str | os.PathLike[str]) -> Iterator[Sample]:
    path = Path(path)
    try:
        handle = path.open()
    except FileNotFoundError:
        raise DataError(
            f"manifest not found: {path}\n"
            "Generate one with scripts/generate_dataset.py, or point "
            "dataset.manifest at an existing manifest."
        ) from None

    known = {f.name for f in fields(Sample)}
    seen: set[str] = set()
    with handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{number}: invalid JSON: {exc}") from None
            if not isinstance(record, dict):
                raise DataError(f"{path}:{number}: record must be an object")

            missing = [name for name in MANIFEST_FIELDS if name not in record]
            if missing:
                raise DataError(f"{path}:{number}: missing field(s) {missing}")
            unknown = set(record) - known
            if unknown:
                raise DataError(f"{path}:{number}: unknown field(s) {sorted(unknown)}")

            sample = Sample(**record)
            if sample.sample_id in seen:
                raise DataError(
                    f"{path}:{number}: duplicate sample_id {sample.sample_id}"
                )
            seen.add(sample.sample_id)
            yield sample


def manifest_hash(path: str | os.PathLike[str]) -> str:
    """SHA-256 of the manifest file, truncated.

    Comparison tools refuse to compare runs whose manifest hashes differ: those
    runs did not read the same data.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def total_bytes(samples: Iterable[Sample]) -> int:
    return sum(sample.byte_size for sample in samples)
