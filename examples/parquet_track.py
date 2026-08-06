"""Parquet: samples as rows in a columnar file.

Parquet suits tabular and record-oriented data. Images in a binary column are an
unusual but instructive case: it shows what row-group sizing does, and it puts a
columnar format on the same measurement footing as the core layouts.

The knob that matters is the **row group**. A reader must decode a whole row group to
reach any row in it, so large groups compress better and stream faster, while small
groups make random access cheaper and multiply per-group overhead. This track exposes it
so you can measure the trade rather than guess.

Requires pyarrow:  pip install '.[parquet]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from dataaware.adapters import DatasetAdapter
from dataaware.errors import DataError
from dataaware.manifest import Sample

#: Rows per row group. 1000 mirrors the tutorial's default shard size, so the Parquet
#: run and the tar-shard run group the same number of samples per unit of I/O.
DEFAULT_ROW_GROUP = 1000

ARTIFACT = "dataset.parquet"


def convert(
    source_root: Path,
    samples: Sequence[Sample],
    output_dir: Path,
    row_group_size: int = DEFAULT_ROW_GROUP,
    compression: str = "none",
    progress_every: int = 0,
) -> dict[str, Any]:
    """Write the dataset as one Parquet file, preserving manifest order.

    ``compression`` defaults to none: the samples are already-compressed JPEG, so
    compressing again costs read CPU for almost nothing - the same reasoning as the
    SquashFS ``-noD -noF`` default in Part III.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError(f"pyarrow is required for the Parquet track: {exc}") from None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ARTIFACT

    schema = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("class_id", pa.int32()),
            pa.field("payload", pa.large_binary()),
        ]
    )

    written = 0
    with pq.ParquetWriter(target, schema, compression=compression) as writer:
        for start in range(0, len(samples), row_group_size):
            chunk = samples[start : start + row_group_size]
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        "sample_id": [s.sample_id for s in chunk],
                        "class_id": [s.class_id for s in chunk],
                        "payload": [
                            (source_root / s.relative_path).read_bytes() for s in chunk
                        ],
                    },
                    schema=schema,
                )
            )
            written += len(chunk)
            if progress_every and written % progress_every < row_group_size:
                print(f"wrote {written}/{len(samples)} rows", flush=True)

    metadata = pq.ParquetFile(target)
    return {
        "artifact": str(target),
        "artifact_bytes": target.stat().st_size,
        "rows": metadata.metadata.num_rows,
        "row_groups": metadata.metadata.num_row_groups,
        "row_group_size": row_group_size,
        "compression": compression,
    }


class ParquetAdapter(DatasetAdapter):
    """Reads sample bytes from a Parquet file by manifest position.

    Row groups are read whole and cached one at a time. That is not an optimisation, it
    is what the format does: reaching one row means decoding its group, so a reader that
    jumps between groups pays for each jump. Keeping the current group is what turns
    sequential access into one decode per group instead of one per row.
    """

    name = "parquet"

    def __init__(self, root, samples, options=None) -> None:
        super().__init__(root, samples, options)
        self._group_index: list[tuple[int, int]] = []
        self._cached_group: int | None = None
        self._cached_rows: dict[int, bytes] = {}

    def open_resource(self) -> Any:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DataError(
                f"pyarrow is required for the Parquet track: {exc}"
            ) from None

        path = self.root if self.root.is_file() else self.root / ARTIFACT
        if not path.is_file():
            raise DataError(
                f"Parquet artifact not found: {path}\n"
                "Convert it first: python3 scripts/convert_dataset.py --to parquet"
            )
        handle = pq.ParquetFile(path)

        # Row boundaries, so a manifest position maps to (group, offset within group).
        self._group_index = []
        start = 0
        for group in range(handle.metadata.num_row_groups):
            rows = handle.metadata.row_group(group).num_rows
            self._group_index.append((start, start + rows))
            start += rows
        if start != len(self.samples):
            raise DataError(
                f"{path} holds {start} rows but the manifest has {len(self.samples)}. "
                "Reconvert from the same manifest."
            )
        return handle

    def read_payload(self, index: int) -> bytes:
        handle = self.resource()
        group = self._group_for(index)
        if group != self._cached_group:
            table = handle.read_row_group(group, columns=["payload"])
            start = self._group_index[group][0]
            self._cached_rows = {
                start + offset: value.as_py()
                for offset, value in enumerate(table.column("payload"))
            }
            self._cached_group = group
        return self._cached_rows[index]

    def _group_for(self, index: int) -> int:
        for group, (start, end) in enumerate(self._group_index):
            if start <= index < end:
                return group
        raise DataError(f"row {index} is outside the Parquet file")

    def describe(self) -> dict[str, Any]:
        handle = self.resource()
        path = self.root if self.root.is_file() else self.root / ARTIFACT
        return {
            "artifact_bytes": path.stat().st_size,
            "row_groups": handle.metadata.num_row_groups,
            "filesystem_objects": 1,
        }
