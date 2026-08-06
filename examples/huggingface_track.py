"""Hugging Face Datasets: an Arrow-backed dataset on disk.

This is the ecosystem case. What matters on LUMI is less the format than the operational
detail around it, and this track exists mostly to make those visible:

**The cache is the problem, not Arrow.** ``datasets`` writes a cache under
``HF_HOME``/``HF_DATASETS_CACHE``, which defaults to your home directory - small, quota-
tight, and the wrong filesystem for job I/O. Worse, in a distributed job *every rank*
will try to build it. Point the cache at project scratch and build it once, before the
job.

**Memory mapping.** ``load_from_disk`` maps the Arrow file rather than reading it, so
startup is cheap and the operating system does the paging. That is a genuine advantage
over formats that must be parsed, and it is why this track's startup cost is worth
watching in the summary.

**Offline.** Compute nodes have no internet. Anything that would download at run time
must be materialised beforehand; ``HF_DATASETS_OFFLINE=1`` turns a silent hang into an
error.

Requires datasets:  pip install '.[huggingface]'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dataaware.adapters import DatasetAdapter
from dataaware.errors import DataError
from dataaware.manifest import Sample

ARTIFACT = "hf_dataset"


def cache_dir_advice(root: Path) -> dict[str, str]:
    """Environment that keeps the cache off home and out of the network.

    Returned rather than applied, so a caller can print it, export it in a job script, or
    ignore it deliberately.
    """
    cache = root / "hf_cache"
    return {
        "HF_HOME": str(cache),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "HF_DATASETS_OFFLINE": "1",
    }


def convert(
    source_root: Path,
    samples: Sequence[Sample],
    output_dir: Path,
    writer_batch_size: int = 1000,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Build an Arrow dataset on disk, preserving manifest order."""
    try:
        import datasets as hf_datasets
        from datasets import Dataset, Features, Value
    except ImportError as exc:
        raise DataError(
            f"the 'datasets' package is required for the Hugging Face track: {exc}"
        ) from None

    # The progress bar writes a carriage-return update per batch. In a Slurm job that is
    # not a progress bar, it is tens of thousands of lines of noise in the error log,
    # which buries any real message. The converter prints its own progress instead.
    for disable in ("disable_progress_bars", "disable_progress_bar"):
        if hasattr(hf_datasets, disable):
            getattr(hf_datasets, disable)()
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ARTIFACT

    def generate():
        for position, sample in enumerate(samples):
            if progress_every and position and position % progress_every == 0:
                print(f"wrote {position}/{len(samples)} rows", flush=True)
            yield {
                "sample_id": sample.sample_id,
                "class_id": sample.class_id,
                "payload": (source_root / sample.relative_path).read_bytes(),
            }

    features = Features(
        {
            "sample_id": Value("string"),
            "class_id": Value("int32"),
            "payload": Value("binary"),
        }
    )
    dataset = Dataset.from_generator(
        generate, features=features, writer_batch_size=writer_batch_size
    )
    dataset.save_to_disk(str(target))

    total = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    return {
        "artifact": str(target),
        "artifact_bytes": total,
        "rows": dataset.num_rows,
        "writer_batch_size": writer_batch_size,
    }


class HuggingFaceAdapter(DatasetAdapter):
    """Reads sample bytes from an Arrow dataset saved with ``save_to_disk``."""

    name = "huggingface"

    def open_resource(self) -> Any:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise DataError(
                f"the 'datasets' package is required for the Hugging Face track: {exc}"
            ) from None

        path = self.root if (self.root / "dataset_info.json").exists() else self.root / ARTIFACT
        if not path.exists():
            raise DataError(
                f"Arrow dataset not found: {path}\n"
                "Convert it first: python3 scripts/convert_dataset.py --to huggingface"
            )
        # Compute nodes have no internet; make an attempted download fail rather than
        # hang.
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        dataset = load_from_disk(str(path))
        if dataset.num_rows != len(self.samples):
            raise DataError(
                f"{path} holds {dataset.num_rows} rows but the manifest has "
                f"{len(self.samples)}. Reconvert from the same manifest."
            )
        return dataset

    def read_payload(self, index: int) -> bytes:
        return self.resource()[index]["payload"]

    def describe(self) -> dict[str, Any]:
        path = self.root if (self.root / "dataset_info.json").exists() else self.root / ARTIFACT
        total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        objects = sum(1 for item in path.rglob("*") if item.is_file())
        return {"artifact_bytes": total, "filesystem_objects": objects}
