"""Reading a representation the core tutorial does not implement.

The optional format tracks - Parquet, HDF5, Hugging Face Datasets - and any dataset of
your own plug in here. An adapter has one job: given a position in the manifest, return
that sample's encoded bytes.

Everything else stays shared. The decode, the batching, the timing, the sample
accounting, and the run summary are the same code the core layouts use, which is what
makes an optional track *comparable* to them rather than merely similar. The comparison
tools will happily put a Parquet run next to a SquashFS run, and refuse to if they read
different data.

Optional dependencies are imported inside each adapter, never at module scope, so the
core tutorial runs with none of them installed.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from .manifest import Sample
from .errors import DataError


class DatasetAdapter(ABC):
    """Return the encoded bytes for a manifest position.

    Subclasses implement :meth:`read_payload`. They must **not** open file handles in
    ``__init__``: DataLoader workers are forked processes, and a handle created before
    the fork and then used in several workers gives corrupt reads or crashes - the usual
    symptom with HDF5 and Parquet alike. Use :meth:`resource`, which opens lazily and
    once per process.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        samples: Sequence[Sample],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.samples = samples
        self.options = options or {}
        self._resource: Any = None
        self._resource_pid: int | None = None

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def read_payload(self, index: int) -> bytes:
        """Return the encoded bytes of ``self.samples[index]``."""

    def open_resource(self) -> Any:
        """Open whatever this adapter reads from. Called once per process.

        Returning ``None`` is fine for adapters that need no persistent handle.
        """
        return None

    def resource(self) -> Any:
        """The per-process handle, opened on first use in this process.

        Re-opened after a fork, because a handle inherited across ``fork`` is not safe
        to use concurrently in parent and child.
        """
        pid = os.getpid()
        if self._resource is None or self._resource_pid != pid:
            self._resource = self.open_resource()
            self._resource_pid = pid
        return self._resource

    def close(self) -> None:
        handle = self._resource
        self._resource = None
        self._resource_pid = None
        if handle is not None and hasattr(handle, "close"):
            try:
                handle.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask results
                print(f"WARNING error closing adapter resource: {exc}", flush=True)

    #: Human-readable name for logs and summaries. Overridden by subclasses.
    name = "adapter"

    def describe(self) -> dict[str, Any]:
        """Optional metrics for the run summary, e.g. an artifact's size.

        Keys must already exist in ``dataaware.schema.OPTIONAL``; the schema
        rejects unknown fields, so a typo here fails loudly rather than vanishing.
        """
        return {}


def load_adapter(
    spec: str,
    root: str | os.PathLike[str],
    samples: Sequence[Sample],
    options: dict[str, Any] | None = None,
) -> DatasetAdapter:
    """Instantiate an adapter from a ``module:Class`` specification.

    The import is deferred to here so that a missing optional dependency surfaces as a
    clear message about that track, rather than as an import error at startup in a run
    that does not use it.
    """
    if ":" not in spec:
        raise DataError(
            f"adapter {spec!r} must look like 'module:Class', for example "
            "'examples.parquet_track:ParquetAdapter'"
        )
    module_name, class_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DataError(
            f"cannot import adapter module {module_name!r}: {exc}\n"
            "If this is an optional track, install its extra: "
            "pip install '.[parquet]' / '.[hdf5]' / '.[huggingface]'"
        ) from None
    try:
        adapter_class = getattr(module, class_name)
    except AttributeError:
        raise DataError(
            f"module {module_name!r} has no attribute {class_name!r}"
        ) from None
    if not (isinstance(adapter_class, type) and issubclass(adapter_class, DatasetAdapter)):
        raise DataError(
            f"{spec} is not a DatasetAdapter subclass; see dataaware/adapters.py"
        )
    return adapter_class(root=root, samples=samples, options=options)
