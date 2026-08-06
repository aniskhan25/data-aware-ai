# Using this with your own dataset

The synthetic dataset exists so everyone measures the same bytes. It is not the point.
The point is your data, and the whole method works on it.

There are two ways in, and most datasets need only the first.

## 1. Your data is already files: write a manifest

If your samples are files on disk — images, audio, npy, anything one-file-per-sample —
you need no code at all. Write a manifest and point a configuration at it.

A manifest is JSON Lines, one record per sample, sorted by `sample_id`:

```json
{"byte_size": 40213, "checksum": "b94605a6125bbb60", "class_id": 3,
 "estimated_decode_cost": 50176, "height": 224,
 "relative_path": "train/cat/img_0001.jpg", "sample_id": "s0a3f19c2e8b1d47",
 "width": 224}
```

| Field | What it is | If you do not have it |
| ----- | ---------- | --------------------- |
| `sample_id` | Stable identity, unique | Derive from the path — see below |
| `relative_path` | Path under `dataset.root` | Required |
| `class_id` | Integer label | Use `0` if unlabelled |
| `byte_size` | File size | Required; it is how bytes read are counted |
| `width`, `height` | Pixels | Use `1` for non-image data |
| `checksum` | First 16 hex of SHA-256 | Optional in practice, but it is what proves two layouts read the same data |
| `estimated_decode_cost` | Work proxy for shard balancing | Use `byte_size` if you have nothing better |

Generate one with a few lines:

```python
import json
from pathlib import Path
from dataaware.manifest import Sample, checksum_bytes, stable_sample_id, write_manifest

root = Path("/scratch/project_XXXXXXXXX/me/my-dataset")
samples = []
for path in sorted(root.rglob("*.jpg")):
    payload = path.read_bytes()
    relative = str(path.relative_to(root))
    samples.append(Sample(
        sample_id=stable_sample_id(relative),      # stable across machines and runs
        relative_path=relative,
        class_id=0,
        byte_size=len(payload),
        width=1, height=1,                          # fill in if you know them
        checksum=checksum_bytes(payload),
        estimated_decode_cost=len(payload),
    ))
write_manifest(root / "manifest.jsonl", samples)
```

`stable_sample_id` hashes the path, so identities do not depend on enumeration order.
That matters once you shard: shard membership must not change because a directory listing
came back differently.

Then copy a configuration and change two lines:

```yaml
dataset:
  layout: loose-files
  root: /scratch/project_XXXXXXXXX/me/my-dataset
  manifest: /scratch/project_XXXXXXXXX/me/my-dataset/manifest.jsonl
```

Everything else in the tutorial now applies unchanged. Inspect it, baseline it, package
it, tune workers, compare placements, validate distributed reading, and get a readiness
verdict — all on your data.

## 2. Your data is not files: write an adapter

If samples live inside something — a Parquet file, an HDF5 dataset, a database, a format
of your own — implement a `DatasetAdapter`. Its only job is: *given a position in the
manifest, return that sample's encoded bytes*.

```python
from dataaware.adapters import DatasetAdapter

class MyAdapter(DatasetAdapter):
    name = "my-format"                 # becomes the layout label in comparisons

    def open_resource(self):
        # Opened lazily, once per process. NEVER open in __init__: DataLoader workers
        # are forked, and a handle created before the fork then used in several workers
        # gives corrupt reads or crashes.
        import myformat
        return myformat.open(self.root / "dataset.bin")

    def read_payload(self, index: int) -> bytes:
        sample = self.samples[index]
        return self.resource().get(sample.sample_id)

    def describe(self) -> dict:
        # Optional artifact metrics for the run summary. Keys must already exist in
        # OPTIONAL_FIELDS in dataaware/schema.py, which rejects unknown fields.
        return {"artifact_bytes": (self.root / "dataset.bin").stat().st_size,
                "filesystem_objects": 1}
```

```yaml
dataset:
  layout: adapter
  adapter: mypackage.mymodule:MyAdapter
  root: /scratch/project_XXXXXXXXX/me/my-dataset
  manifest: /scratch/project_XXXXXXXXX/me/my-dataset/manifest.jsonl
```

The decode, batching, timing, sample accounting, and run summary stay shared. That is
what makes your format **comparable** with the tutorial's layouts rather than merely
measured alongside them — the comparison tools check the manifest hash and refuse if two
runs read different data.

Worked examples live in `examples/`: `parquet_track.py`, `hdf5_track.py`, and
`huggingface_track.py` are each about thirty lines of real work.

## Adapting the experiments

| Part | What to change |
| ---- | -------------- |
| I Inspect | Nothing. Point it at your tree |
| II Baseline | Nothing, once the manifest exists |
| III Layouts | Skip SquashFS if your data is mutable; skip shards if you need path-based random access |
| IV Workers | Nothing. The ladder is dataset-independent |
| V Storage | Check the size against your allocation first — the staging check will refuse if it is unsafe |
| VI Distributed | Build at least as many shards as `ranks x workers`, ideally a multiple |
| VII Decision | Set `--planned-epochs` to what you will actually run |

Two adjustments worth making deliberately:

**Decode cost.** The tutorial decodes JPEG. If your samples are cheap to decode (raw
arrays) or expensive (large PNG, video frames), the balance between storage and CPU
shifts, and `MAIN_LIMITING_FACTOR` in Part VII will say so. That is the answer changing
because your workload is different, not the method failing.

**Sample size.** The tutorial's dataset is metadata-heavy: 2.7 KB files where per-file
overhead dominates. If your samples are megabytes, packaging will matter far less and
Part III may legitimately conclude "stay with loose files".

## What not to change

Keep one manifest across every experiment. Every comparison in this tutorial rests on
runs having read the same samples in the same order, and `manifest_hash` is what proves
it. Rebuilding a manifest between runs — even from identical data — silently invalidates
the comparison, and the tools will tell you so rather than produce a tidy table.
