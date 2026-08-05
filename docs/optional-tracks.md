# Optional format tracks

None of this is required to complete the tutorial. The core path measures three layouts;
these tracks apply the **same measurement** to three more ecosystems, and to any dataset
of your own.

## How a track plugs in

A track supplies two things: a converter, and a `DatasetAdapter` whose only job is *given
a position in the manifest, return that sample's encoded bytes*.

Everything else is shared — the decode, the batching, the timing, the sample accounting,
the run summary. That is what makes an optional track **comparable** with a core layout
rather than merely described alongside one. The comparison tools will put a Parquet run
next to a SquashFS run, and refuse if they read different data.

```yaml
dataset:
  layout: adapter
  adapter: examples.parquet_track:ParquetAdapter
  root: ${TUTORIAL_ROOT}/parquet
  manifest: ${TUTORIAL_ROOT}/source/manifest.jsonl   # the same manifest
```

Each track's summary is labelled with the adapter's own name (`parquet`, `hdf5`,
`huggingface`), so several tracks appear as separate rows instead of collapsing into one
called "adapter".

## Isolation

| Track | Extra | In LUMI's PyTorch containers? |
| ----- | ----- | ----------------------------- |
| Parquet | `pip install '.[parquet]'` | **Yes** (pyarrow) |
| Hugging Face | `pip install '.[huggingface]'` | **Yes** (datasets) |
| HDF5 | `pip install '.[hdf5]'` | **No** — load a module or install h5py yourself |

Dependencies are imported inside the track that needs them, never at module scope. The
core tutorial runs with none of them installed, and a missing one produces a message
naming the extra rather than an import error at startup.

## Convert and measure

```bash
python3 scripts/convert_dataset.py --to parquet \
    --source   "$TUTORIAL_ROOT/source" \
    --manifest "$TUTORIAL_ROOT/source/manifest.jsonl" \
    --output   "$TUTORIAL_ROOT/parquet" \
    --group-size 1250

sbatch jobs/run_loader.sh configs/formats/parquet.yaml
```

`--group-size` is the knob worth understanding. It is the row group in Parquet, the chunk
in HDF5, and the writer batch in Arrow — in every case, *how much has to be read to reach
one sample*.

## Measured on LUMI

50 000 samples, 13 workers, 1000 batches, two repeats. Both artifacts hold all 50 000 rows
and are byte-identical to the loose files, verified by checksum.

| Access pattern | Parquet | Hugging Face |
| -------------- | ------- | ------------ |
| Random (`shuffle: true`) | **983** samples/s | 11 290 samples/s |
| Sequential (`shuffle: false`) | **20 070** samples/s | 17 090 samples/s |

Artifact sizes: Parquet 135 MB in 1 file (40 row groups); Arrow 135 MB in 3 files. The
loose tree is 144 MB in 50 002 files.

### What this shows

**Parquet is both the fastest and the slowest representation in this tutorial — a 20x
swing from one configuration flag.** Read sequentially it beats every core layout,
including tar shards at 6926 samples/s. Read at random it is slower than SquashFS.

The reason is structural, not incidental: a columnar format decodes a whole row group to
reach any row in it. With 40 groups of 1250 rows, random access re-reads about 3.4 MB of
row group to obtain one 2.7 KB sample. Sequential access pays that cost once per group and
amortises it over 1250 samples.

This is Part III's usable / suitable / scalable distinction in a single table. Parquet is
*usable* for image samples either way. It is only *suitable* if the access pattern is
sequential.

**Memory mapping degrades gracefully.** Arrow lost only a third going from sequential to
random (17 090 → 11 290), against Parquet's 20-fold collapse. `load_from_disk` maps the
file and lets the operating system page it, so a random read costs a page fault rather
than a group decode. If you need shuffled access and a columnar-ish format, that
difference is the argument.

### Two honest caveats

The 20x gap is **amplified by this adapter**, which caches exactly one row group. A
production reader with a larger cache, or one that consumes whole batches per group rather
than row by row, would land somewhere between these numbers. The direction is real; the
magnitude is partly a property of the code in `examples/parquet_track.py`.

These runs used 13 workers and 1000 batches, while the Part III table used 4 workers and
200. **The two tables are therefore not directly comparable**, and `compare_layouts.py`
says so — it reports `CONTROLLED_COMPARISON=false` and names `num_workers` and
`measured_batches` as the differences. To place a track beside the Part III results, re-run
it with matching settings.

### Reading the `Opens` column for a track

For loose files, `files_opened` is literally filesystem opens. For an adapter it counts
*logical sample reads*, because the adapter does not report how many times it touched the
underlying file. A Parquet run showing 64 000 "opens" against a single-file artifact is
counting samples, not opens. `filesystem_objects` is the number to read for artifact
count: 1 for Parquet and HDF5, 3 for Arrow, 50 002 for the loose tree.

## HDF5

Implemented and unit-tested, but **not measured on LUMI**: the PyTorch containers do not
include h5py. The track is complete and will run wherever h5py is available.

Two things decide HDF5 performance, and both are exposed:

- **Chunking** (`--group-size`) is the same trade as a Parquet row group.
- **Concurrency.** A single HDF5 handle is not safe to share across processes. The adapter
  base class opens handles lazily, once per process, keyed on pid — a handle created
  before a DataLoader fork and used in several workers gives corrupt reads or crashes,
  which is the classic HDF5-with-workers failure.

## Hugging Face Datasets

The format is rarely the problem; the operational detail around it is.

- **The cache defaults to your home directory** — small, quota-tight, and the wrong
  filesystem for job I/O. Worse, in a distributed job every rank tries to build it. The
  converter prints the `HF_HOME` and `HF_DATASETS_CACHE` exports that put it on scratch.
- **Compute nodes have no internet.** `HF_DATASETS_OFFLINE=1` turns a silent hang into an
  error. The adapter sets it if you have not.
- **The progress bar is not a progress bar in a batch job.** It is tens of thousands of
  carriage-return lines in the Slurm error log, burying anything real. The converter
  disables it — the first version of this track produced a 93 KB error log for a
  successful run.

## LUMI-O

Object storage is a lifecycle, not a layout, so it has its own page:
[`docs/object-storage.md`](object-storage.md). Verify a round trip with

```bash
python3 scripts/lumio_roundtrip.py --list-remotes
python3 scripts/lumio_roundtrip.py --remote lumi-<project>-private \
    --bucket data-aware-ai --file "$TUTORIAL_ROOT/source.squashfs"
```

It uploads, downloads, compares SHA-256, and cleans up. It refuses a remote whose name
does not contain `private` unless you insist, because publishing data by accident is not
recoverable. **No credential is ever read, printed, accepted, or stored** by anything in
this repository, and `rclone.conf` is gitignored.

## Adding a track of your own

1. Subclass `DatasetAdapter` and implement `read_payload(index)`. Open handles in
   `open_resource`, never in `__init__` — see the fork warning above.
2. Give it a `name`, so it gets its own row in comparisons.
3. Return artifact metrics from `describe()`. Keys must already exist in
   `OPTIONAL_FIELDS` in `dataaware/schema.py`, which rejects unknown fields, so a typo
   fails loudly.
4. Prove it returns the same bytes as the manifest.
   `test_a_track_returns_the_same_bytes_as_the_manifest` in `tests/test_adapters.py` is
   the pattern, and it is the property the whole comparison rests on.
