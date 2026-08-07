# Optional format tracks

None of this is required to complete the tutorial. The core path measures three layouts;
these tracks apply the **same measurement** to three more ecosystems, and to any dataset
of your own.

## How a track plugs in

A track supplies two things: a converter, and a `DatasetAdapter` whose only job is *given
a position in the manifest, return that sample's encoded bytes*.

Everything else is shared - the decode, the batching, the timing, the sample accounting,
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

Which container you use decides what is already available. Measured 2026-08-07:

| Track | Extra | `sif-images` PyTorch | LUMI AI Factory (LAIF) |
| ----- | ----- | -------------------- | ---------------------- |
| Parquet | `pip install '.[parquet]'` | **Yes**, pyarrow 21.0.0 | **Yes**, pyarrow 25.0.0 |
| Hugging Face | `pip install '.[huggingface]'` | **Yes**, datasets 4.0.0 | **Yes**, datasets 5.0.0 |
| HDF5 | `pip install '.[hdf5]'` | **No** | **Yes**, h5py 3.16.0 |

The LAIF image also carries zarr 3.3.0, polars, pandas, and boto3. It does *not* carry
`webdataset` or `netCDF4`, neither of which this tutorial needs: the shard layout reads
tar archives with the standard library.

```text
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

That symlink moves when a new image is published. Pin the versioned path in `env.sh` if
you want a run to stay reproducible.

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
in HDF5, and the writer batch in Arrow - in every case, *how much has to be read to reach
one sample*.

## Measured on LUMI

All three tracks, 50 000 samples, 13 workers, 1000 batches, **three repeats each**, in the
LAIF container on project scratch. Every artifact holds all 50 000 rows and is
byte-identical to the loose files, verified by checksum. Every run reported zero failed,
duplicate, and missing samples.

| Access pattern | Parquet | HDF5 | Hugging Face (Arrow) |
| -------------- | ------- | ---- | -------------------- |
| Random (`shuffle: true`) | **902** (CV 0.26) | 11 723 (CV 0.07) | 8 240 (CV 0.22) |
| Sequential (`shuffle: false`) | **22 469** (CV 0.02) | 11 112 (CV 0.08) | 15 020 (CV 0.12) |
| Sequential / random | **25x** | 1.05x | 1.8x |

Median samples/s over three repeats. Artifact sizes: Parquet 135 MB in 1 file (40 row
groups); HDF5 138 MB in 1 file (50 chunks of 1000); Arrow 135 MB in 3 files. The loose
tree is 144 MB in 50 002 files.

### What this shows

**How much you must read to reach one sample decides how much access order costs you.**
That single quantity orders the whole table:

| Track | Read granularity | Sensitivity to access order |
| ----- | ---------------- | --------------------------- |
| Parquet | a whole row group, 1250 samples | 25x |
| Arrow | a memory-mapped page | 1.8x |
| HDF5 | a chunk of 1000, but cached per handle | none measurable |

**Parquet is both the fastest and the slowest representation in this tutorial - a 25x
swing from one configuration flag.** Read sequentially it beats every core layout,
including tar shards at 6926 samples/s. Read at random it is slower than the loose files.

The direction is structural; the magnitude is partly this adapter's. Reaching a random
sample may require loading and decompressing substantial row-group data, and with a
single-row-group cache shuffled access repeatedly evicts and reloads groups - about 3.4 MB
of row group per 2.7 KB sample. Sequential access pays that once per group and amortises
it over 1250 samples. A reader with a larger cache, different page layout, column
selection, or batch-oriented access would land somewhere between these numbers. Note also
that random Parquet is the noisiest configuration measured anywhere in this tutorial
(CV 0.26), which is what cache thrashing looks like.

This is Part III's usable / suitable / scalable distinction in a single table. Parquet is
*usable* for image samples either way. It is only *suitable* if the access pattern is
sequential.

**HDF5 is indifferent to access order.** 11 723 random against 11 112 sequential is a 5 %
difference against repeat CVs of 0.07 and 0.08 - inside the noise, so the honest reading is
that the two are indistinguishable, not that random is faster. h5py keeps a per-handle
chunk cache and the adapter opens one handle per worker process, so a shuffled read stream
still finds most of its samples in an already-decoded chunk. If you need full shuffling
every epoch and a single-file artifact, this is the property to want.

**Memory mapping degrades gracefully.** Arrow lost about half going from sequential to
random (15 020 to 8 240), against Parquet's 25-fold collapse. `load_from_disk` memory-maps
the file, so the operating-system page cache can serve many accesses without reloading a
whole row group. Random access still costs page faults, offset resolution, and some
decoding - it is cheaper here, not free.

> These numbers were measured in the LAIF container and supersede an earlier two-repeat
> Parquet and Arrow run taken in a `sif-images` container. The directions were the same;
> the absolute values differ, which is the usual reason to state the environment.

### A caveat on comparability

These runs used 13 workers and 1000 batches, while the Part III table used 4 workers and
200. **The two tables are therefore not directly comparable**, and `compare_layouts.py`
says so - it reports `CONTROLLED_COMPARISON=false` and names `num_workers` and
`measured_batches` as the differences. To place a track beside the Part III results, re-run
it with matching settings.

### Reading the `Opens` column for a track

For loose files, `files_opened` is literally filesystem opens. For an adapter it counts
*logical sample reads*, because the adapter does not report how many times it touched the
underlying file. A Parquet run showing 64 000 "opens" against a single-file artifact is
counting samples, not opens. `filesystem_objects` is the number to read for artifact
count: 1 for Parquet and HDF5, 3 for Arrow, 50 002 for the loose tree.

## HDF5

Measured on LUMI in the LAIF container, which carries h5py 3.16.0. The `sif-images`
PyTorch containers do not, so pick your image accordingly or install h5py yourself.

Two things decide HDF5 performance, and both are exposed:

- **Chunking** (`--group-size`) is the same trade as a Parquet row group, but the per-handle
  chunk cache absorbs most of it: see the access-order result above.
- **Concurrency.** A single HDF5 handle is not safe to share across processes. The adapter
  base class opens handles lazily, once per process, keyed on pid - a handle created
  before a DataLoader fork and used in several workers gives corrupt reads or crashes,
  which is the classic HDF5-with-workers failure. The 13-worker runs above exercise this
  path, and returned zero failed samples.

## Hugging Face Datasets

The format is rarely the problem; the operational detail around it is.

- **The cache defaults to your home directory** - small, quota-tight, and the wrong
  filesystem for job I/O. Worse, in a distributed job every rank tries to build it. The
  converter prints the `HF_HOME` and `HF_DATASETS_CACHE` exports that put it on scratch.
- **Compute nodes have no internet.** `HF_DATASETS_OFFLINE=1` turns a silent hang into an
  error. The adapter sets it if you have not.
- **The progress bar is not a progress bar in a batch job.** It is tens of thousands of
  carriage-return lines in the Slurm error log, burying anything real. The converter
  disables it - the first version of this track produced a 93 KB error log for a
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
   `open_resource`, never in `__init__` - see the fork warning above.
2. Give it a `name`, so it gets its own row in comparisons.
3. Return artifact metrics from `describe()`. Keys must already exist in
   `OPTIONAL` in `dataaware/schema.py`, which rejects unknown fields, so a typo
   fails loudly.
4. Prove it returns the same bytes as the manifest.
   `test_a_track_returns_the_same_bytes_as_the_manifest` in `tests/test_adapters.py` is
   the pattern, and it is the property the whole comparison rests on.
