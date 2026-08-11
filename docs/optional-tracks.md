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

The LAIF image measured above, and the one every number on this page came from:

```text
/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif
```

Pin that path in `env.sh` rather than `lumi-multitorch-latest.sif`. The symlink moves when
a new image is published, and the library versions move with it: it pointed at a
`20260513` build carrying pyarrow 23.0.1 and datasets 4.8.5 before this run, and at a
`20260807` build after it. A run that used the symlink cannot be reproduced from the path
alone.

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

## Every format, one setting

All six representations under identical conditions: 50 000 samples, 13 workers, 1000
batches, project scratch, the pinned LAIF image above, **three repeats each**, same
manifest. Every artifact is byte-identical to the loose files by checksum, and all 27 runs
reported zero failed, duplicate, and missing samples. Figures are `min / median / max`
samples per second.

Index-addressable formats, shuffled access (`shuffle: true`):

| Format | min / median / max | vs loose files |
| ------ | ------------------ | -------------: |
| HDF5 | 11 356 / **11 723** / 13 228 | **4.5x** |
| Arrow | 8 132 / **8 240** / 12 750 | 3.2x |
| loose files | 2 516 / **2 609** / 2 618 | - |
| SquashFS | 2 260 / **2 303** / 2 640 | 0.9x |
| Parquet | 724 / **902** / 1 345 | 0.3x |

Sequential access (`shuffle: false`):

| Format | min / median / max | vs loose files |
| ------ | ------------------ | -------------: |
| Parquet | 22 076 / **22 469** / 22 914 | **8.6x** |
| tar shards | 14 160 / **15 736** / 16 482 | 6.0x |
| Arrow | 13 827 / **15 020** / 18 319 | 5.8x |
| HDF5 | 10 916 / **11 112** / 13 097 | 4.3x |

Tar shards stream by construction and have no index to permute, so they appear only in the
second table; their order comes from shard assignment and a 1000-sample shuffle buffer.

Three things fall out of reading both tables together.

**No format wins both columns.** Parquet is last under shuffling and first under sequential
reading, a 25-fold swing from one flag. If you need a full shuffle every epoch, HDF5 is the
fastest single-file artifact measured here; if you can read in order, Parquet is.

**SquashFS gains nothing at this worker count.** It matched loose files at 13 workers,
having beaten them 9-fold at 4 workers in step 3 of the tutorial. Between those two runs
the loose tree improved 6.4-fold while SquashFS did not, which says the step 3 advantage
was largely about keeping reads in flight, something extra workers also buy. SquashFS still
wins on object count, 1 against 50 002, and on measurement stability. This reversal is a
single observation across three repeats and is not explained here.

**Correctness held everywhere.** Nothing in these numbers came at the cost of a misread
sample, which is the precondition for comparing them at all.

## Access order in detail

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

**Parquet is both the fastest and the slowest representation measured - a 25x swing from
one configuration flag.** Read sequentially it beats everything else, including tar shards
at 15 736 samples/s. Read at random it is slower than a loose tree of 50 000 files.

The direction is structural; the magnitude is partly this adapter's. Reaching a random
sample may require loading and decompressing substantial row-group data, and with a
single-row-group cache shuffled access repeatedly evicts and reloads groups - about 3.4 MB
of row group per 2.7 KB sample. Sequential access pays that once per group and amortises
it over 1250 samples. A reader with a larger cache, different page layout, column
selection, or batch-oriented access would land somewhere between these numbers. Random
Parquet also has the widest spread of anything measured here, 724 to 1345 across three
repeats, which is what cache thrashing looks like.

Parquet is usable for image samples either way. It is only suitable if the access pattern
is sequential.

**HDF5 is indifferent to access order.** Its two ranges are 11 356 to 13 228 shuffled and
10 916 to 13 097 sequential - almost entirely overlapping, so the honest reading is that
the two are indistinguishable, not that shuffled is faster. h5py keeps a per-handle chunk
cache and the adapter opens one handle per worker process, so a shuffled read stream still
finds most of its samples in an already-decoded chunk. If you need full shuffling every
epoch and a single-file artifact, this is the property to want.

**Memory mapping degrades gracefully.** Arrow lost about half going from sequential to
random (15 020 to 8 240), against Parquet's 25-fold collapse. `load_from_disk` memory-maps
the file, so the operating-system page cache can serve many accesses without reloading a
whole row group. Random access still costs page faults, offset resolution, and some
decoding - it is cheaper here, not free.

> These numbers were measured in the LAIF container and supersede an earlier two-repeat
> Parquet and Arrow run taken in a `sif-images` container. The directions were the same;
> the absolute values differ, which is the usual reason to state the environment.

### A caveat on comparability

The tables on this page are internally controlled: every row shares a container, worker
count, batch count, and manifest, which is what makes them a comparison rather than a list.

They are **not** comparable with the step 3 table in the README, which used 4 workers and
200 batches in a different container. `compare_layouts.py` refuses to mix them, reporting
`CONTROLLED_COMPARISON=false` and naming `num_workers` and `measured_batches` as the
differences. That is also why the SquashFS row here disagrees with step 3: same layout,
different conditions, and the conditions are the finding.

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
