# Dataset layouts

Reference for the three layouts the tutorial compares, and a starting-point table for
formats it does not implement.

## The three core layouts

### `loose-files`

One file per sample in an ordinary directory tree. How most datasets arrive, and the
baseline everything else is measured against.

```yaml
dataset:
  layout: loose-files
  root: ${TUTORIAL_ROOT}/source
  manifest: ${TUTORIAL_ROOT}/source/manifest.jsonl
```

Every sample costs a path lookup, an open, a read, and a decode. At a few thousand
files that is unremarkable; at a million it is a metadata workload that affects other
users of a shared filesystem, not only your job.

### `squashfs`

The same tree packaged into one read-only image. **The reader is identical to
`loose-files`** - same code, same paths - because that is the property being tested. A
packaged dataset that required application changes would be a different proposition
entirely.

```yaml
dataset:
  layout: squashfs
  root: ${TUTORIAL_ROOT}/mnt/source     # where the image appears
  image: ${TUTORIAL_ROOT}/source.squashfs
  squashfs_mode: prebound
  manifest: ${TUTORIAL_ROOT}/source/manifest.jsonl
```

| Mode | Expects | Cleanup |
| ---- | ------- | ------- |
| `prebound` | The image already mounted or bound at `root` | Nothing to clean; the container owns the bind |
| `squashfuse` | `image` set; the loader mounts it | Unmounted in a `finally` block, so a failed run does not leak a FUSE mount |

`prebound` is the path LUMI documentation points to: bind the image into the container
at launch with `image-src`, which needs no privileges. `env.example.sh` shows the
`TUTORIAL_CONTAINER_BINDS` form.

**Compression.** The default `-noD -noF` stores sample bytes uncompressed. For JPEG
and PNG this is right - the bytes are already compressed, so compressing again costs
read CPU for almost no space.

Both flags are needed, and the second is easy to miss. `-noD` only disables
compression of full *data blocks*; files below the block size are stored as
*fragments*, which `-noF` covers. In a metadata-heavy dataset almost every file is a
fragment, so `-noD` alone leaves the image fully compressed. Measured on LUMI: a
50 000-file tree of 2.7 KB JPEGs packed with `-noD` alone still came out at 79 % of
source size and took five minutes, because every file went through the compressor.

For uncompressed source data (`.npy`, `.csv`, `.bin`), use `-comp zstd` instead and
expect a smaller image at the cost of decompression while reading. The inspection
report's `compression_likely_to_help` flag is a first indication of which case you
are in.

**Trade-off.** Read-only. Any change to the dataset means rebuilding the image.

### `webdataset`

Samples grouped into tar archives and read sequentially. This changes the access
pattern, not just the packaging.

```yaml
dataset:
  layout: webdataset
  root: ${TUTORIAL_ROOT}/shards
  manifest: ${TUTORIAL_ROOT}/source/manifest.jsonl
loader:
  shuffle: false          # required
  shuffle_buffer: 1000
```

Shard layout follows the WebDataset convention - members sharing a basename form one
sample:

```text
shard-00000.tar
    s00000000.jpg     the sample bytes
    s00000000.cls     its class label
```

Built on the standard library's `tarfile` rather than the `webdataset` package: no
added dependency, and shard-to-reader assignment stays in code you can read. That
assignment is what Part VI tests, so it should not be hidden in a library.

**Reader assignment** is round-robin (`shards[index::total]`). Interleaving rather than
contiguous blocks means that when shard sizes vary, large shards spread across readers
instead of concentrating in one. When there are fewer shards than readers, some
readers get nothing - and that is *not* corrected, because it is a real failure mode
Part VI makes visible. Filling idle readers by duplicating shards would hide it.

This means shard count should be at least `world_size x num_workers` **for this loader**,
which partitions at shard granularity in both dimensions. A loader that partitions samples
within a shard, or schedules shards dynamically, has a different requirement. Divisibility
alone does not guarantee balance when shard sizes differ.

**Shuffling** is weaker than a map-style dataset's. There is no index to permute, so
order comes from shard order (fixed by the seed) plus `shuffle_buffer`, which shuffles
within a window. Shard order is deliberately *not* varied per epoch: repeated epochs
stay comparable, which is what this benchmark needs.

**Balancing.** `--balance-by count` gives shards equal sample counts;
`--balance-by work` gives them similar total `estimated_decode_cost` via greedy
longest-processing-time-first packing. Equal counts do not mean equal work, and with
synchronised ranks the slowest shard sets the pace.

**Coverage accounting.** `drop_last` behaves differently here. A map-style dataset
batches one index stream and drops exactly `total % batch_size` samples. A streaming
dataset gives each worker its own stream, so *each* worker may drop up to
`batch_size - 1`. Which ones is not knowable in advance, so the shortfall is expressed
as an allowance rather than a lowered expectation - see `coverage_expectation` in
`dataaware/loaders.py`.

## Choosing between them

| Requirement | Candidate |
| ----------- | --------- |
| Ordinary filenames, directory traversal | `squashfs` |
| Arbitrary path-based random access | `squashfs` |
| Sequential sample streaming | `webdataset` |
| Explicit rank-level shard assignment | `webdataset` |
| Full-dataset shuffle every epoch | `squashfs`; a buffer is weaker |
| Frequent updates to individual records | Neither; both are immutable |
| Small dataset, exploratory work | `loose-files` - do not add machinery you do not need |

## Formats this tutorial does not implement

A starting point, not an authoritative compatibility list. Nothing here has been
measured by this repository.

| Layout or format | Appropriate use | Main advantage | Main limitation |
| ---------------- | --------------- | -------------- | --------------- |
| Loose files | Small datasets, exploration | Simple, widely compatible | Poor at very large file counts |
| SquashFS | Immutable datasets needing ordinary paths | Many files become one image | Read-only; rebuild after changes |
| Tar/WebDataset | Independent samples consumed as a stream | Natural sharding, sequential access | Needs compatible loader logic |
| Parquet | Tabular or record-oriented data | Column selection, compression | Unnatural for arbitrary file trees |
| Arrow / Hugging Face Datasets | NLP and multimodal ecosystems | Schema, memory mapping, integration | Cache behaviour needs coordinating |
| HDF5 | Dense arrays, scientific data | Chunking, compression, structure | Concurrent access needs care |
| NetCDF | Scientific multidimensional data | Rich metadata, ecosystem | Specialised |
| Zarr | Chunked arrays, object storage | Flexible chunk access | Poor chunk sizing recreates the small-object problem |
| NPY/NPZ | Small NumPy intermediates | Simple | Limited at dataset scale |
| SQLite/LMDB | Indexed local access | Convenient key-value reads | Concurrency and locking need testing |

## Adding a layout

1. Add its name to `LAYOUTS` in `dataaware/config.py`, with any fields it needs and
   their validation.
2. Handle it in `prepared_layout` (making data readable, plus cleanup) and
   `build_dataset` in `dataaware/loaders.py`.
3. Return layout metrics from `prepared_layout`; add any new metric names to
   `OPTIONAL` in `dataaware/schema.py`, which rejects unknown fields.
4. Prove it reads the same bytes as the existing layouts. `test_all_layouts_return_identical_sample_bytes`
   in `tests/test_loader.py` is the pattern - a layout that reads different data cannot
   be compared, however fast it is.
