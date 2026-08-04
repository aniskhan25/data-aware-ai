# Data-Aware AI on LUMI

A hands-on tutorial for deciding how an AI dataset should be stored, packaged,
staged, and read on LUMI.

> Is your dataset ready for efficient training?

This repository helps AI and HPC users answer that question with evidence. The goal
is not to benchmark LUMI storage. The goal is to spend small jobs to avoid wasting
large jobs: inspect the dataset, compare controlled alternatives, diagnose the input
path, and produce a defensible data-readiness decision.

Use this repository when you want to:

- identify whether many small files are creating a poor dataset layout;
- compare loose files, SquashFS, and tar-based shards;
- tune data-loading workers relative to allocated CPUs;
- compare shared storage with node-local staging;
- verify that distributed ranks read unique and balanced samples;
- determine whether the input pipeline is ready for a larger AI job.

> Data layout is a workload decision, not a file-extension decision.
> Do not scale a workload that cannot feed its current allocation.

**Status: early development.** Parts I to IV are implemented and runnable end to
end, with results measured on LUMI. Later parts are marked below with the release that adds them.
See [Development status](#15-development-status).

---

## 1. What decision will this tutorial help you make?

A single decision, supported by measurements you took yourself:

> Given this dataset and this workload, what layout, storage location, and worker
> count should I use — and is the input path ready for a larger job?

The core principle:

> A dataset that can be read successfully is not necessarily a dataset that can be
> read efficiently at scale.

The method, one rung at a time:

```text
Inspect the dataset
        ↓
Loose-file baseline
        ↓
Read-only packaged dataset
        ↓
Sharded streaming dataset
        ↓
Storage-placement comparison
        ↓
Distributed-reader validation
        ↓
Data-readiness decision
```

Each rung answers one question. You are not expected to adopt every rung — a rung
that shows no improvement is a result, and staying with the simpler configuration is
a legitimate outcome.

| Rung                | Question                                                           |
| ------------------- | ------------------------------------------------------------------ |
| Dataset inspection  | What kind of data layout do I currently have?                      |
| Loose-file baseline | What does the unmodified dataset cost?                             |
| SquashFS            | Does packaging small files reduce overhead while preserving paths?  |
| Tar shards          | Does sample streaming improve throughput and distribution?         |
| Worker tuning       | Can the CPU pipeline feed the workload efficiently?                |
| Storage placement   | Does scratch, flash, or node-local staging fit this workload?       |
| Distributed reading | Are ranks receiving unique and balanced samples?                   |
| Final decision      | Is the data path ready for larger AI jobs?                         |

This tutorial complements the short data lesson in the
[LUMI AI Guide](https://github.com/aniskhan25/LUMI-AI-Guide/tree/main/2-data),
which stays as concise onboarding material. It follows the teaching style of
[Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai): a
controlled baseline, one variable at a time, deliberate bottlenecks, and
machine-readable evidence. Complete the data checks here before interpreting poor
GPU scaling as a communication or compute problem.

---

## 2. LUMI storage vocabulary

| Term | What it is | Used here for |
| ---- | ---------- | ------------- |
| Project scratch | Large parallel (Lustre) space for active job I/O | The default location for datasets and results |
| Project flash | Smaller, faster parallel space for workloads needing faster disk operations | An optional Part V comparison |
| Compute-node `/tmp` | Node-local space that lives in memory and consumes the job's allocated memory | The node-local staging experiment |
| LUMI-O | S3-compatible object storage | Staging, sharing, and secondary copies (optional) |

Three facts shape every experiment here: project scratch is the main location for
job input, output, and checkpoint I/O; compute-node `/tmp` is memory, so staging
there spends the job's memory allocation; and large numbers of small files create
pressure on filesystem metadata services, which affects other users of a shared
system too.

Longer version, including the usable / suitable / scalable distinction the tutorial
relies on: [`docs/storage-locations.md`](docs/storage-locations.md). The authority on
quotas, lifetimes, and policy is the official documentation:
<https://docs.lumi-supercomputer.eu/storage/>.

---

## 3. Setup

### 3.1 On LUMI

```bash
git clone https://github.com/aniskhan25/data-aware-ai.git
cd data-aware-ai

cp env.example.sh env.sh
$EDITOR env.sh          # set LUMI_PROJECT; TUTORIAL_ROOT follows from it
```

`env.sh` holds your project allocation and is gitignored. Nothing in this repository
contains a project ID, and nothing you commit should.

The workload needs Python with NumPy, Pillow, PyYAML, and PyTorch. Use a provided
PyTorch container or module rather than building a conda environment on Lustre — an
environment is tens of thousands of small files, which is the metadata pressure this
tutorial exists to teach you about. Point `TUTORIAL_CONTAINER` at a container in
`env.sh` and the job scripts will use it.

Then generate the dataset once:

```bash
sbatch jobs/prepare_dataset.sh configs/datasets/balanced.yaml
```

Generation is deterministic: the same profile and seed always produce the same bytes
and the same manifest, on any machine.

### 3.2 Locally, without LUMI

The whole pipeline runs on a laptop against a tiny dataset. Use this to learn the
tooling; the numbers it produces are not meaningful performance results.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[loader,dev]'

export DAAI_TEST_ROOT=/tmp/daai-test
python scripts/generate_dataset.py --profile tiny --output "$DAAI_TEST_ROOT"
python scripts/benchmark_loader.py --config configs/test/tiny.yaml
pytest -q
```

---

## 4. Part I: Inspect the dataset

**Question: what kind of data layout do I currently have?**

This is the cheapest experiment here. It reads metadata only — it never opens a
file — so it needs no GPU and no PyTorch. Do it before allocating anything
expensive.

### Run it

```bash
python scripts/inspect_dataset.py --path "$TUTORIAL_ROOT/source" --verbose
```

For a large tree, or when you want the node-local staging advice to mean anything,
run it as a job instead:

```bash
sbatch jobs/inspect_dataset.sh
```

Walking millions of entries is a sustained metadata workload that login nodes should
not carry. And the staging advice compares the dataset against your job's memory
allocation, so it is only meaningful **inside the allocation you intend to train
with** — otherwise pass `--memory-bytes`.

### Expected output shape

```text
DATASET_PATH=/scratch/project_XXXXXXXXX/data-aware-ai/source
TOTAL_FILES=50000
TOTAL_BYTES=...
ESTIMATED_DATASET_GIB=...
MEDIAN_FILE_BYTES=...
P5_FILE_BYTES=...
P95_FILE_BYTES=...
P95_TO_MEDIAN_RATIO=...
FILE_SIZE_CV=...
SMALL_FILE_THRESHOLD_BYTES=65536
FILES_UNDER_THRESHOLD=...
SMALL_FILE_FRACTION=...
DIRECTORIES=...
MAX_DIRECTORY_DEPTH=...
MAX_FILES_IN_ONE_DIRECTORY=...
FILESYSTEM_OBJECTS=...
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
REPORT_PATH=outputs/inspection/dataset_report.json
```

`--verbose` adds the size histogram, the extension mix, each candidate's reasoning,
and the list of things the tool cannot determine.

### Metrics to inspect

| Metric | What it tells you |
| ------ | ----------------- |
| `TOTAL_FILES` and `FILESYSTEM_OBJECTS` | The metadata footprint. This is what packaging collapses |
| `SMALL_FILE_FRACTION` | Whether per-file overhead, rather than bytes, dominates reads |
| `MEDIAN_FILE_BYTES` with `P5`/`P95` | The bulk of the distribution, not just its average |
| `P95_TO_MEDIAN_RATIO` | Whether equal sample counts per shard would mean equal work |
| `MAX_FILES_IN_ONE_DIRECTORY` | Whether any single directory is a hotspot |
| `DATASET_FRACTION_OF_MEMORY` | Whether node-local staging is even possible |
| `UNREADABLE_*` | Whether the report covers the whole tree |

### Interpretation

| Observation | Suggested next experiment |
| ----------- | ------------------------- |
| Few large files | Benchmark the native representation |
| Many small immutable files | Compare loose files with SquashFS |
| Independent training samples | Compare with tar shards |
| Large variation in sample size | Test shard balancing |
| Dataset fits comfortably in allocated memory | Test node-local staging |
| Structured records (`.parquet`, `.arrow`) | Consider the Parquet or Hugging Face track |
| Dense multidimensional arrays (`.h5`, `.nc`) | Consider the HDF5 track |

The tool prints these as `CANDIDATE_EXPERIMENTS`, each with the observation that
motivated it.

### Common misinterpretations

- **"It suggested SquashFS, so I should use SquashFS."** It suggested *measuring*
  SquashFS. The candidates are hypotheses; Part III decides.
- **"`FILE_SIZE_CV` is high, so my samples vary wildly."** The coefficient of
  variation is dominated by outliers — one stray manifest among uniform images
  inflates it. `P95_TO_MEDIAN_RATIO` is the robust measure, and it is what drives
  the shard-balancing suggestion.
- **"No staging advice appeared, so staging is fine."** A `null` means allocated
  memory was unknown, not that staging is safe. Re-run inside the allocation.
- **"The report says my dataset is 400 GB."** If `UNREADABLE_DIRECTORIES` is
  non-zero, it says the *readable part* is 400 GB.

### Important limitation

The inspector cannot infer application semantics. It does not know whether random
access is required, whether sample ordering matters, whether records are mutable,
whether an ecosystem mandates a format, or how expensive a record is to decode. It
proposes experiments; you and the measurements decide.

### Decision

Note the candidates and continue to Part II. Even when inspection suggests
packaging, the baseline comes first — there is nothing to compare against without
it.

Full report schema: [`docs/inspection-report.md`](docs/inspection-report.md).

---

## 5. Part II: Establish a loose-file baseline

**Question: what does the unmodified dataset cost?**

Loose files are how most datasets arrive, and every later comparison is measured
against this run. Each sample performs a manifest lookup, a file open, a byte read,
an image decode, batch assembly, and a small synthetic compute step.

### Run it

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

Locally:

```bash
python scripts/benchmark_loader.py --config configs/test/tiny.yaml
```

### What changes, what stays fixed

Nothing changes yet — this is the reference point. From Part III onward, exactly one
thing changes per experiment: the dataset representation, or the worker count, or the
storage location. The manifest, sample order, decoder, batch size, CPU allocation,
and measurement length stay fixed.

### Expected output shape

```text
RUN_SUMMARY=.../outputs/loose-files/run_summary.json
LAYOUT=loose-files
STORAGE=scratch
WORLD_SIZE=1
NUM_WORKERS=4
SAMPLES_MEASURED=12800
BYTES_READ=...
SAMPLES_PER_SECOND=...
MIB_PER_SECOND=...
STARTUP_SECONDS=...
MEAN_BATCH_WAIT_SECONDS=...
P95_BATCH_WAIT_SECONDS=...
MEAN_DATA_WAIT_FRACTION=...
FAILED_SAMPLES=0
DUPLICATE_SAMPLES=0
MISSING_SAMPLES=0
```

The full summary, including the resolved configuration and provenance, is written to
`run_summary.json`. Re-read any stored summary without re-running anything:

```bash
python scripts/summarize_run.py outputs/loose-files/run_summary.json
```

### Metrics to inspect

| Metric | What it tells you |
| ------ | ----------------- |
| `samples_per_second` | Useful throughput. The headline number |
| `mib_per_second` | Byte rate. Context, never the conclusion |
| `mean_data_wait_fraction` | Share of the loop spent waiting for data |
| `p95_batch_wait_seconds` | Whether stalls are occasional and severe |
| `startup_seconds` | Cost paid once per job, before any samples |
| `failed_samples` | Must be 0. Anything else invalidates the comparison |
| `duplicate_samples` | Repeats within one epoch. Must be 0 |
| `missing_samples` | Samples a complete epoch never covered. Must be 0 |

### Interpretation

A high `mean_data_wait_fraction` here does **not** mean your real training job will
be 95 % data-bound. The synthetic compute step is deliberately far cheaper than a
GPU training step, so the same input pipeline shows a lower fraction in a real job.
The fraction is for comparing layouts under an identical compute step — not for
predicting a training job.

Read `p95_batch_wait_seconds` alongside the mean. A pipeline that is usually fine but
stalls badly one batch in twenty shows an acceptable-looking mean.

### Common misinterpretations

- **"One run showed X, so X is true."** Results on a shared system vary with load.
  Run comparisons more than once and look at the spread. A difference smaller than
  the run-to-run variation is not a difference.
- **"This measures LUMI's storage."** It measures one application's input path,
  under unknown concurrent load from other users.
- **"The tiny profile shows loose files are fine."** The tiny dataset fits in page
  cache. It proves the plumbing works and nothing else.

### Decision

Record this run. If `failed_samples` is non-zero, fix that before measuring anything
else — a run that could not read its data is not a baseline. Otherwise continue to
Part III.

Methodology in full, including how duplicates and missing samples are counted:
[`docs/measurement-methodology.md`](docs/measurement-methodology.md).

---

## 6. Part III: Compare dataset layouts

Two challenges, each changing exactly one thing about how the same samples are
stored. The manifest, decoder, batch size, worker count, and measurement length stay
fixed throughout, so the comparison isolates the representation.

All three layouts read the same manifest, and the sample bytes are verified
identical across them. Without that, a throughput difference could just be a
difference in what was read.

### 6.1 Challenge A: loose files versus SquashFS

**Question: does packaging an immutable file tree reduce overhead while preserving
ordinary path-based access?**

Build the image, then measure it:

```bash
sbatch jobs/build_squashfs.sh
sbatch jobs/run_loader.sh configs/baseline/squashfs.yaml
```

The image holds the same tree with the same paths, so **the loader code is
identical** — `loose-files` and `squashfs` share an implementation on purpose. Only
`dataset.root` and the object count change. That is the property being tested: a
packaged dataset that needed application changes would be a different proposition.

Two ways to make the image readable, both set in the config:

| `dataset.squashfs_mode` | What it expects | When to use it |
| ----------------------- | --------------- | -------------- |
| `prebound` (default) | The image already mounted or bound at `dataset.root` | Inside a container — bind it with `image-src`. This is the path LUMI documentation points to |
| `squashfuse` | `dataset.image` pointing at the file | Outside a container. The loader mounts it and always unmounts it, including on failure |

Compression is a real choice, not a detail. The default (`-noD`) stores data blocks
uncompressed, because the tutorial dataset is JPEG and PNG whose bytes are already
compressed — recompressing them spends CPU on every read for almost nothing. For
uncompressed source data (`.npy`, `.csv`, `.bin`), `-comp zstd` gives a smaller image
at the cost of decompression while reading.

#### Interpretation

| Observation | Interpretation |
| ----------- | -------------- |
| Higher throughput and lower waits | Loose-file access was a meaningful cost |
| Similar throughput, far fewer objects | Still likely preferable operationally: it stops pressuring metadata services other users share |
| Higher startup, better steady state | Separate the mount cost from repeated-epoch performance. Check `startup_seconds` and `mount_seconds` |
| No improvement | Decoding or another layer dominates. Part IV will say which |
| Worse | The access pattern does not suit packaging |

SquashFS is a candidate when the dataset is read-only and the application needs
ordinary filenames. It is not a universally optimal AI format, and a rebuild is
needed after any change.

### 6.2 Challenge B: SquashFS versus tar shards

**Question: does the workload need a filesystem-like view, or can it consume a
stream of samples?**

```bash
sbatch jobs/build_webdataset.sh
sbatch jobs/run_loader.sh configs/baseline/webdataset.yaml
```

This changes the access pattern, not just the packaging. Samples are read
sequentially from tar archives, and shards are assigned to readers round-robin —
which is what lets many readers cover a dataset without any two reading the same
bytes. That assignment is implemented in `dataaware/shards.py` rather than hidden in
a library, because Part VI tests it directly.

`loader.shuffle` **must be false** for this layout, and the configuration rejects
`true`. A streaming layout has no index to permute: order comes from shard order
plus `loader.shuffle_buffer`, which shuffles within a window rather than across the
dataset. That is a genuinely weaker shuffle, and it is part of what you are choosing.

Additional metrics reported here:

```text
NUM_SHARDS=...            SHARD_BYTES_CV=...
SAMPLES_PER_SHARD=...     SHARD_WORK_CV=...
SHARD_OPENS=...           SHARD_OPEN_SECONDS=...
```

`SHARD_OPENS` is the honest headline: `files_opened` counts one open per shard here,
against one per sample for a loose tree.

#### Interpretation

| Requirement | Likely candidate |
| ----------- | ---------------- |
| Ordinary filenames and directory traversal | SquashFS |
| Sequential training-sample streaming | Tar shards |
| Arbitrary path-based lookup | SquashFS |
| Explicit rank-level shard assignment | Tar shards |
| Frequent updates to individual records | Neither immutable layout is ideal |
| Full-dataset shuffling each epoch | SquashFS; a shuffle buffer is weaker |

### 6.3 Compare them

```bash
python scripts/compare_layouts.py \
    outputs/loose-files/run_summary.json \
    outputs/squashfs/run_summary.json \
    outputs/webdataset/run_summary.json
```

Pass several summaries per layout to get medians and spread instead of single
numbers — `outputs/*/run_summary.json` works.

The tool distinguishes two kinds of mismatch, because they are not equally serious:

- **Blocking** — a different `manifest_hash` or schema version. The runs did not read
  the same data, so no table from them means anything. The comparison stops with exit
  code 3.
- **Uncontrolled** — same data, but a differing batch size, worker count, seed, or
  measurement length. The numbers still mean something individually, so they are
  shown with `CONTROLLED_COMPARISON=false` and a loud caution.

Correctness is checked before performance. A group with failed, duplicate, or missing
samples is called out explicitly, because throughput inflated by redundant work is
not throughput.

Real output, measured on LUMI project scratch with the `metadata-heavy` profile
(50 000 files of about 2.7 KB), 4 workers on 7 cores, 200 measured batches of 64,
**three repeats per layout**:

```text
--- read this before the numbers ---
! Throughput varied by more than 10 % between repeats for loose-files,
  webdataset. Treat differences of that size as noise.

| Layout        | Runs | Samples/s | MiB/s | Mean wait | P95 wait | Opens | FS objects |
|---------------|------|-----------|-------|-----------|----------|-------|------------|
| loose-files * | 3    | 405.1     | 1.033 | 0.1572    | 0.7345   | 12800 | 50000      |
| squashfs      | 3    | 3652      | 9.312 | 0.01665   | 0.05455  | 12800 | 1          |
| webdataset    | 3    | 6926      | 17.66 | 0.008237  | 0.03308  | 9     | 51         |

--- squashfs against loose-files ---
THROUGHPUT_CHANGE_PERCENT=+801.4
WAIT_CHANGE_PERCENT=-89.41
STARTUP_CHANGE_PERCENT=+308.3
FILESYSTEM_OBJECT_REDUCTION=5e+04x

--- webdataset against loose-files ---
THROUGHPUT_CHANGE_PERCENT=+1610
WAIT_CHANGE_PERCENT=-94.76
STARTUP_CHANGE_PERCENT=+53.88
FILESYSTEM_OBJECT_REDUCTION=980.4x
```

Columns are medians. The spread behind them is where the lessons are:

| Layout | Median | Min | Max | CV | Startup (CV) |
| ------ | ------ | --- | --- | -- | ------------ |
| loose-files | 405 | 363 | **1582** | **0.88** | 0.58 s (0.79) |
| squashfs | 3652 | 3430 | 4048 | 0.08 | 2.38 s (0.01) |
| webdataset | 6926 | 5572 | 7403 | 0.14 | 0.90 s (0.21) |

**Loose files are not just slow, they are unpredictable.** One of the three repeats
reached 1582 samples/s against a median of 405 — almost certainly page cache, since a
previous job had just read the same 50 000 files on that node. A coefficient of
variation of 0.88 means *any single* loose-file measurement on a shared parallel
filesystem is close to meaningless. This is why the tutorial insists on repeats and
on separating cold from warm behaviour; it is not hedging.

**Packaging buys stability as well as speed.** SquashFS is the most reproducible
layout here (CV 0.08) because it reads one object instead of fifty thousand. That
argument for packaging does not appear in a throughput number at all.

**Do not over-read the winner.** Shards beat SquashFS by 1.9x on medians, not the 3.2x
a single pair of runs suggested. That gap is real but modest, and the choice should
also weigh shuffling quality (shards give only a buffer shuffle), path-based access,
and SquashFS's consistent ~2.4 s mount cost. Note too that `squashfs` still performs
one open per sample — inside the image rather than on Lustre.

What the evidence does support without qualification: **loose files are unusable for
this dataset**, and both packaged layouts fix that decisively.

### Common misinterpretations

- **"Shards won, so use shards."** Shards won *on this dataset, at this scale, with
  this access pattern, in one run*. Repeat the comparison, and check the requirement
  table above: shuffling quality and path-based access are not throughput.
- **"SquashFS showed no gain, so packaging is pointless."** Collapsing tens of
  thousands of objects into one still reduces metadata pressure on a filesystem other
  users share. That is an operational argument the throughput number does not capture.
- **"The comparison printed a table, so it is controlled."** Check
  `CONTROLLED_COMPARISON` and read the cautions.

### Decision

Pick the layout whose measured behaviour *and* access-pattern requirements fit your
workload, then carry it into Part IV. Staying with loose files is a legitimate
outcome if packaging showed no benefit.

Layout reference: [`docs/dataset-layouts.md`](docs/dataset-layouts.md).

---

## 7. Part IV: Tune the input workers

**Question: can the CPU pipeline feed the workload efficiently?**

Adding DataLoader workers helps until something else saturates. This part finds where
that is, and — more usefully — names *what* saturated, because "use 13 workers" is
only actionable if you know whether the limit was storage, CPU, or memory.

### Run it

```bash
source env.sh                       # sbatch resolves the account at submission time
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
```

That submits one job per rung, twice each. The rungs are `0, 2, 7, 13,
oversubscribed(28)`, and only `loader.num_workers` changes between them.

Two arguments worth understanding. The layout should be the one Part III chose — the
script sets that layout's root and loader settings together, since getting only half
right points the run at the wrong data. And **lengthen the measured window**: at
12 000 samples/s the default 200 batches is under a second of measurement, which is
far too short to be stable.

### Read it

```bash
python3 scripts/compare_workers.py \
    "$TUTORIAL_ROOT"/outputs/workers/*-webdataset-r*/run_summary.json
```

Real output, measured on LUMI with tar shards on project scratch, `--cpus-per-task=7`,
1000 measured batches, 2 repeats per rung:

```text
--- read this before the numbers ---
! Throughput varied by more than 10 % between repeats at [0] workers. The shape
  of the ladder there is not reliable.
! Rungs [28] request more processes than the 14 allocated CPUs. Their results
  describe an oversubscribed pipeline, which is the point of that rung but not a
  configuration to adopt.

| Workers | Runs | Samples/s | Mean wait | P95 wait | Wait frac | CPU util | Peak MiB | Invol cs/s |
|---------|------|-----------|-----------|----------|-----------|----------|----------|------------|
| 0       | 2    | 2618      | 0.02605   | 0.02944  | 0.9815    | 0.0679   | 534.5    | 4.185      |
| 2       | 2    | 5060      | 0.01172   | 0.02746  | 0.9265    | 0.0152   | 566.8    | 10.88      |
| 7       | 2    | 12050     | 0.004281  | 0.01963  | 0.8061    | 0.0392   | 567.1    | 86.97      |
| 13 *    | 2    | 13760     | 0.003527  | 0.007863 | 0.7568    | 0.0483   | 551.7    | 2219       |
| 28      | 2    | 13590     | 0.003614  | 0.009271 | 0.7677    | 0.0453   | 556.9    | 1160       |

LADDER_PATTERN=plateau
BEST_WORKERS=13
BEST_AFFORDABLE_WORKERS=13
RECOMMENDED_WORKERS=13
MAIN_LIMITING_FACTOR=storage-or-synchronisation
```

### `--cpus-per-task=7` gives you 14 CPUs, not 7

The caution above says "the 14 allocated CPUs" for a job that asked for 7. That is
not a bug: LUMI counts *logical* CPUs, and simultaneous multithreading means 7 cores
present as 14 hardware threads. `cpus_available` reports what the affinity mask
actually allows, which is what worker advice has to be measured against.

This is why the ladder's top affordable rung is **13**: 13 workers plus the main
process exactly fills 14. A rung labelled "one worker per core" would in fact be
using half the allocation.

### Interpretation

| Observation | Action |
| ----------- | ------ |
| Throughput rises and waits fall | Additional workers are useful |
| Throughput plateaus | Stop adding workers |
| Memory rises without throughput benefit | Reduce workers or prefetch depth |
| Throughput falls | The pipeline is oversubscribed |
| CPU saturated | Optimise decoding or allocate more CPUs per rank |
| CPU idle but waits still high | Investigate storage or synchronisation, not workers |

The tool classifies the ladder as `still-improving`, `plateau`, `regression`, or
`flat`, and picks the **cheapest** rung within 5 % of the best — not the fastest one.
Each worker is a process holding memory, and buying 2 % with four times the processes
is a bad trade on a shared node.

It will also **never recommend an oversubscribed rung**, even when that rung measures
fastest. An earlier run of this exact ladder had 28 workers winning outright; adopting
it would have contradicted the caution printed about it and borrowed CPU from
everything else on the node.

### What this ladder actually found

Workers took throughput from 2618 to 13 760 samples/s — a **5.3x** gain, and the
single largest tuning win in the tutorial so far. But look at the two columns that
explain it: **CPU utilisation never exceeded 7 %**, while the wait fraction only fell
from 98 % to 76 %.

That combination is the diagnosis. The pipeline was never CPU-bound; it was bound by
*I/O latency*, and extra workers helped by keeping more reads in flight, not by
supplying more compute. It is also why the gain flattens at 13 rather than collapsing:
the workers are mostly blocked, so they do not fight each other for cores much.

The involuntary context-switch column is where oversubscription shows itself: 87/s at
7 workers, 2219/s at 13. Throughput still improved — but that scheduling pressure is
paid by everything else sharing the node, and it is invisible in a samples/s number.

**The handoff matters more than the number.** With CPU idle and 76 % of the loop still
spent waiting, the remaining bottleneck is not the worker count. That is exactly what
Part V exists to test.

### Common misinterpretations

- **"28 workers was faster once, so oversubscription is fine."** It was faster in a
  short-window run and no faster in a long one, while multiplying scheduling pressure.
  A result that only appears in a one-second measurement is not a result.
- **"CPU utilisation is 5 %, so the node is idle and I should add work."** The CPUs are
  idle *because* the pipeline is waiting on storage. Adding compute would not help.
- **"More workers always helped here, so more workers always help."** This dataset is
  small-file and I/O-latency-bound. A decode-heavy dataset saturates CPU and turns
  over much earlier — try `configs/datasets/decode_heavy.yaml` and watch
  `MAIN_LIMITING_FACTOR` change to `cpu-decode`.

### Decision

Adopt `RECOMMENDED_WORKERS` for the layout you chose, then read
`MAIN_LIMITING_FACTOR`. If it points at storage, continue to Part V.

---

## 8. Part V: Compare storage placement

*Planned. Project scratch against node-local `/tmp`, with flash as an option —
including staging cost in the comparison and calculating break-even epochs. Memory
safety checks refuse unsafe `/tmp` staging.*

---

## 9. Part VI: Validate distributed reading

*Planned. Eight ranks on one node, plus three deliberately broken cases: too few
shards, duplicate samples, and imbalanced shards.*

---

## 10. Part VII: Produce a data-readiness decision

*Planned. Turn the collected summaries into a written recommendation with a
readiness state of `READY`, `READY_WITH_CAUTION`, `NOT_READY`, or `INCONCLUSIVE`.*

---

## 11. Optional format tracks

*Planned. Hugging Face Datasets, Parquet, and HDF5 examples, plus a LUMI-O staging
lifecycle exercise. None of these are required to complete the core tutorial.*

---

## 12. Adapting the tutorial to your own dataset

*Planned. A `DatasetAdapter` interface and a user-manifest workflow, so the method
applies beyond the synthetic example.*

Already usable today: the manifest format is not tied to the generator. Write a JSON
Lines manifest with the fields documented in [`data/README.md`](data/README.md),
point a configuration at it and at your dataset root, and the loose-file baseline
will measure your data.

---

## 13. Repository layout

```text
data-aware-ai/
├── configs/          Reproducible experiment definitions
│   ├── baseline/     Layout experiments (Parts II and III)
│   ├── datasets/     Dataset generation profiles
│   └── test/         Local smoke-test configuration
├── dataaware/        Shared library: config, manifest, schema, generation, metrics
├── jobs/             Slurm launch scripts
├── scripts/          Command line entry points
├── data/             Small manifests only; generated datasets are not in Git
├── docs/             Supporting reference material
├── outputs/          Machine-readable run results
├── logs/             Slurm output and error logs
└── tests/            Local correctness tests
```

Every experiment writes one schema-validated JSON summary. Comparison and reporting
tools read only summaries, never logs.

---

## 14. Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| `undefined environment variable(s) TUTORIAL_ROOT` | No `env.sh`. Copy `env.example.sh` and edit it |
| `unknown option(s) ['num_worker'] in section 'loader'` | A typo. Unknown fields are rejected on purpose, so you never measure a configuration nobody chose |
| `manifest not found` | Generate the dataset first: `sbatch jobs/prepare_dataset.sh` |
| `dataset.layout 'squashfs' is not implemented` | Correct for this release. See [Development status](#15-development-status) |
| `PyTorch is required to run the loader benchmark` | Install `.[loader]`, load a PyTorch module, or set `TUTORIAL_CONTAINER` |
| `path does not exist` from the inspector | Check the path; the dataset may still need generating |
| Inspection reports `CANDIDATE_EXPERIMENTS=none` | No files were found under that path |
| No `DATASET_FRACTION_OF_MEMORY` line | Allocated memory was unknown. Run inside your job allocation, or pass `--memory-bytes` |
| Slurm job produces no log file | `logs/` must exist before submitting. It is in the repository; do not delete it |
| `WARNING n sample(s) failed to load` | The run is not a valid comparison. Check the dataset is complete and readable |
| `... is not empty; pass --overwrite` | Regenerating over an existing dataset needs `--overwrite` |

---

## 15. Development status

| Part | Status |
| ---- | ------ |
| Repository foundation, config, schema, generator, tests, CI | **Done** |
| Part I: dataset inspection | **Done** |
| Part II: loose-file baseline | **Done** |
| Part III: SquashFS and tar shards | **Done** |
| Part IV: worker tuning | **Done** |
| Part V: storage placement and staging | Planned |
| Part VI: distributed validation and broken cases | Planned |
| Part VII: readiness decision | Planned |
| Optional tracks: Hugging Face, Parquet, HDF5, LUMI-O | Planned |

The run-summary schema (version `1.0`) is already designed for the later parts, so
summaries produced now stay readable by the comparison tools that arrive with them.

---

## 16. References

- LUMI data storage options: <https://docs.lumi-supercomputer.eu/storage/>
- LUMI Lustre documentation: <https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/>
- LUMI SquashFS and FUSE: <https://docs.lumi-supercomputer.eu/storage/formats/FUSE/>
- LUMI-O overview: <https://docs.lumi-supercomputer.eu/storage/lumio/>
- Managing data in LUMI-O: <https://docs.lumi-supercomputer.eu/storage/lumio/clients-general/>
- Moving data to and from LUMI: <https://docs.lumi-supercomputer.eu/firststeps/movingdata/>
- LUMI AI Guide data lesson: <https://github.com/aniskhan25/LUMI-AI-Guide/tree/main/2-data>
- Scaling-Aware AI on LUMI: <https://github.com/aniskhan25/scaling-aware-ai>

## License

MIT. See [LICENSE](LICENSE).
