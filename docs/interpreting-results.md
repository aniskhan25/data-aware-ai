# Interpreting results

What each part's numbers can and cannot support, and the misreadings that cost the most.
The README carries the headline findings; this is the detail behind them.

A rule that applies to every part: **a difference smaller than the run-to-run variation is
not a difference.** The tools say so where they can, but they cannot say it for a
comparison you make in your head.

---

## Part I — inspection

| Observation | Suggested next experiment |
| ----------- | ------------------------- |
| Few large files | Benchmark the native representation |
| Many small immutable files | Compare loose files with SquashFS |
| Independent training samples | Compare with tar shards |
| Large variation in sample size | Test shard balancing |
| Dataset fits comfortably in allocated memory | Test node-local staging |
| Structured records (`.parquet`, `.arrow`) | Consider the Parquet or Hugging Face track |
| Dense multidimensional arrays (`.h5`, `.nc`) | Consider the HDF5 track |

**Misreadings**

- *"It suggested SquashFS, so use SquashFS."* It suggested **measuring** SquashFS. Part III
  decides.
- *"`FILE_SIZE_CV` is high, so my samples vary wildly."* CV is dominated by outliers — one
  stray manifest among uniform images inflates it. `P95_TO_MEDIAN_RATIO` is the robust
  measure and is what drives the shard-balancing suggestion.
- *"No staging advice appeared, so staging is safe."* A null means allocated memory was
  unknown, not that staging is fine. Re-run inside the allocation.
- *"My dataset is 400 GB."* If `UNREADABLE_DIRECTORIES` is non-zero, it says the *readable
  part* is 400 GB.

---

## Part II — baseline

| Metric | What it tells you |
| ------ | ----------------- |
| `samples_per_second` | Useful throughput. The headline |
| `mib_per_second` | Byte rate. Context, never the conclusion |
| `mean_data_wait_fraction` | Share of the loop spent waiting for data |
| `p95_batch_wait_seconds` | Whether stalls are occasional and severe |
| `startup_seconds` | Paid once per job, before any samples |
| `failed` / `duplicate` / `missing_samples` | Must all be 0, or this is not a baseline |

**Misreadings**

- *"The wait fraction is 99 %, so my training will be 99 % data-bound."* The synthetic
  compute step is deliberately far cheaper than a GPU training step, so a real job shows a
  lower fraction. The fraction compares layouts under an identical compute step; it does
  not predict a training job.
- *"One run showed X."* Results vary with system load. Repeat before acting.
- *"This measures LUMI's storage."* It measures one application's input path under unknown
  concurrent load.

---

## Part III — layouts

### SquashFS

| Observation | Interpretation |
| ----------- | -------------- |
| Higher throughput and lower waits | Loose-file access was a meaningful cost |
| Similar throughput, far fewer objects | Still likely preferable: it stops pressuring metadata services others share |
| Higher startup, better steady state | Separate mount cost from repeated-epoch performance |
| No improvement | Decoding or another layer dominates; Part IV will say which |
| Worse | The access pattern does not suit packaging |

Two ways to make an image readable: `prebound` (already mounted or bound at
`dataset.root` — what a container bind produces, and the path LUMI documentation points
to) and `squashfuse` (the loader mounts and always unmounts it, including on failure).

### Tar shards

| Requirement | Likely candidate |
| ----------- | ---------------- |
| Ordinary filenames, directory traversal | SquashFS |
| Arbitrary path-based lookup | SquashFS |
| Sequential training-sample streaming | Tar shards |
| Explicit rank-level shard assignment | Tar shards |
| Frequent updates to individual records | Neither immutable layout |
| Full-dataset shuffle every epoch | SquashFS; a shuffle buffer is weaker |

`loader.shuffle` must be false for a streaming layout, and the configuration rejects
`true`. Order comes from shard order plus `loader.shuffle_buffer`, which shuffles within a
window rather than across the dataset. That is a genuinely weaker shuffle and part of what
you are choosing.

### The comparison tool

Two kinds of mismatch, not equally serious:

- **Blocking** — a different `manifest_hash` or schema version. The runs did not read the
  same data, so no table from them means anything. Exit code 3.
- **Uncontrolled** — same data, differing batch size, worker count, seed, or measurement
  length. Numbers still mean something individually, shown with
  `CONTROLLED_COMPARISON=false`.

Correctness is checked before performance: a group with failed, duplicate, or missing
samples is called out, because throughput inflated by redundant work is not throughput.

**Misreadings**

- *"Shards won, so use shards."* They won on this dataset, at this scale, with this access
  pattern. Check the requirement table: shuffle quality and path access are not throughput.
- *"SquashFS showed no gain, so packaging is pointless."* Collapsing 50 000 objects into
  one reduces metadata pressure on a filesystem others share — an operational argument the
  throughput number does not capture.
- *"The tool printed a table, so it is controlled."* Check `CONTROLLED_COMPARISON`.

---

## Part IV — workers

| Observation | Action |
| ----------- | ------ |
| Throughput rises, waits fall | More workers are useful |
| Throughput plateaus | Stop adding workers |
| Memory rises without throughput benefit | Reduce workers or prefetch depth |
| Throughput falls | The pipeline is oversubscribed |
| CPU saturated | Optimise decoding, or allocate more CPUs per rank |
| CPU idle but waits still high | Investigate storage or synchronisation, not workers |

The ladder is classified as `still-improving`, `plateau`, `regression`, or `flat`, and the
**cheapest** rung within 5 % of the best is recommended — not the fastest. Each worker is a
process holding memory.

A rung that exceeds the affinity mask is **never** recommended even when it measures
fastest, because it borrows capacity from everything else on the node.

### Cores versus threads

`--cpus-per-task=N` allocates N *physical cores*. Each carries two SMT threads, so the
affinity mask holds 2N *logical CPUs*. Reports distinguish them:

```text
ALLOCATED_PHYSICAL_CORES=7    LOGICAL_CPUS_IN_AFFINITY=14
```

and each rung reports `processes_per_physical_core`. A value of 2.0 means both SMT threads
of every core carry a runnable process: SMT-saturated. That is inside the allocation and
can genuinely help an I/O-bound loader — the reference measurement gained 53 % going from
1.0 to 2.0 — but the cores are shared, not doubled.

**Misreadings**

- *"13 workers fits 14 CPUs, so it is free."* It fills the mask while putting two processes
  on every core.
- *"CPU utilisation is 5 %, so the node is idle and I should add work."* The CPUs are idle
  *because* the pipeline is waiting on storage.
- *"More workers always helped, so more workers always help."* This dataset is small-file
  and I/O-latency-bound. A decode-heavy dataset saturates CPU and turns over much earlier —
  try `configs/datasets/decode_heavy.yaml` and watch `MAIN_LIMITING_FACTOR` change.
- *"Involuntary context switches rose, so scheduling contention is the problem."* On a
  shared partition this metric is dominated by other jobs. See
  [`reference-results.md`](reference-results.md).

---

## Part V — storage

| Observation | Decision |
| ----------- | -------- |
| Staging cost recovered quickly | Node-local staging may be appropriate |
| Staging wins only after many epochs | Use it only for sufficiently repeated access |
| One-pass workload never recovers the copy | Read from shared storage |
| Dataset is an unsafe fraction of memory | Do not stage; the job may die mid-copy |
| Flash gives no gain beyond noise | Stay on scratch |
| Flash materially reduces waits | Consider it for the active campaign |

The report gives **predicted wall time**, not cost. Flash is billed at a higher rate than
scratch and is a much smaller shared resource, so a nominally faster placement can still be
the wrong choice. Near-ties resolve to scratch, LUMI's documented default for job I/O.

**Misreadings**

- *"tmp was faster, so stage."* Read `BREAK_EVEN_EPOCHS` against the epochs you will
  actually run.
- *"Staging is safe, it used 1 % of memory."* True for that dataset. The check exists
  because the same code with a 200 GB dataset would kill the job.
- *"Break-even was 75 epochs, so staging is pointless."* For this dataset and layout. A
  metadata-heavy loose-file tree is where staging usually does win.

---

## Part VI — distributed

| Requirement | Field |
| ----------- | ----- |
| Every rank receives work | `IDLE_RANKS=[]` |
| No two ranks read the same sample | `DUPLICATE_SAMPLES=0` |
| No sample assigned to nobody | `MISSING_SAMPLES=0` |
| Coverage verifiable | `COVERAGE_FRACTION=1` with `measured_epochs` set |
| No rank holds the others up | `RANK_ELAPSED_SPREAD` small |

`MISSING_SAMPLES` is only evaluated when the ranks between them read enough to cover the
dataset once. A partial window says nothing about coverage, and the verdict's `notes` says
so rather than reporting the remainder as missing.

### How many shards?

For **this loader's** assignment strategy — ranks and DataLoader workers both partition at
shard granularity, round-robin — provide at least `world_size × num_workers` shards, and
preferably a multiple of it so readers receive equal counts. More shards generally improve
balancing.

This rule is implementation-specific. Other loaders partition samples *within* shards,
reuse shards, or schedule them dynamically, and would not have the same requirement.
Divisibility alone does not guarantee balance when shard sizes differ — that is what the
imbalanced-shards case demonstrates.

**Misreadings**

- *"Aggregate samples/s went up, so scaling works."* The duplicate case had the highest
  aggregate throughput measured and was the worst run. Check `DUPLICATE_SAMPLES` first.
- *"No duplicates and nothing missing, so the run is fine."* The imbalanced case has both
  and still wastes a third of the allocation. Read `RANK_ELAPSED_SPREAD`.
- *"Six idle ranks would obviously error."* Nothing errored. Coverage was 100 % and
  duplicates zero; the only signals were `IDLE_RANKS` and the elapsed spread.

---

## Part VII — readiness

| State | Meaning |
| ----- | ------- |
| `READY` | No blocking issue and nothing to caution about |
| `READY_WITH_CAUTION` | Usable, but a measured limitation remains |
| `NOT_READY` | A correctness problem must be fixed first |
| `INCONCLUSIVE` | Measurements were incomplete |

**Misreadings**

- *"READY_WITH_CAUTION means something is broken."* It means something is unverified or
  imprecise. Correctness problems give `NOT_READY`.
- *"INCONCLUSIVE means the experiments failed."* It means they were not all run.
- *"The tool chose the layout, so the choice is made."* It chose on measured throughput and
  correctness. It cannot see whether you need path-based random access or a full-dataset
  shuffle, and its limitations section says so.
