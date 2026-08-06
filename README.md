# Data-Aware AI on LUMI

**Is your dataset ready for efficient training?**

Answer it with evidence, before you spend a large allocation finding out. This tutorial
runs a series of small jobs that inspect your dataset, compare alternatives under
controlled conditions, and produce a data-readiness verdict you could defend in a review.

> A dataset that can be read successfully is not necessarily a dataset that can be read
> efficiently at scale.

It is not a benchmark of LUMI storage. It is a method for deciding what to do with *your*
data - and it works on your data, not only on the worked example. See
[Use it on your own dataset](#use-it-on-your-own-dataset).

---

## Prerequisites

| You need | Notes |
| -------- | ----- |
| A LUMI project | With compute billing units. Find yours: `groups \| tr ' ' '\n' \| grep project_` |
| Project scratch | `/scratch/<project>/`. Work in a personal subdirectory - projects are shared |
| Basic Slurm and Python | You should be comfortable with `sbatch`, `squeue`, and reading a traceback |
| A PyTorch container | LUMI provides them under `/appl/local/containers/sif-images/` |
| ~1 GB of scratch, ~50 000 inodes | For the worked example. Check your quota with `lumi-workspaces` |
| Optional: project flash | Only for one comparison in Part V. Skip it if you have none |

The whole tutorial submits roughly **30 short jobs**, almost all on the CPU partition,
each well under five minutes. No GPU is required at any point: this validates the *input
path*, which never touches an accelerator.

---

## Quick start

```bash
git clone https://github.com/aniskhan25/data-aware-ai.git
cd data-aware-ai

cp env.example.sh env.sh
$EDITOR env.sh                  # set LUMI_PROJECT; the other paths follow from it
source env.sh                   # REQUIRED before every sbatch - see below

printf 'Account: %s\nRoot:    %s\n' "$SBATCH_ACCOUNT" "$TUTORIAL_ROOT"
```

> **`source env.sh` in every new login shell.** `sbatch` resolves your account at
> *submission* time. The job scripts source `env.sh` too, but that happens after the job
> is already queued - too late. Without it, submission fails with
> `AssocMaxSubmitJobLimit`, because Slurm falls back to an association that permits no
> jobs.

Then the whole path, in order:

```bash
# 1. Generate the worked-example dataset (once, ~2 min)
sbatch jobs/prepare_dataset.sh configs/datasets/metadata_heavy.yaml

# 2. Part I - characterise it before allocating anything expensive
sbatch jobs/inspect_dataset.sh

# 3. Part II - the loose-file baseline everything is measured against
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml

# 4. Part III - package it two ways, and measure each after it is built
./jobs/run_stage.sh squashfs
./jobs/run_stage.sh webdataset
python3 scripts/compare_layouts.py "$TUTORIAL_ROOT"/outputs/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json

# 5. Part IV - how many DataLoader workers?
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
python3 scripts/compare_workers.py "$TUTORIAL_ROOT"/outputs/workers/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json

# 6. Part V - scratch, flash, or node-local /tmp?
sbatch jobs/run_storage_comparison.sh configs/staging/scratch.yaml
sbatch jobs/run_storage_comparison.sh configs/staging/tmp.yaml
python3 scripts/compare_storage.py "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json

# 7. Part VI - do 8 readers get unique, balanced data?
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml

# 8. Part VII - the verdict
python3 scripts/render_decision.py --planned-epochs 3 \
    --inspection  "$TUTORIAL_ROOT"/outputs/inspection/dataset_report.json \
    --layouts     "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json \
    --workers     "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json \
    --storage     "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json \
    --distributed "$TUTORIAL_ROOT"/outputs/distributed/healthy/distributed_verdict.json
```

`./jobs/run_stage.sh` chains a build and its measurement with a Slurm `afterok`
dependency. Submitting them as two bare `sbatch` calls is a race: `sbatch` returns as soon
as a job is queued, so the benchmark can start before the artifact exists.

**Non-LUMI users:** the whole pipeline runs on a laptop against a tiny dataset, to learn
the tooling. The numbers it produces are not meaningful performance results.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e '.[loader,dev]'
export DAAI_TEST_ROOT=/tmp/daai-test
python scripts/generate_dataset.py --profile tiny --output "$DAAI_TEST_ROOT"
python scripts/benchmark_loader.py --config configs/test/tiny.yaml
```

---

## About the numbers in this README

> Every performance figure below is a **reference measurement** taken on LUMI project
> scratch with the `metadata-heavy` profile: 50 000 JPEG files of about 2.7 KB. Your
> absolute values will differ with system load, software versions, and the state of your
> project storage. **Compare the qualitative trends, not the exact throughputs.**

Full environment, raw tables, and measurement limitations:
[`docs/reference-results.md`](docs/reference-results.md).

The worked example deliberately uses a *metadata-heavy* dataset, because that is where
layout decisions bite hardest. `configs/datasets/balanced.yaml` (20 000 files at
224×224) is a good second run: decoding costs more, and you should expect the
conclusions to shift.

---

## The method

Each part answers one question, changing one thing at a time.

| Part | Question | Output |
| ---- | -------- | ------ |
| **I** Inspect | What layout do I have? | Candidate experiments |
| **II** Baseline | What does it cost unmodified? | The reference every comparison uses |
| **III** Layouts | Does packaging or sharding help? | A layout choice |
| **IV** Workers | Can the CPU pipeline feed the workload? | A worker count and the limiting factor |
| **V** Storage | Scratch, flash, or node-local? | A placement, with staging cost included |
| **VI** Distributed | Do ranks read unique, balanced data? | A correctness verdict |
| **VII** Decision | Is the data path ready? | `READY` / `NOT_READY` / `INCONCLUSIVE` |

You are not expected to adopt every rung. A rung that shows no improvement is a result,
and staying with the simpler configuration is a legitimate outcome.

Every experiment writes a schema-validated JSON summary. The comparison tools read only
summaries, and **refuse to compare runs that read different data** - a differing manifest
hash stops the comparison rather than producing a tidy, meaningless table.

---

## LUMI storage in one table

| Term | What it is | Used here for |
| ---- | ---------- | ------------- |
| Project scratch | Large parallel (Lustre) space for active job I/O | The default for datasets and results |
| Project flash | Smaller, faster parallel space; **billed at a higher rate** | One optional comparison in Part V |
| Compute-node `/tmp` | Node-local space that **lives in memory** and is charged against the job's allocation | The staging experiment |
| LUMI-O | S3-compatible object storage | Staging and secondary copies (optional) |

Three facts shape every experiment here: scratch is the documented location for job I/O;
`/tmp` is memory, so staging there spends the job's memory allocation; and large numbers
of small files pressure filesystem metadata services, which affects everyone on the
system, not just you.

More, including the *usable / suitable / scalable* distinction the tutorial rests on:
[`docs/storage-locations.md`](docs/storage-locations.md). Quotas and policy are the
official documentation's business: <https://docs.lumi-supercomputer.eu/storage/>.

---

## Part I - Inspect the dataset

**What kind of data layout do I currently have?**

```bash
sbatch jobs/inspect_dataset.sh          # or: python3 scripts/inspect_dataset.py --path ... --verbose
```

Reads metadata only - it never opens a file - so it is cheap and needs no GPU. Run it
inside the allocation you intend to train with, or the node-local staging advice has
nothing to measure against.

Reference result for the worked example:

```text
TOTAL_FILES=50002          MEDIAN_FILE_BYTES=2673
SMALL_FILE_FRACTION=1      FILESYSTEM_OBJECTS=50104
P95_TO_MEDIAN_RATIO=1.013  MAX_FILES_IN_ONE_DIRECTORY=500
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
```

Each candidate arrives with the observation that motivated it. It proposes experiments;
it does not choose a format, and it says so - it cannot see whether you need random
access by path, whether order matters, or how expensive a record is to decode.

→ [`docs/inspection-report.md`](docs/inspection-report.md) for the report schema.

---

## Part II - The loose-file baseline

**What does the unmodified dataset cost?**

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

Each sample does a manifest lookup, an open, a read, a decode, and a small synthetic
compute step. This run is the reference for everything that follows.

The headline metric is **useful sample throughput**, not bandwidth. Bandwidth hides
decode bottlenecks, worker stalls, and duplicated samples. Correctness counters
(`FAILED_SAMPLES`, `DUPLICATE_SAMPLES`, `MISSING_SAMPLES`) must all be zero - a run that
could not read its data is not a baseline.

→ [`docs/measurement-methodology.md`](docs/measurement-methodology.md) for what is timed,
what warm-up excludes, and how duplicates are counted.

---

## Part III - Compare dataset layouts

**Does packaging or sharding help?**

```bash
./jobs/run_stage.sh squashfs        # build, then measure, chained with afterok
./jobs/run_stage.sh webdataset
python3 scripts/compare_layouts.py "$TUTORIAL_ROOT"/outputs/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json
```

All three layouts read the same manifest, and the sample bytes are verified identical
across them. Without that, a throughput difference could just be a difference in what was
read.

Reference result - three repeats each:

| Layout | Median | Min | Max | CV | FS objects | Startup |
| ------ | ------ | --- | --- | - | ---------- | ------- |
| loose-files | 405 | 363 | **1582** | **0.88** | 50 000 | 0.58 s |
| squashfs | 3 652 | 3 430 | 4 048 | 0.08 | 1 | 2.38 s |
| webdataset | 6 926 | 5 572 | 7 403 | 0.14 | 51 | 0.90 s |

Three lessons, and the third is the one people miss:

**Loose files are unusable here** - 17× slower than shards, with 99.6 % of the loop spent
waiting. The bytes are trivial; 50 000 separate opens are not.

**Loose files are also unpredictable.** One repeat hit 1582 against a median of 405 - a
CV of 0.88, consistent with page-cache warming from a previous job on the same node. *Any
single* loose-file measurement on a shared filesystem is close to meaningless. This is why
the tutorial insists on repeats.

**Packaging buys stability, not just speed.** SquashFS is the most reproducible layout
(CV 0.08) because it reads one object instead of fifty thousand. That argument never
appears in a throughput number.

Compression matters and is easy to get wrong. The default is **`-noD -noF`** - *both*
flags. `-noD` alone only disables compression of full data blocks; files below the block
size are stored as *fragments*, which needs `-noF`. In a small-file dataset nearly every
file is a fragment, so `-noD` alone leaves the image fully compressed. Measured: a
50 000-file tree packed with `-noD` alone still came out at 79 % of source size and took
five minutes, because every file went through the compressor.

→ [`docs/dataset-layouts.md`](docs/dataset-layouts.md) for the requirement table -
throughput is one dimension, and path-based access, shuffle quality, and mutability are
others.

---

## Part IV - Tune the input workers

**Can the CPU pipeline feed the workload?**

```bash
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
python3 scripts/compare_workers.py "$TUTORIAL_ROOT"/outputs/workers/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json
```

Lengthen the window: at 14 000 samples/s the default 200 batches is under a second of
measurement, far too short to be stable.

### `--cpus-per-task=7` allocates 7 cores, and shows 14 logical CPUs

This distinction decides how to read the whole ladder. Seven **physical cores** each carry
two SMT threads, so the affinity mask contains fourteen **logical CPUs**. Thirteen workers
plus the parent process fill that mask - while placing *two runnable processes on every
physical core*. That is SMT-saturated, not fourteen independent CPUs.

Reference result:

| Workers | Samples/s | Proc/core | CPU util | Wait frac |
| ------- | --------- | --------- | -------- | --------- |
| 0 | 2 167 | 0.14 | 6 % | 99 % |
| 2 | 4 048 | 0.43 | 2 % | 93 % |
| 6 | 9 515 | **1.00** | 3 % | 83 % |
| 7 | 9 733 | 1.14 | 3 % | 83 % |
| **13** | **14 570** | **2.00** | 5 % | 77 % |
| 28 | 15 120 | 4.14 | 5 % | 76 % |

`RECOMMENDED_WORKERS=13`, `MAIN_LIMITING_FACTOR=storage-or-synchronisation`.

Workers gave a **6.7× gain**, the largest single tuning win in the tutorial. Note where it
came from: going from one process per core (6) to two (13) added **53 %**. Saturating both
SMT threads genuinely helps a loader that is mostly *blocked*, which is why the tool
recommends 13 while naming it SMT-saturated rather than pretending the cores are free.

Beyond the affinity mask, 28 workers added 3.8 % and is never recommended - a rung that
exceeds its allocation borrows capacity from everything else on the node.

**The diagnosis matters more than the number.** CPU utilisation never exceeded 6 % while
the wait fraction only fell from 99 % to 77 %. This pipeline was never CPU-bound; extra
workers helped by keeping more reads in flight. With the CPUs idle and three quarters of
the loop still waiting, the worker count is no longer the bottleneck - which is exactly
what Part V exists to test.

---

## Part V - Compare storage placement

**Scratch, flash, or node-local `/tmp`?**

```bash
sbatch jobs/run_storage_comparison.sh configs/staging/scratch.yaml
sbatch jobs/run_storage_comparison.sh configs/staging/tmp.yaml
./jobs/run_stage.sh flash            # optional; needs a flash allocation
python3 scripts/compare_storage.py "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json
```

> Faster steady-state reads do not automatically mean a faster end-to-end job.

Staging must be paid for before the first sample is read. The `/tmp` job determines the
allocated memory, **refuses to stage** if the dataset exceeds a safety fraction of it,
copies, validates the copy, measures, and removes it - in a `finally` block plus a shell
trap for Slurm's `SIGTERM`. `staging_seconds` and `validation_seconds` appear in the
summary and in the break-even arithmetic. There is no configuration in which staging cost
is excluded.

Refusal is the default for anything doubtful, **including an unknown allocation**.

Reference result - two repeats each:

| Placement | Samples/s | Epoch s | Staging s | Break-even |
| --------- | --------- | ------- | --------- | ---------- |
| scratch | 13 480 | 3.712 | 0 | - |
| flash | 14 070 | 3.556 | 0 | immediate |
| tmp | 13 720 | 3.648 | **4.755** | **75 epochs** |

Node-local staging saved 0.063 s per epoch for a 4.76 s copy. A three-epoch workload pays
4.7 s for 0.19 s of benefit. It was *technically* 1.8 % faster in steady state - precisely
the number that would have justified it had the copy cost been left out.

**And all three are indistinguishable.** Run-to-run spread (4.9-5.1 %) exceeds every
difference between placements (1.8-4.4 %). The report says so rather than crowning a
winner on a margin thinner than its own noise. So the honest conclusion is not "flash is
fastest" - it is *"speed does not separate these; decide on setup cost and operational
fit"*, and scratch has no setup cost, is not scarce, and is billed at a lower rate than
flash.

The comparison reports **predicted wall time**, not cost. Storage billing rates and
scarcity are yours to weigh.

---

## Part VI - Validate distributed reading

**Do the ranks get unique, balanced data?**

```bash
python3 scripts/shard_summary.py "$TUTORIAL_ROOT"/shards --readers 8   # free, before any job
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
```

This asks about correctness, not speed. Eight tasks with seven cores each reproduce the
rank count and nominal per-rank CPU share of a full LUMI-G job. They do **not** reproduce
LUMI-G's NUMA layout or CPU-GPU binding: this runs on a CPU partition and never touches an
accelerator. For placement on the real accelerator node, use `standard-g` with eight GCDs
and explicit binding.

Three deliberately broken cases ship alongside the healthy one:

| Case | Reads | Unique | Duplicates | Idle | Elapsed spread | Aggregate/s | Valid |
| ---- | ----- | ------ | ---------- | ---- | -------------- | ----------- | ----- |
| healthy | 50 000 | 50 000 | 0 | none | 11 % | 16 630 | **yes** |
| too few shards | 50 000 | 50 000 | 0 | **6 of 8** | **99 %** | 6 751 | no |
| duplicate samples | **400 000** | 50 000 | **350 000** | none | 0.03 % | **21 030** | no |
| imbalanced shards | 50 000 | 50 000 | 0 | none | **33 %** | 15 530 | **yes** |

**Read the duplicate row carefully.** Its aggregate throughput is the highest of the four
- 26 % above the healthy run - and it is the worst result on the table. Eight ranks each
read every shard: 400 000 reads to cover 50 000 samples, 87.5 % of the work wasted, 19.0 s
of wall time against the healthy run's 3.2 s for the same epoch. Its rank spread is a
near-perfect 0.03 % because every rank is doing identically useless work.

**No throughput number can see this.** That is why the tutorial gathers *which* samples
each rank read, not how many.

Two rows also invert the usual intuition: `too few shards` has **zero** duplicates and
100 % coverage - the data is read perfectly correctly, three quarters of the allocation
just does nothing. And `imbalanced shards` reports `PARTITIONING_VALID=true` while wasting
a third of the allocation, because correct partitioning is necessary but not sufficient.

`measured_epochs: 1` rather than a batch count, so "every sample read once, by exactly one
reader" is a checkable statement.

→ [`docs/interpreting-results.md`](docs/interpreting-results.md) for the shard-count rule,
which is specific to how this loader partitions.

---

## Part VII - The data-readiness decision

**Is the data path ready?**

```bash
python3 scripts/render_decision.py --planned-epochs 3 --inspection ... --layouts ...
```

`--planned-epochs` is not cosmetic. It decides the staging recommendation outright: the
same measurements say "stage" for a long campaign and "do not stage" for a one-pass job,
and nothing in the data can tell which you intend to run.

```text
DATA_READINESS=READY_WITH_CAUTION
RECOMMENDED_LAYOUT=webdataset          RECOMMENDED_STORAGE=scratch
RECOMMENDED_WORKERS_PER_RANK=13        NODE_LOCAL_STAGING=not-recommended
DISTRIBUTED_PARTITIONING=valid         MAIN_LIMITING_FACTOR=storage-or-synchronisation
NEXT_EXPERIMENT=scaling-aware-ai-one-gcd-baseline
```

`data_readiness.md` carries each recommendation with the measurement behind it, plus the
limitations it always states.

Two rules constrain what this may conclude:

**No single throughput number decides readiness.** A pipeline that reads the wrong data
quickly is `NOT_READY`; one that reads the right data slowly can be `READY`. Duplicates,
missing samples, idle ranks, and failed samples all block.

**Absent evidence is not good news.** A missing required input gives `INCONCLUSIVE`, never
a cheerful default. Exit codes: 0 ready, 5 not ready, 6 inconclusive.

With a ready verdict, continue to
[Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai), which asks
whether more GCDs produce useful throughput. If scaling disappoints there, this report is
your evidence that the data path was not the cause.

---

## Use it on your own dataset

The synthetic dataset exists so everyone measures the same bytes. **Your data is the
point.**

If your samples are already files, you need no code - write a JSON Lines manifest and
change two lines of configuration:

```yaml
dataset:
  layout: loose-files
  root: /scratch/project_XXXXXXXXX/me/my-dataset
  manifest: /scratch/project_XXXXXXXXX/me/my-dataset/manifest.jsonl
```

If your samples live inside something else - Parquet, HDF5, a database, your own format -
implement a `DatasetAdapter` with one method: *given a manifest position, return that
sample's bytes*. The decode, batching, timing, accounting, and summary stay shared, which
is what makes your format **comparable** with the tutorial's layouts.

Every part then applies unchanged.

→ [`docs/adapting-your-dataset.md`](docs/adapting-your-dataset.md) - manifest fields, a
manifest-building snippet, the adapter interface, and what to change per part.

---

## Optional format tracks

Not required. The same measurement applied to three more ecosystems, plus the LUMI-O
lifecycle.

```bash
python3 scripts/convert_dataset.py --to parquet --source ... --manifest ... --output ...
sbatch jobs/run_loader.sh configs/formats/parquet.yaml
```

| Access pattern | Parquet | Hugging Face (Arrow) |
| -------------- | ------- | -------------------- |
| Random (`shuffle: true`) | **983** samples/s | 11 290 |
| Sequential (`shuffle: false`) | **20 070** samples/s | 17 090 |

Parquet is both the fastest and the slowest representation in this tutorial - a 20× swing
from one flag. In this adapter, a random sample may require loading and decompressing a
whole row group; with a single-row-group cache, shuffled access repeatedly evicts and
reloads groups. The magnitude is partly a property of the adapter, but the direction is
structural, and it is the *usable / suitable* distinction in one table.

Parquet and `datasets` are present in LUMI's PyTorch containers; **h5py is not**.

→ [`docs/optional-tracks.md`](docs/optional-tracks.md) and
[`docs/object-storage.md`](docs/object-storage.md).

---

## Repository layout

```text
configs/     Experiment definitions: baseline, workers, staging, distributed, formats
dataaware/   Shared library: config, manifest, schema, layouts, metrics, decision
examples/    Optional track converters and adapters
jobs/        Slurm launchers
scripts/     Command line entry points
docs/        Reference material
```

---

## Troubleshooting

The three that catch almost everyone:

| Symptom | Cause |
| ------- | ----- |
| `AssocMaxSubmitJobLimit` on submit | You did not `source env.sh` in this shell |
| `undefined environment variable(s) TUTORIAL_ROOT` | Same |
| A job "succeeds" but produces nothing | Submit from the repository root |

→ [`docs/troubleshooting.md`](docs/troubleshooting.md) for the rest.

---

## Status

Parts I-VII and the optional tracks are implemented and runnable end to end. Every
reported performance measurement was taken on LUMI.

Three things remain unverified on LUMI, and are marked rather than assumed:

- **HDF5** is implemented and unit-tested, but LUMI's PyTorch containers lack h5py.
- **LUMI-O** needs an interactive `lumio-conf`, so the round-trip script has not been run
  against a real bucket.
- The staging **refusal path** has only been exercised in unit tests; the worked example
  is 1 % of the allocation, well inside the safety margin.

Repeat counts are modest - two or three per configuration. That is enough to show
variability, not to characterise it tightly. For a decision you intend to defend, run five
or more and report the median and range.

---

## References

- LUMI storage: <https://docs.lumi-supercomputer.eu/storage/> ·
  [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/) ·
  [FUSE/SquashFS](https://docs.lumi-supercomputer.eu/storage/formats/FUSE/) ·
  [LUMI-O](https://docs.lumi-supercomputer.eu/storage/lumio/)
- [Moving data to and from LUMI](https://docs.lumi-supercomputer.eu/firststeps/movingdata/)
- [LUMI AI Guide, data lesson](https://github.com/aniskhan25/LUMI-AI-Guide/tree/main/2-data)
- [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai)

MIT licensed. See [LICENSE](LICENSE).
