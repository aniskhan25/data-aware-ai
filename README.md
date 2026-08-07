# Data-Aware AI on LUMI

A hands-on tutorial for deciding how a dataset should be inspected, packaged, placed, and read before an AI workload scales.

> Is your dataset ready for efficient training?

This repository helps AI and HPC users answer that question with evidence. The goal is not to benchmark LUMI storage. The goal is to spend small jobs to avoid wasting large ones: inspect the data, compare alternatives under controlled conditions, and produce a data-readiness verdict you could defend in a review.

Use this repo when you want to:

- find out whether your dataset layout is the reason training is slow
- compare loose files, a SquashFS image, and tar shards on your own data
- choose a DataLoader worker count from measurement rather than habit
- decide whether node-local staging pays for itself
- check that many readers get unique, balanced data before scaling
- print enough evidence to justify the next allocation

> [!IMPORTANT]
> A dataset that can be read is not a dataset that can be read efficiently at scale. Every part changes one thing and measures it. A rung that shows no improvement is a result, and keeping the simpler configuration is a legitimate outcome.

## LUMI Storage Basics

| Term | What it is | Used here for |
|---|---|---|
| Project scratch | Large parallel (Lustre) space for active job I/O | The default for datasets and results |
| Project flash | Smaller, faster parallel space, billed at a higher rate | One optional comparison in Part V |
| Compute-node `/tmp` | Node-local space that lives in memory, charged against the job's allocation | The staging experiment |
| LUMI-O | S3-compatible object storage | Staging and secondary copies, optional |

Three facts shape every experiment:

- scratch is the documented location for job I/O
- `/tmp` is memory, so staging there spends the job's memory allocation
- many small files pressure filesystem metadata services, which affects everyone on the system

More, including the usable / suitable / scalable distinction the tutorial rests on: [`docs/storage-locations.md`](docs/storage-locations.md). Quotas and policy are the official documentation's business: <https://docs.lumi-supercomputer.eu/storage/>.

## Prerequisites

| You need | Notes |
|---|---|
| A LUMI project | With compute billing units. Find yours: `groups \| tr ' ' '\n' \| grep project_` |
| Project scratch | `/scratch/<project>/`. Work in a personal subdirectory, projects are shared |
| Basic Slurm and Python | Comfortable with `sbatch`, `squeue`, and reading a traceback |
| A PyTorch container | LUMI provides them under `/appl/local/containers/sif-images/` |
| ~1 GB of scratch, ~50 000 inodes | For the worked example. Check with `lumi-workspaces` |
| Optional: project flash | Only for one comparison in Part V. Skip it if you have none |

The whole tutorial submits roughly 30 short jobs, almost all on the CPU partition, each well under five minutes. No GPU is required at any point: this validates the input path, which never touches an accelerator.

## Setup

Run from the repository root on LUMI:

```bash
git clone https://github.com/aniskhan25/data-aware-ai.git
cd data-aware-ai

cp env.example.sh env.sh
$EDITOR env.sh                  # set LUMI_PROJECT; the other paths follow from it
source env.sh
```

> [!IMPORTANT]
> Run `source env.sh` in every new login shell. `sbatch` resolves your account at submission time. The job scripts source `env.sh` too, but that happens after the job is already queued, which is too late. Without it, submission fails with `AssocMaxSubmitJobLimit`.

Generate the worked-example dataset once:

```bash
sbatch jobs/prepare_dataset.sh configs/datasets/metadata_heavy.yaml
```

The worked example is deliberately metadata-heavy: 50 000 JPEG files of about 2.7 KB, which is where layout decisions bite hardest. `configs/datasets/balanced.yaml` (20 000 files at 224x224) is a good second run, and you should expect the conclusions to shift.

Slurm logs are written to `logs/`. Run artifacts are written to `$TUTORIAL_ROOT/outputs/`.

## The Method

Each part answers one question, changing one thing at a time.

| Part | Question | Output |
|---|---|---|
| **I** Inspect | What layout do I have? | Candidate experiments |
| **II** Baseline | What does it cost unmodified? | The reference every comparison uses |
| **III** Layouts | Does packaging or sharding help? | A layout choice |
| **IV** Workers | Can the CPU pipeline feed the workload? | A worker count and the limiting factor |
| **V** Storage | Scratch, flash, or node-local? | A placement, with staging cost included |
| **VI** Distributed | Do ranks read unique, balanced data? | A correctness verdict |
| **VII** Decision | Is the data path ready? | `READY` / `NOT_READY` / `INCONCLUSIVE` |

Every experiment writes a schema-validated JSON summary. The comparison tools read only summaries, and refuse to compare runs that read different data: a differing manifest hash stops the comparison rather than producing a tidy, meaningless table.

> [!NOTE]
> Performance figures in this README are reference measurements taken on LUMI project scratch with the `metadata-heavy` profile. Absolute values shift with system load and software versions. Compare the trends, not the numbers. Full tables and environment: [`docs/reference-results.md`](docs/reference-results.md).

## Part I: Inspect The Dataset

> What kind of data layout do I currently have?

```bash
sbatch jobs/inspect_dataset.sh
```

Reads metadata only, never opening a file, so it is cheap and needs no GPU. Run it inside the allocation you intend to train with, or the node-local staging advice has nothing to measure against.

Expected output shape:

```text
TOTAL_FILES=...            MEDIAN_FILE_BYTES=...
SMALL_FILE_FRACTION=...    FILESYSTEM_OBJECTS=...
P95_TO_MEDIAN_RATIO=...    MAX_FILES_IN_ONE_DIRECTORY=...
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
```

Each candidate arrives with the observation that motivated it. It proposes experiments; it does not choose a format, and it says so. It cannot see whether you need random access by path, whether order matters, or how expensive a record is to decode.

See [`docs/inspection-report.md`](docs/inspection-report.md) for the report schema.

## Part II: The Loose-File Baseline

> What does the unmodified dataset cost?

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

Each sample does a manifest lookup, an open, a read, a decode, and a small synthetic compute step. This run is the reference for everything that follows.

Expected output shape:

```text
LAYOUT=loose-files
SAMPLES_PER_SECOND=...
MEAN_DATA_WAIT_FRACTION=...
FAILED_SAMPLES=0
DUPLICATE_SAMPLES=0
MISSING_SAMPLES=0
```

The headline metric is useful sample throughput, not bandwidth. Bandwidth hides decode bottlenecks, worker stalls, and duplicated samples.

> [!WARNING]
> The three correctness counters must all be zero. A run that could not read its data is not a baseline, however fast it looks.

See [`docs/measurement-methodology.md`](docs/measurement-methodology.md) for what is timed, what warm-up excludes, and how duplicates are counted.

## Part III: Compare Dataset Layouts

> Does packaging or sharding help?

```bash
./jobs/run_stage.sh squashfs
./jobs/run_stage.sh webdataset
python3 scripts/compare_layouts.py "$TUTORIAL_ROOT"/outputs/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json
```

`run_stage.sh` chains a build and its measurement with a Slurm `afterok` dependency. Submitting them as two bare `sbatch` calls is a race: `sbatch` returns as soon as a job is queued, so the benchmark can start before the artifact exists.

All three layouts read the same manifest, and the sample bytes are verified identical across them. Without that, a throughput difference could just be a difference in what was read.

Reference result, three repeats each:

| Layout | Median samples/s | CV | FS objects | Startup |
|---|---:|---:|---:|---:|
| loose-files | 405 | 0.88 | 50 000 | 0.58 s |
| squashfs | 3 652 | 0.08 | 1 | 2.38 s |
| webdataset | 6 926 | 0.14 | 51 | 0.90 s |

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| loose files far behind packaged layouts | package or shard before scaling | the bytes are trivial, 50 000 separate opens are not |
| high CV on loose files | never trust a single loose-file measurement | one repeat hit 1582 against a median of 405, consistent with page-cache warming |
| low CV on squashfs | prefer it when reproducibility matters | one object instead of fifty thousand, an argument no throughput number shows |

> [!WARNING]
> SquashFS compression is easy to get wrong. Use both `-noD` and `-noF`. `-noD` alone only disables compression of full data blocks; files below the block size are stored as fragments, which needs `-noF`. In a small-file dataset nearly every file is a fragment, so `-noD` alone leaves the image fully compressed: measured, a 50 000-file tree still came out at 79 % of source size and took five minutes.

See [`docs/dataset-layouts.md`](docs/dataset-layouts.md) for the requirement table. Throughput is one dimension; path-based access, shuffle quality, and mutability are others.

## Part IV: Tune The Input Workers

> Can the CPU pipeline feed the workload?

```bash
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
python3 scripts/compare_workers.py "$TUTORIAL_ROOT"/outputs/workers/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json
```

Lengthen the measured window. At 14 000 samples/s the default 200 batches is under a second of measurement, far too short to be stable.

### `--cpus-per-task=7` allocates 7 cores and shows 14 logical CPUs

This distinction decides how to read the whole ladder. Seven physical cores each carry two SMT threads, so the affinity mask contains fourteen logical CPUs. Thirteen workers plus the parent process fill that mask while placing two runnable processes on every physical core. That is SMT-saturated, not fourteen independent CPUs.

Reference result:

| Workers | Samples/s | Proc/core | CPU util | Wait frac |
|---:|---:|---:|---:|---:|
| 0 | 2 167 | 0.14 | 6 % | 99 % |
| 6 | 9 515 | 1.00 | 3 % | 83 % |
| 13 | 14 570 | 2.00 | 5 % | 77 % |
| 28 | 15 120 | 4.14 | 5 % | 76 % |

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| large gain from one to two processes per core | adopt the SMT-saturated rung | saturating both threads helps a loader that is mostly blocked |
| CPU utilisation stays low, wait fraction stays high | stop tuning workers, go to Part V | the pipeline was never CPU-bound |
| a rung beyond the affinity mask looks marginally faster | never adopt it | it borrows capacity from everything else on the node |

Workers gave the largest single tuning win in the tutorial, but the diagnosis matters more than the number: with the CPUs idle and three quarters of the loop still waiting, the worker count is no longer the bottleneck.

## Part V: Compare Storage Placement

> Scratch, flash, or node-local `/tmp`?

```bash
sbatch jobs/run_storage_comparison.sh configs/staging/scratch.yaml
sbatch jobs/run_storage_comparison.sh configs/staging/tmp.yaml
./jobs/run_stage.sh flash            # optional, needs a flash allocation
python3 scripts/compare_storage.py "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json
```

Staging must be paid for before the first sample is read. The `/tmp` job determines the allocated memory, refuses to stage if the dataset exceeds a safety fraction of it, copies, validates the copy, measures, and removes it. `staging_seconds` and `validation_seconds` appear in the summary and in the break-even arithmetic. There is no configuration in which staging cost is excluded, and refusal is the default for anything doubtful, including an unknown allocation.

Reference result, two repeats each:

| Placement | Samples/s | Epoch s | Staging s | Break-even |
|---|---:|---:|---:|---:|
| scratch | 13 480 | 3.712 | 0 | - |
| flash | 14 070 | 3.556 | 0 | immediate |
| tmp | 13 720 | 3.648 | 4.755 | 75 epochs |

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| break-even far beyond the planned epochs | do not stage | a 4.76 s copy bought 0.19 s over three epochs |
| placements differ by less than the repeat spread | choose on setup cost, not speed | spread of 4.9 to 5.1 % exceeds every difference between placements |
| a placement is faster in steady state only | check the end-to-end number | `/tmp` was 1.8 % faster in steady state, and slower overall |

> [!WARNING]
> Faster steady-state reads do not mean a faster end-to-end job. The comparison reports predicted wall time, not cost; storage billing rates and scarcity are yours to weigh.

## Part VI: Validate Distributed Reading

> Do the ranks get unique, balanced data?

```bash
python3 scripts/shard_summary.py "$TUTORIAL_ROOT"/shards --readers 8   # free, before any job
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
```

This asks about correctness, not speed. Eight tasks with seven cores each reproduce the rank count and nominal per-rank CPU share of a full LUMI-G job. They do not reproduce LUMI-G's NUMA layout or CPU-GPU binding: this runs on a CPU partition and never touches an accelerator. For placement on the real accelerator node, use `standard-g` with eight GCDs and explicit binding.

Three deliberately broken cases ship alongside the healthy one:

| Case | Reads | Unique | Duplicates | Idle | Elapsed spread | Aggregate/s | Valid |
|---|---:|---:|---:|---|---:|---:|---|
| healthy | 50 000 | 50 000 | 0 | none | 11 % | 16 630 | yes |
| too few shards | 50 000 | 50 000 | 0 | 6 of 8 | 99 % | 6 751 | no |
| duplicate samples | 400 000 | 50 000 | 350 000 | none | 0.03 % | 21 030 | no |
| imbalanced shards | 50 000 | 50 000 | 0 | none | 33 % | 15 530 | yes |

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| highest aggregate throughput of the four | reject the run | eight ranks each read every shard, 87.5 % of the work wasted, 19.0 s against the healthy run's 3.2 s |
| zero duplicates and full coverage, but idle ranks | add shards | the data is read correctly, three quarters of the allocation just does nothing |
| `PARTITIONING_VALID=true` with a large elapsed spread | balance shards by estimated work | correct partitioning is necessary, not sufficient |

> [!WARNING]
> No throughput number can see the duplicate case. Its rank spread is a near-perfect 0.03 % because every rank is doing identically useless work. That is why the tutorial gathers which samples each rank read, not how many.

Runs use `measured_epochs: 1` rather than a batch count, so "every sample read once, by exactly one reader" is a checkable statement.

See [`docs/interpreting-results.md`](docs/interpreting-results.md) for the shard-count rule, which is specific to how this loader partitions.

## Part VII: The Data-Readiness Decision

> Is the data path ready?

```bash
python3 scripts/render_decision.py --planned-epochs 3 \
    --inspection  "$TUTORIAL_ROOT"/outputs/inspection/dataset_report.json \
    --layouts     "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json \
    --workers     "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json \
    --storage     "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json \
    --distributed "$TUTORIAL_ROOT"/outputs/distributed/healthy/distributed_verdict.json
```

`--planned-epochs` is not cosmetic. It decides the staging recommendation outright: the same measurements say "stage" for a long campaign and "do not stage" for a one-pass job, and nothing in the data can tell which you intend to run.

Expected output shape:

```text
DATA_READINESS=...
RECOMMENDED_LAYOUT=...          RECOMMENDED_STORAGE=...
RECOMMENDED_WORKERS_PER_RANK=...    NODE_LOCAL_STAGING=...
DISTRIBUTED_PARTITIONING=...    MAIN_LIMITING_FACTOR=...
NEXT_EXPERIMENT=...
```

`data_readiness.md` carries each recommendation with the measurement behind it, plus the limitations it always states. Two rules constrain what it may conclude:

- **No single throughput number decides readiness.** A pipeline that reads the wrong data quickly is `NOT_READY`; one that reads the right data slowly can be `READY`. Duplicates, missing samples, idle ranks, and failed samples all block.
- **Absent evidence is not good news.** A missing required input gives `INCONCLUSIVE`, never a cheerful default.

Exit codes: 0 ready, 5 not ready, 6 inconclusive.

With a ready verdict, continue to [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai), which asks whether more GCDs produce useful throughput. If scaling disappoints there, this report is your evidence that the data path was not the cause.

## Use It On Your Own Dataset

The synthetic dataset exists so everyone measures the same bytes. Your data is the point.

If your samples are already files, you need no code. Write a JSON Lines manifest and change two lines of configuration:

```yaml
dataset:
  layout: loose-files
  root: /scratch/project_XXXXXXXXX/me/my-dataset
  manifest: /scratch/project_XXXXXXXXX/me/my-dataset/manifest.jsonl
```

If your samples live inside something else, whether Parquet, HDF5, a database, or your own format, implement a `DatasetAdapter` with one method: given a manifest position, return that sample's bytes. The decode, batching, timing, accounting, and summary stay shared, which is what makes your format comparable with the tutorial's layouts.

Every part then applies unchanged.

See [`docs/adapting-your-dataset.md`](docs/adapting-your-dataset.md) for manifest fields, a manifest-building snippet, the adapter interface, and what to change per part.

## Optional Format Tracks

Not required. The same measurement applied to three more ecosystems, plus the LUMI-O lifecycle.

```bash
python3 scripts/convert_dataset.py --to parquet --source ... --manifest ... --output ...
sbatch jobs/run_loader.sh configs/formats/parquet.yaml
```

| Access pattern | Parquet | Hugging Face (Arrow) |
|---|---:|---:|
| Random (`shuffle: true`) | 983 samples/s | 11 290 |
| Sequential (`shuffle: false`) | 20 070 samples/s | 17 090 |

Parquet is both the fastest and the slowest representation in this tutorial, a 20x swing from one flag. In this adapter, a random sample may require loading and decompressing a whole row group; with a single-row-group cache, shuffled access repeatedly evicts and reloads groups. The magnitude is partly a property of the adapter, but the direction is structural.

Parquet and `datasets` are present in LUMI's PyTorch containers; h5py is not.

See [`docs/optional-tracks.md`](docs/optional-tracks.md) and [`docs/object-storage.md`](docs/object-storage.md).

## Repository Layout

```text
configs/     Experiment definitions: baseline, workers, staging, distributed, formats
dataaware/   Shared library: config, manifest, schema, layouts, metrics, decision
examples/    Optional track converters and adapters
jobs/        Slurm launchers
logs/        Slurm output directory
scripts/     Command line entry points
docs/        Reference material
```

## Troubleshooting

The three that catch almost everyone:

| Symptom | Cause |
|---|---|
| `AssocMaxSubmitJobLimit` on submit | You did not `source env.sh` in this shell |
| `undefined environment variable(s) TUTORIAL_ROOT` | Same |
| A job "succeeds" but produces nothing | Submit from the repository root |

See [`docs/troubleshooting.md`](docs/troubleshooting.md) for the rest.

## Status

Parts I to VII and the optional tracks are implemented and runnable end to end. Every reported performance measurement was taken on LUMI. Three things remain unverified there, and are marked rather than assumed:

- **HDF5** is implemented and unit-tested, but LUMI's PyTorch containers lack h5py.
- **LUMI-O** needs an interactive `lumio-conf`, so the round-trip script has not been run against a real bucket.
- The staging **refusal path** has only been exercised in unit tests; the worked example is 1 % of the allocation, well inside the safety margin.

Repeat counts are modest, two or three per configuration. That is enough to show variability, not to characterise it tightly. For a decision you intend to defend, run five or more and report the median and range.

## References

- LUMI storage: <https://docs.lumi-supercomputer.eu/storage/> ·
  [Lustre](https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/) ·
  [FUSE/SquashFS](https://docs.lumi-supercomputer.eu/storage/formats/FUSE/) ·
  [LUMI-O](https://docs.lumi-supercomputer.eu/storage/lumio/)
- [Moving data to and from LUMI](https://docs.lumi-supercomputer.eu/firststeps/movingdata/)
- [LUMI AI Guide, data lesson](https://github.com/aniskhan25/LUMI-AI-Guide/tree/main/2-data)
- [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai)

MIT licensed. See [LICENSE](LICENSE).
