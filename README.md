# Data-Aware AI on LUMI

How datasets should be inspected, packaged, placed, and read on LUMI, and what it costs when they are not.

> A dataset that can be read is not a dataset that can be read efficiently at scale.

This tutorial assumes completion of the LUMI onboarding material and familiarity with Slurm, project storage, and official LUMI containers.

Each step asks one question, runs it against a synthetic dataset, and reports the measurement. Several steps break something on purpose, because the failures are the part worth recognising: duplicate reads that show up as record throughput, ranks that sit idle while one does the work, a staging copy that never repays, a layout that collapses past a worker count that suits a different layout.

The dataset is generated deterministically, so every number below is reproducible. Measurements come from LUMI project scratch with the `metadata-heavy` profile, 50 000 JPEG files of about 2.7 KB, reported as `min / median / max` samples per second.

## Setup

```bash
git clone https://github.com/aniskhan25/data-aware-ai.git
cd data-aware-ai

cp env.example.sh env.sh
vim env.sh                      # set LUMI_PROJECT; the other paths follow from it
source env.sh                   # required in every login shell, before any sbatch

sbatch jobs/prepare_dataset.sh configs/datasets/metadata_heavy.yaml
```

The dataset is synthetic and generated deterministically from `(seed, index)`, so every reader measures the same bytes and can reproduce the numbers below. `configs/datasets/balanced.yaml` (20 000 files at 224x224) is a useful second run: decoding costs more, and the conclusions shift.

## 1. Inspect The Dataset

> What layout do I have?

```bash
sbatch jobs/inspect_dataset.sh
```

It walks the tree reading file metadata only, never opening a file, so it finishes in seconds however large the dataset.

```text
TOTAL_FILES=50002               TOTAL_GIB=0.134
MEDIAN_FILE_BYTES=2673          P95_TO_MEDIAN_RATIO=1.013
SMALL_FILE_FRACTION=1           FILESYSTEM_OBJECTS=50104
MAX_FILES_IN_ONE_DIRECTORY=500  DATASET_FRACTION_OF_MEMORY=0.004187
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
```

50 002 files holding 137 MiB, every one under the 64 KB small-file threshold and a median of 2.7 KB. Sizes are uniform, the 95th percentile landing within 1.3 % of the median. The tree presents 50 104 objects to the filesystem, and at 0.4 % of the job's memory allocation it would fit in node-local `/tmp`.

## 2. Establish The Baseline

> What does the unmodified dataset cost?

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

Three repeats, one file per sample:

| Layout | min / median / max | FS objects | Data wait |
|---|---:|---:|---:|
| loose-files | 362 / **405** / 1582 | 50 002 | 100 % |

405 samples per second from a tree of 50 002 files, with the loop waiting on data essentially the whole time. The 1582 outlier is consistent with page-cache warming by an earlier job on the same node. `FAILED_SAMPLES`, `DUPLICATE_SAMPLES`, and `MISSING_SAMPLES` are zero, and every later comparison is against this run.

## 3. Compare Dataset Layouts

> Does packaging or sharding help?

```bash
./jobs/run_stage.sh squashfs        # build, then measure, chained with afterok
./jobs/run_stage.sh webdataset

squeue -u $USER                     # wait until empty before comparing

python3 scripts/compare_layouts.py "$TUTORIAL_ROOT"/outputs/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json
```

Build the image with both `-noD` and `-noF`. `-noD` alone disables compression only for full data blocks, and files below the block size are stored as fragments. Nearly every file here is a fragment, so `-noD` alone still compresses the whole image: measured, 79 % of source size and five minutes to pack, against 1.005x and seconds with both flags.

Three repeats each:

| Layout | min / median / max | vs baseline | FS objects | Startup |
|---|---:|---:|---:|---:|
| loose-files | 362 / **405** / 1582 | - | 50 002 | 1.37 s |
| squashfs | 3430 / **3652** / 4048 | **9.0x** | 1 | 2.38 s |
| webdataset | 5571 / **6926** / 7403 | **17.1x** | 51 | 0.64 s |

SquashFS reached 9.0x the baseline and tar shards 17.1x, from identical bytes: all three read the same manifest, which `compare_layouts.py` verifies before it will compare them. Packaging bought stability as well as speed, the loose-file range spanning a factor of 4.4 against SquashFS's 1.2.

## 4. Tune The Input Workers

> Can the CPU pipeline feed the workload?

```bash
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000

squeue -u $USER                     # wait until empty before comparing

python3 scripts/compare_workers.py "$TUTORIAL_ROOT"/outputs/workers/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json
```

1000 batches, not the default 200: at 14 000 samples/s, 200 batches is under a second of measurement.

`--cpus-per-task=7` allocates 7 physical cores, each carrying two SMT threads, so the affinity mask holds 14 logical CPUs and 13 workers plus the parent process fill it at two per core.

Two repeats each, on the webdataset layout:

| Workers | min / median / max | vs 0 workers | Proc/core | CPU util | Data wait |
|---:|---:|---:|---:|---:|---:|
| 0 | 1972 / **2167** / 2362 | - | 0.14 | 6 % | 99 % |
| 2 | 3550 / **4048** / 4546 | 1.9x | 0.43 | 2 % | 93 % |
| 6 | 9471 / **9515** / 9559 | 4.4x | 1.00 | 3 % | 83 % |
| 7 | 9217 / **9733** / 10249 | 4.5x | 1.14 | 3 % | 83 % |
| 13 | 14453 / **14566** / 14679 | **6.7x** | 2.00 | 5 % | 77 % |
| 28 | 14221 / **15123** / 16026 | 7.0x | 4.14 | 5 % | 76 % |

`RECOMMENDED_WORKERS=13`, `MAIN_LIMITING_FACTOR=storage-or-synchronisation`, for tar shards specifically.

13 workers reached 14 566, a 6.7x gain over none, and going from one process per physical core to two accounted for 53 % of it: saturating both SMT threads helps a loader that is mostly blocked. 28 workers exceeds the affinity mask and its range overlaps 13's, so it is not measurably faster while borrowing capacity from the rest of the node. CPU utilisation stayed under 6 % throughout and the data wait only fell from 99 % to 77 %, so the pipeline was never CPU-bound; the extra workers helped by keeping more reads in flight.

### Every format at 13 workers

Step 3 ranked layouts at 4 workers. Re-measuring all six representations at the tuned worker count, three repeats each, same manifest and container, changes the answer. Formats that carry an index can be read shuffled or sequentially; tar shards only stream.

Shuffled (`shuffle: true`):

| Format | min / median / max | vs loose files |
|---|---:|---:|
| HDF5 | 11 356 / **11 723** / 13 228 | **4.5x** |
| Arrow | 8 132 / **8 240** / 12 750 | 3.2x |
| loose files | 2 516 / **2 609** / 2 618 | - |
| SquashFS | 2 260 / **2 303** / 2 640 | 0.9x |
| Parquet | 724 / **902** / 1 345 | 0.3x |

Sequential (`shuffle: false`):

| Format | min / median / max | vs loose files |
|---|---:|---:|
| Parquet | 22 076 / **22 469** / 22 914 | **8.6x** |
| tar shards | 14 160 / **15 736** / 16 482 | 6.0x |
| Arrow | 13 827 / **15 020** / 18 319 | 5.8x |
| HDF5 | 10 916 / **11 112** / 13 097 | 4.3x |

No format wins both columns. Parquet is last shuffled and first sequential, a 25-fold swing from one flag, because reaching one sample means reading a whole row group of 1250 and a single-group cache thrashes under shuffling. HDF5's two ranges overlap almost entirely: its per-handle chunk cache makes access order not matter, which is the property to want if you shuffle every epoch.

All 27 runs reported zero failed, duplicate, and missing samples. These settings differ from step 3, so `compare_layouts.py` refuses to mix the two tables.

### The best worker count is layout-dependent

Running the ladder against each layout rather than only against tar shards, medians in samples per second:

| Workers | Proc/core | loose files | SquashFS | tar shards |
|---:|---:|---:|---:|---:|
| 0 | 0.14 | 371 | 2 057 | 2 700 |
| 2 | 0.43 | 892 | 3 449 | 5 237 |
| 4 | 0.71 | 1 108 | 7 073 | 7 895 |
| 7 | 1.14 | 1 327 | **9 934** | 12 696 |
| 8 | 1.29 | | **10 031** | |
| 10 | 1.57 | | 9 006 | |
| 11 | 1.71 | | 6 618 | |
| 13 | 2.00 | **2 148** | 2 575 | **14 815** |

SquashFS plateaus between 7 and 10 workers and then falls away, losing three quarters of its throughput by 13. Loose files and tar shards both improve all the way to 13. CPU utilisation at the SquashFS collapse is 1 to 3 %, so nothing is compute-bound; the image is one mount serving every worker, and past roughly ten concurrent readers contention on it dominates.

13 workers is therefore near-optimal for tar shards and close to the worst available choice for SquashFS, which is why the table above rates SquashFS below a loose tree. At its own best rung it reaches 10 031 against 2 148 for loose files, a 4.7x gain.

The `w=13` SquashFS figure is a median of eight runs spanning 388 to 3 823, against 9 273 to 10 594 at 7 workers. Past its contention point the layout does not degrade predictably.

### The SquashFS limit is a reader count, not a share of the allocation

Doubling the allocation to `--cpus-per-task=14` tests whether the ceiling moves with it:

| Workers | SquashFS @ 7 cores | SquashFS @ 14 cores | tar shards @ 14 cores |
|---:|---:|---:|---:|
| 4 | 7 073 | 6 663 | |
| 8 | **10 031** | **7 802** | |
| 13 | 2 575 | 3 497 | **17 917** |
| 16 | | 1 399 | |
| 20 | | 823 | |
| 27 | | 754 | 17 826 |

It does not move. SquashFS peaks at the same 8 workers on both allocations, so the limit is roughly eight concurrent readers of one image, not a fraction of the cores you hold. Twice the CPU bought nothing, and by 27 workers throughput is down to 754, a tenth of the peak.

Tar shards behave the opposite way, 14 815 at 7 cores becoming 17 917 at 14, because 41 separate files carry no single contention point. CPU utilisation stays near zero across every SquashFS rung.

## 5. Compare Storage Placement

> Scratch, flash, or node-local `/tmp`?

```bash
sbatch jobs/run_storage_comparison.sh configs/staging/scratch.yaml
sbatch jobs/run_storage_comparison.sh configs/staging/tmp.yaml
./jobs/run_stage.sh flash           # optional, needs a flash allocation

squeue -u $USER                     # wait until empty before comparing

python3 scripts/compare_storage.py "$TUTORIAL_ROOT"/outputs/storage/*/run_summary.json \
    --output "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json
```

Compute-node `/tmp` lives in memory and is charged against the job's memory allocation, so the job refuses to stage anything that would take more than a safety fraction of it. The copy is validated before it is measured and removed afterwards, and its cost is included below.

Two repeats each:

| Placement | min / median / max | Epoch s | Staging s | Break-even |
|---|---:|---:|---:|---:|
| scratch | 13243 / **13475** / 13708 | 3.712 | 0 | - |
| flash | 13648 / **14074** / 14500 | 3.556 | 0 | immediate |
| tmp | 13281 / **13719** / 14158 | 3.648 | 4.755 | **75 epochs** |

All three ranges overlap. Nothing here separates the placements on speed, so the decision rests on setup cost, and scratch has none.

Staging saved 0.064 s per epoch for a 4.755 s copy, so it repays after 75 epochs; a three-epoch job pays 4.755 s to save 0.19 s. `/tmp` was 1.8 % faster in steady state and slower over the job as a whole.

## 6. Validate Distributed Reading

> Do the ranks read unique, balanced data?

```bash
python3 scripts/shard_summary.py "$TUTORIAL_ROOT"/shards --readers 8   # free, before any job
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
```

Eight tasks with seven cores each reproduce the rank count and per-rank CPU share of a full LUMI-G job. They do not reproduce its NUMA layout or CPU-GPU binding; this runs on a CPU partition.

Runs use `measured_epochs: 1` rather than a batch count, so "every sample read once, by exactly one reader" is checkable. Three deliberately broken cases ship alongside the healthy one:

| Case | Reads | Unique | Duplicates | Idle | Elapsed spread | Aggregate/s | Valid |
|---|---:|---:|---:|---|---:|---:|---|
| healthy | 50 000 | 50 000 | 0 | none | 11 % | 16 630 | yes |
| too few shards | 50 000 | 50 000 | 0 | 6 of 8 | 99 % | 6 751 | no |
| duplicate samples | 400 000 | 50 000 | 350 000 | none | 0.03 % | **21 030** | no |
| imbalanced shards | 50 000 | 50 000 | 0 | none | 33 % | 15 530 | yes |

The duplicate case has the highest aggregate throughput of the four, 26 % above the healthy run, and is the worst result. Eight ranks each read every shard: 400 000 reads to cover 50 000 samples, 87.5 % of the work wasted, 19.0 s of wall time against the healthy run's 3.2 s. Its rank spread is 0.03 % because every rank is doing identically useless work. No throughput number can detect this. The verdict compares which samples each rank read, not how many.

Two rows invert the usual reading. `too few shards` has zero duplicates and full coverage: the data is read correctly, and three quarters of the allocation does nothing. `imbalanced shards` reports `PARTITIONING_VALID=true` while wasting a third of the allocation on waiting for the slowest rank. Correct partitioning is necessary, not sufficient.

## 7. Render The Readiness Decision

> Is the data path ready?

```bash
python3 scripts/render_decision.py --planned-epochs 3 \
    --inspection  "$TUTORIAL_ROOT"/outputs/inspection/dataset_report.json \
    --layouts     "$TUTORIAL_ROOT"/outputs/layout-comparison/summary.json \
    --workers     "$TUTORIAL_ROOT"/outputs/worker-comparison/summary.json \
    --storage     "$TUTORIAL_ROOT"/outputs/storage-comparison/summary.json \
    --distributed "$TUTORIAL_ROOT"/outputs/distributed/healthy/distributed_verdict.json
```

```text
DATA_READINESS=READY_WITH_CAUTION
RECOMMENDED_LAYOUT=webdataset          RECOMMENDED_STORAGE=scratch
RECOMMENDED_WORKERS_PER_RANK=13        NODE_LOCAL_STAGING=not-recommended
DISTRIBUTED_PARTITIONING=valid         MAIN_LIMITING_FACTOR=storage-or-synchronisation
NEXT_EXPERIMENT=scaling-aware-ai-one-gcd-baseline
```

Layout and worker tuning together moved the pipeline from 405 to 14 566 samples per second, a factor of 36, with correctness verified at every step. The caution is that storage placements were indistinguishable within measurement noise.

`RECOMMENDED_LAYOUT` ranges only over the three layouts measured in step 3. Sequential Parquet was faster still at 22 469, and the tool cannot recommend a representation it was never given, nor know whether shuffling every epoch is a requirement, which would cost Parquet a factor of 25.

`--planned-epochs` decides the staging recommendation outright: the same measurements say "stage" for a long campaign and "do not stage" for a one-pass job. Duplicates, missing samples, idle ranks, and failed samples all force `NOT_READY` regardless of throughput. A missing required input gives `INCONCLUSIVE` rather than a default. Exit codes: 0 ready, 5 not ready, 6 inconclusive.

With a ready verdict, continue to [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai), which asks whether more GCDs produce useful throughput.

## License

MIT. See [LICENSE](LICENSE).
