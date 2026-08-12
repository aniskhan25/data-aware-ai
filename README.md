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

The inspection walks the tree reading file metadata only, never opening a file, so it finishes in seconds however large the dataset.

```text
TOTAL_FILES=50002               TOTAL_GIB=0.134
MEDIAN_FILE_BYTES=2673          P95_TO_MEDIAN_RATIO=1.013
SMALL_FILE_FRACTION=1           FILESYSTEM_OBJECTS=50104
MAX_FILES_IN_ONE_DIRECTORY=500  DATASET_FRACTION_OF_MEMORY=0.004187
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
```

50 002 files holding 137 MiB, every one under the 64 KB small-file threshold and a median of 2.7 KB. Sizes are uniform, the 95th percentile landing within 1.3 % of the median. The tree presents 50 104 objects to the filesystem, and at 0.4 % of the job's memory allocation it would fit in node-local `/tmp`.

```bash
sbatch jobs/inspect_dataset.sh
```

## 2. Establish The Baseline

> What does the unmodified dataset cost?

Three repeats, one file per sample:

| Layout | min / median / max | FS objects | Data wait |
|---|---:|---:|---:|
| loose-files | 362 / **405** / 1582 | 50 002 | 100 % |

405 samples per second from a tree of 50 002 files, with the loop waiting on data essentially the whole time. The 1582 outlier is consistent with page-cache warming by an earlier job on the same node. `FAILED_SAMPLES`, `DUPLICATE_SAMPLES`, and `MISSING_SAMPLES` are zero, and every later comparison is against this run.

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

## 3. Compare Dataset Layouts

> Does packaging or sharding help?

Three repeats each:

| Layout | min / median / max | vs baseline | On disk | FS objects | Startup |
|---|---:|---:|---:|---:|---:|
| loose-files | 362 / **405** / 1582 | - | 137 MiB | 50 002 | 1.37 s |
| squashfs | 3430 / **3652** / 4048 | **9.0x** | 138 MiB | 1 | 2.38 s |
| webdataset | 5571 / **6926** / 7403 | **17.1x** | 220 MiB | 51 | 0.64 s |

Packing the samples into a single SquashFS image made them 9 times faster to read. Splitting them into 50 tar shards made them 17 times faster.

Speed was not the only gain. Across the three repeats the loose tree's fastest run was 4.4 times its slowest, while SquashFS varied by only 1.2. One object is far less exposed to whatever else the filesystem is doing than fifty thousand are.

The two packagings differ in what they cost to store. SquashFS is 138 MiB against the tree's 137 MiB, essentially free, because `mksquashfs -noD -noF` stores the samples uncompressed: JPEG is already compressed, so compressing it again would spend CPU on every read and save nothing.

Tar shards cost 220 MiB for the same samples, a 60 % overhead, because tar pads every member out to a 512-byte boundary and a 2.7 KB sample wastes most of a block. What that buys is the fastest reads, and shards that can be handed to separate readers, which is what step 6 needs.

```bash
./jobs/run_stage.sh layouts
```

Builds the image and the shards, measures all three layouts, and writes `outputs/layout-comparison/summary.json`. Each build is chained to its own measurement, and the comparison waits on all of them.

## 4. Tune The Input Workers

> How many processes should be reading the data?

A PyTorch DataLoader reads and decodes samples in separate processes, `num_workers` of them; zero means the training loop does that work itself. Each rung below is two runs against the tar-shard layout, measuring 1000 batches rather than the default 200 because at these speeds 200 batches is under a second and too short to be stable.

The rungs are picked against the allocation. `--cpus-per-task=7` grants 7 physical cores, and each LUMI core runs two SMT threads, so Slurm's affinity mask covers 14 logical CPUs. Counting the parent process, 6 workers is one process per core, 13 fills the mask at two per core, and 28 deliberately overruns it.

| Workers | min / median / max | vs 0 workers | Proc/core | CPU util | Data wait |
|---:|---:|---:|---:|---:|---:|
| 0 | 1972 / **2167** / 2362 | - | 0.14 | 6 % | 99 % |
| 2 | 3550 / **4048** / 4546 | 1.9x | 0.43 | 2 % | 93 % |
| 6 | 9471 / **9515** / 9559 | 4.4x | 1.00 | 3 % | 83 % |
| 7 | 9217 / **9733** / 10249 | 4.5x | 1.14 | 3 % | 83 % |
| 13 | 14453 / **14566** / 14679 | **6.7x** | 2.00 | 5 % | 77 % |
| 28 | 14221 / **15123** / 16026 | 7.0x | 4.14 | 5 % | 76 % |

```text
RECOMMENDED_WORKERS=13
MAIN_LIMITING_FACTOR=storage-or-synchronisation
```

Throughput rose 6.7 times between no workers and 13. The largest single step was the last one, 9 515 at one process per core to 14 566 at two, a 53 % gain. Two workers can share a core here without competing for it because each spends most of its time blocked on the filesystem rather than computing.

28 workers is not an improvement. Its median is higher, but its range overlaps 13's, and it asks for four processes per core, taking capacity from other jobs on the node.

The diagnosis matters more than the chosen number. CPU utilisation never passed 6 % and the data wait fell only from 99 % to 77 %, so the pipeline was never short of CPU. More workers helped by keeping more reads in flight, not by adding processing power, and three quarters of the loop is still spent waiting.

### That number does not transfer between layouts

The ladder above ran against tar shards. Running it against each layout separately, medians in samples per second:

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

Loose files and tar shards improve all the way to 13 workers. SquashFS peaks at 7 to 8 and then collapses, losing three quarters of its throughput by 13, because the image is a single mount and every worker reads through it. Nothing is compute-bound at any rung.

That ceiling is a count of readers, not a share of the allocation. Doubling to `--cpus-per-task=14` left the SquashFS peak at the same 8 workers and it still fell to 754 by 27, while tar shards rose from 14 815 to 17 917 on the extra cores.

### Choosing a format and a worker count

Median samples/s at 13 workers, all six on the same 50 000 samples:

| Format | Shuffled | Sequential | Workers to use |
|---|---:|---:|---|
| Parquet | 902 | **22 469** | 13 |
| tar shards | streams only | **15 736** | 13 or more |
| Arrow | 8 240 | 15 020 | 13 |
| HDF5 | **11 723** | 11 112 | 13 |
| loose files | 2 609 | not measured | 13 or more |
| SquashFS | 2 303 | not measured | **8** |

Read in order, Parquet is fastest, and tar shards are next while also being the only layout here that hands disjoint pieces to separate readers, which step 6 needs. If the workload must shuffle every epoch, HDF5 is the one format whose speed does not change with access order, and Parquet becomes the worst choice available at a twenty-fivefold penalty.

Read the SquashFS row with its worker count. 2 303 is what it does at 13, past its ceiling; at 8 it reaches 10 031, which would put it second in the shuffled column. It is the choice when the application needs ordinary file paths, and it needs the ladder run against it rather than the tar-shard recommendation borrowed.

Only the three core layouts were laddered across worker counts. For the format tracks, 13 is where they were measured, not a demonstrated ceiling. All 27 runs reported zero failed, duplicate, and missing samples.

```bash
./jobs/run_worker_ladder.sh webdataset 2 run.measured_batches=1000
#                           layout     repeats
```

Submits one job per rung plus a comparison that waits on all of them, and writes `outputs/worker-comparison/summary.json`.

## 5. Compare Storage Placement

> Scratch, flash, or node-local `/tmp`?

Compute-node `/tmp` lives in memory and is charged against the job's memory allocation, so the job refuses to stage anything that would take more than a safety fraction of it. The copy is validated before it is measured and removed afterwards, and its cost is included below.

Two repeats each:

| Placement | min / median / max | Epoch s | Staging s | Break-even |
|---|---:|---:|---:|---:|
| scratch | 13243 / **13475** / 13708 | 3.712 | 0 | - |
| flash | 13648 / **14074** / 14500 | 3.556 | 0 | immediate |
| tmp | 13281 / **13719** / 14158 | 3.648 | 4.755 | **75 epochs** |

All three ranges overlap. Nothing here separates the placements on speed, so the decision rests on setup cost, and scratch has none.

Staging saved 0.064 s per epoch for a 4.755 s copy, so it repays after 75 epochs; a three-epoch job pays 4.755 s to save 0.19 s. `/tmp` was 1.8 % faster in steady state and slower over the job as a whole.

```bash
./jobs/run_stage.sh storage
./jobs/run_stage.sh flash     # optional, only if the project has a flash allocation
```

Writes `outputs/storage-comparison/summary.json`.

## 6. Validate Distributed Reading

> Do the ranks read unique, balanced data?

Eight tasks with seven cores each reproduce the rank count and per-rank CPU share of a full LUMI-G job. They do not reproduce its NUMA layout or CPU-GPU binding; this runs on a CPU partition.

Runs use `measured_epochs: 1` rather than a batch count, so "every sample read once, by exactly one reader" is checkable. Three deliberately broken cases ship alongside the healthy one:

| Case | Reads | Unique | Duplicates | Idle | Elapsed spread | Aggregate/s | Elapsed | Valid |
|---|---:|---:|---:|---|---:|---:|---:|---|
| healthy | 50 000 | 50 000 | 0 | none | 11 % | 16 630 | **3.18 s** | yes |
| too few shards | 50 000 | 50 000 | 0 | 6 of 8 | 99 % | 6 751 | 7.41 s | no |
| duplicate samples | 400 000 | 50 000 | 350 000 | none | 0.03 % | **21 030** | **19.03 s** | no |
| imbalanced shards | 50 000 | 50 000 | 0 | none | 33 % | 15 534 | 4.01 s | yes |

The duplicate case has the highest aggregate throughput of the four and takes six times as long as the healthy run to cover the same 50 000 samples. Eight ranks each read every shard, so seven reads in eight are wasted. Its rank spread is near zero because every rank is doing identically useless work. No throughput number can detect this. The verdict compares which samples each rank read, not how many.

Two rows invert the usual reading. `too few shards` has zero duplicates and full coverage: the data is read correctly, and three quarters of the allocation does nothing. `imbalanced shards` reports `PARTITIONING_VALID=true` while wasting a third of the allocation on waiting for the slowest rank. Correct partitioning is necessary, not sufficient.

```bash
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
#                                     also: too_few_shards, duplicate_samples, imbalanced_shards
```

`scripts/shard_summary.py "$TUTORIAL_ROOT"/shards --readers 8` predicts idle readers before any job is submitted.

## 7. Render The Readiness Decision

> Is the data path ready?

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

```bash
python3 scripts/render_decision.py --planned-epochs 3
```

Each input defaults to where its step wrote it. Any that is missing is reported as missing rather than assumed to be fine.

## License

MIT. See [LICENSE](LICENSE).
