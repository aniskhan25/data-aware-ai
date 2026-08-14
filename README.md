# Data-Aware AI on LUMI

How datasets should be inspected, packaged, placed, and read on LUMI, and what it costs when they are not.

> A dataset that can be read is not a dataset that can be read efficiently at scale.

This tutorial assumes completion of the LUMI onboarding material and familiarity with Slurm, project storage, and official LUMI containers.

Each step asks one question, runs it against a synthetic dataset, and reports the measurement. Several steps break something on purpose, because the failures are the part worth recognising: duplicate reads that show up as record throughput, ranks that sit idle while one does the work, a staging copy that never repays, a layout that collapses past a worker count that suits a different layout.

The dataset is generated deterministically, so every number below is reproducible. Measurements come from LUMI project scratch with the `metadata-heavy` profile, 50 000 JPEG files of about 2.7 KB, reported as `min / median / max` samples per second.

The command behind each result is collapsed under **Reproduce this measurement**. Open it only if you intend to run the step yourself; the findings read without it.

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

<details>
<summary>Reproduce this measurement</summary>

```bash
sbatch jobs/inspect_dataset.sh
```

</details>

## 2. Establish The Baseline

> What does the unmodified dataset cost?

Three repeats, one file per sample:

| Layout | min / median / max | FS objects | Data wait |
|---|---:|---:|---:|
| loose-files | 1317 / **3813** / 6498 | 50 002 | 93 % |

The loop spends almost all its time waiting for data. More striking is that three runs of the identical measurement returned 1317, 3813 and 6498, a factor of five apart, because throughput here depends on how much of the tree the page cache is holding when the job starts. A single loose-file number is not a baseline; the range is. `FAILED_SAMPLES`, `DUPLICATE_SAMPLES`, and `MISSING_SAMPLES` are zero in all three.

<details>
<summary>Reproduce this measurement</summary>

```bash
sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
```

</details>

## 3. Compare Dataset Layouts

> Does packaging or sharding help?

Three repeats each, run one after another rather than side by side:

| Layout | min / median / max | On disk | FS objects | Startup |
|---|---:|---:|---:|---:|
| loose-files | 1317 / **3813** / 6498 | 137 MiB | 50 002 | 0.88 s |
| squashfs | 4775 / **5259** / 5810 | 138 MiB | 1 | 2.71 s |
| webdataset | 6646 / **7465** / 8577 | 220 MiB | 41 | 0.61 s |

Read the ranges before the medians. The loose tree spans a factor of 4.9 between its slowest and fastest run of the same measurement; SquashFS spans 1.2 and tar shards 1.3.

That spread is the finding, and it is why this step reports no speedup factor. A loose-file measurement is dominated by how much of the tree the page cache happens to be holding, which depends on what ran on that node beforehand. Against a cold tree the packaged layouts look several times faster; against a warm one the gap nearly closes. Neither number is wrong and neither is reproducible.

What is reproducible is the floor. Tar shards never dropped below 6646 and SquashFS never below 4775, while the loose tree reached 1317. Packaging does not just raise throughput, it makes throughput something you can count on, because reading one object or 41 is far less exposed to the state of a shared filesystem than reading fifty thousand.

The two packagings differ in what they cost to store. SquashFS is 138 MiB against the tree's 137 MiB, essentially free, because `mksquashfs -noD -noF` stores the samples uncompressed: JPEG is already compressed, so compressing it again would spend CPU on every read and save nothing.

Tar shards cost 220 MiB for the same samples, a 60 % overhead, because tar pads every member out to a 512-byte boundary and a 2.7 KB sample wastes most of a block. What that buys is the fastest reads, and shards that can be handed to separate readers, which is what step 6 needs.

<details>
<summary>Reproduce this measurement</summary>

```bash
./jobs/run_stage.sh layouts 3 run.measured_batches=1000
#                            repeats
```

Builds the image and the shards, then runs each measurement in turn. The runs are chained rather than submitted together: concurrent repeats read the same tree at the same time and warm each other's cache, which inflates the loose-file baseline and understates what packaging buys.

</details>

## 4. Tune The Input Workers

> How many processes should be reading the data?

A PyTorch DataLoader reads and decodes samples in separate processes, `num_workers` of them; zero means the training loop does that work itself. Each rung below is five runs of 1000 batches against the tar-shard layout.

The job holds 7 physical cores, and every LUMI core runs two hardware threads, so the operating system offers 14 CPUs. `Proc/core` below counts the workers plus the parent process against those 7 cores, and the rungs are chosen to land on values that mean something: **1.00** is one process per core, **2.00** fills the allocation at two per core, and **4.14** is well past what the job holds.

| Workers | min / median / max | vs 0 workers | Proc/core | CPU util | Data wait |
|---:|---:|---:|---:|---:|---:|
| 0 | 2080 / **2454** / 3363 | - | 0.14 | 7 % | 99 % |
| 2 | 3586 / **4223** / 5125 | 1.7x | 0.43 | 1 % | 94 % |
| 6 | 8663 / **11 558** / 13 377 | 4.7x | 1.00 | 4 % | 81 % |
| 7 | 9287 / **11 291** / 13 483 | 4.6x | 1.14 | 4 % | 80 % |
| 13 | 11 257 / **13 062** / 15 727 | **5.3x** | 2.00 | 5 % | 77 % |
| 28 | 14 586 / **15 966** / 16 384 | 6.5x | 4.14 | 5 % | 74 % |

```text
RECOMMENDED_WORKERS=13
MAIN_LIMITING_FACTOR=not-yet-saturated
```

Nearly all of the gain is won by the sixth worker: 2454 to 11 558, a factor of 4.7. Past that the rungs are hard to separate. 6 and 7 land on top of each other, and 13's 13 % edge over 6 comes with ranges that overlap across most of their width, so this ladder does not establish that filling both hardware threads per core helps.

28 workers is the fastest measured and also the steadiest, but it asks for four processes per core and takes capacity from other jobs on the node. 13 is the fastest count that fits the allocation, which is what the tool recommends. It also reports the ladder as not yet saturated: throughput was still climbing at the highest rung, so the ceiling is past 28 and this ladder did not find it.

Workers did not fix the bottleneck, they only widened it. Even at the best rung the loop still spends 74 % of its time waiting for data, against 99 % with no workers at all, and CPU utilisation never passes 6 %. The pipeline was never short of processing power; more workers simply kept more reads in flight at once. Three quarters of the loop is still waiting, which is what step 5 goes after.

### That number does not transfer between layouts

The ladder above ran against tar shards. Running it against each layout in turn, three repeats, medians in samples per second:

| Workers | Proc/core | loose files | SquashFS | tar shards |
|---:|---:|---:|---:|---:|
| 2 | 0.43 | 1 056 | 3 380 | 4 400 |
| 8 | 1.29 | 3 427 | **7 327** | 12 875 |
| 13 | 2.00 | 6 887 | 1 734 | **14 521** |
| 28 | 4.14 | 13 730 | 621 | 14 156 |

SquashFS peaks at 8 workers and then falls apart: 24 % of its peak at 13 workers, 8 % at 28. The image is a single mount and every worker reads through it, so past roughly eight concurrent readers contention dominates. Nothing is compute-bound at any rung. That ceiling is a count of concurrent readers rather than a share of the allocation: doubling the job to 14 cores left the peak at the same 8 workers.

Tar shards peak at 13 and hold flat to 28, because 41 separate files have no single point to contend on. Loose files keep climbing throughout, but read their spread before believing it: at 13 workers the three repeats span 6041 to 14 942, a factor of 2.5, while SquashFS at its peak spans 1.3.

So 13 workers, the number the ladder above recommends, is near-optimal for tar shards and one of the worst choices available for SquashFS. A worker count tuned against one layout can cost more than the layout change gained.

### Choosing a format and a worker count

Median samples/s at 13 workers, all six on the same 50 000 samples, with each format's own best worker count from a separate ladder:

| Format | Shuffled | Sequential | Best workers |
|---|---:|---:|---|
| Parquet | 902 | **22 469** | **8** |
| tar shards | streams only | **15 736** | 13 or more |
| Arrow | 8 240 | 15 020 | 13 |
| HDF5 | **11 723** | 11 112 | 13 |
| loose files | 2 609 | not measured | 13 or more |
| SquashFS | 2 303 | not measured | **8** |

Read in order, Parquet is fastest, and tar shards are next while also being the only layout here that hands disjoint pieces to separate readers, which step 6 needs. If the workload must shuffle every epoch, HDF5 is the one format whose speed does not change with access order, and Parquet becomes the worst choice available at a twenty-fivefold penalty.

The throughput columns are all at 13 workers, which is what makes them comparable, but 13 suits only half the table. Laddering each format separately puts Parquet's peak at 8 workers, after which it gives back 13 % by 28; SquashFS peaks at 8 too. HDF5 genuinely peaks at 13. Arrow was still climbing at 28, but 28 overruns the allocation, so 13 is the fastest count it can actually use. Single-file artifacts reach their limit sooner than a directory of 40 shards does, and 13 is the right answer only for the two that happen to land there.

Read the SquashFS and Parquet rows with that in mind. SquashFS at 2 303 is what it does at 13 workers, well past its ceiling; at 8 it is roughly four times that.

<details>
<summary>Reproduce this measurement</summary>

```bash
./jobs/run_worker_ladder.sh webdataset 5 run.measured_batches=1000
#                           layout     repeats
```

Runs the rungs one at a time, interleaving the repeats and reversing the order each round so no rung always reads the warm cache, then writes `outputs/worker-comparison/summary.json`.

</details>

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

<details>
<summary>Reproduce this measurement</summary>

```bash
./jobs/run_stage.sh storage
./jobs/run_stage.sh flash     # only if the project has a flash allocation
```

Writes `outputs/storage-comparison/summary.json`.

</details>

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

<details>
<summary>Reproduce this measurement</summary>

```bash
sbatch jobs/run_distributed_loader.sh configs/distributed/healthy.yaml
#                                     also: too_few_shards, duplicate_samples, imbalanced_shards
```

`scripts/shard_summary.py "$TUTORIAL_ROOT"/shards --readers 8` predicts idle readers before any job is submitted.

</details>

## 7. Render The Readiness Decision

> Is the data path ready?

```text
DATA_READINESS=READY_WITH_CAUTION
RECOMMENDED_LAYOUT=webdataset          RECOMMENDED_STORAGE=scratch
RECOMMENDED_WORKERS_PER_RANK=13        NODE_LOCAL_STAGING=not-recommended
DISTRIBUTED_PARTITIONING=valid         MAIN_LIMITING_FACTOR=not-yet-saturated
NEXT_EXPERIMENT=scaling-aware-ai-one-gcd-baseline
```

Layout and worker tuning together took the pipeline from a loose tree that returned anywhere between 1317 and 6498 samples per second to tar shards at 13 062, with correctness verified at every step. The caution is that storage placements were indistinguishable within measurement noise.

`RECOMMENDED_LAYOUT` ranges only over the three layouts measured in step 3. Sequential Parquet was faster still at 22 469, and the tool cannot recommend a representation it was never given, nor know whether shuffling every epoch is a requirement, which would cost Parquet a factor of 25.

`--planned-epochs` decides the staging recommendation outright: the same measurements say "stage" for a long campaign and "do not stage" for a one-pass job. Duplicates, missing samples, idle ranks, and failed samples all force `NOT_READY` regardless of throughput. A missing required input gives `INCONCLUSIVE` rather than a default. Exit codes: 0 ready, 5 not ready, 6 inconclusive.

With a ready verdict, continue to [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai), which asks whether more GCDs produce useful throughput.

<details>
<summary>Reproduce this measurement</summary>

```bash
python3 scripts/render_decision.py --planned-epochs 3
```

Each input defaults to where its step wrote it. Any that is missing is reported as missing rather than assumed to be fine.

</details>

## License

MIT. See [LICENSE](LICENSE).
