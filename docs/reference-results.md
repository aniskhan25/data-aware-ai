# Reference results

Every performance figure quoted in the README, with the environment that produced it and
the limits of what it supports.

> **These are reference measurements, not targets.** Your absolute values will differ with
> system load, software versions, container contents, and the state of your project
> storage. Compare the qualitative trends. If your ratios differ wildly from these, that is
> information about your dataset — not a sign you did it wrong.

## Environment

| | |
| --- | --- |
| System | LUMI, `small` CPU partition (Parts I–VI), login nodes for reporting |
| Container | `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif` |
| Python / PyTorch | 3.12.9 / 2.6.0+rocm6.2.4 |
| Storage | Project scratch (Lustre), project flash, compute-node `/tmp` |
| Allocation per task | `--cpus-per-task=7` (7 physical cores, 14 logical CPUs), `--mem=32G` |
| Dataset | `metadata-heavy`: 50 000 JPEG, 64×64, ~2.7 KB each, 134 MB, 100 classes |
| Repeats | 3 for layouts, 2 for workers/storage/formats, 1 per distributed case |

The dataset is generated deterministically from `(seed, index)`, so the same profile and
seed reproduce the same bytes and the same manifest on any machine.

## Part I — inspection

```text
TOTAL_FILES=50002          TOTAL_BYTES=143870558
MEDIAN_FILE_BYTES=2673     P5=2637   P95=2705
P95_TO_MEDIAN_RATIO=1.013  FILE_SIZE_CV=7.763
SMALL_FILE_FRACTION=1      FILESYSTEM_OBJECTS=50104
MAX_FILES_IN_ONE_DIRECTORY=500   MAX_DIRECTORY_DEPTH=2
CANDIDATE_EXPERIMENTS=loose-file-baseline,squashfs,webdataset,tmp-staging
```

Note `FILE_SIZE_CV=7.763` against `P95_TO_MEDIAN_RATIO=1.013`. The coefficient of
variation is dominated by one 2.4 MB manifest among 50 000 uniform images. This is why the
shard-balancing suggestion is driven by the robust ratio, not by CV.

## Part III — layouts (3 repeats each)

| Layout | Median | Min | Max | CV | MiB/s | Mean wait | P95 wait | FS objects | Startup (CV) |
| ------ | ------ | --- | --- | -- | ----- | --------- | -------- | ---------- | ------------ |
| loose-files | 405.1 | 362.5 | 1581.8 | 0.88 | 1.033 | 0.1572 | 0.7345 | 50 000 | 0.58 s (0.79) |
| squashfs | 3652.1 | 3429.8 | 4047.9 | 0.08 | 9.312 | 0.01665 | 0.05455 | 1 | 2.38 s (0.01) |
| webdataset | 6926.3 | 5571.5 | 7402.9 | 0.14 | 17.66 | 0.008237 | 0.03308 | 51 | 0.90 s (0.21) |

Artifact sizes: SquashFS image 144.6 MB (`SIZE_RATIO=1.005`, genuinely uncompressed with
`-noD -noF`); 40–50 tar shards totalling 232 MB; loose tree 144 MB in 50 002 files.

A `-noD`-only image came out at 114 MB (79 % of source) in 298 s, against 144.6 MB in
1027 s with `-noD -noF`. Small files are stored as fragments, which `-noD` does not cover.

## Part IV — worker ladder (2 repeats, 1000 batches, tar shards)

| Workers | Samples/s | Proc/physical core | CPU util | Wait frac | Peak MiB | Invol cs/s |
| ------- | --------- | ------------------ | -------- | --------- | -------- | ---------- |
| 0 | 2167 | 0.14 | 6.3 % | 98.6 % | 521 | 1024 |
| 2 | 4048 | 0.43 | 1.5 % | 92.6 % | 546 | 292 |
| 6 | 9515 | 1.00 | 3.5 % | 83.1 % | 546 | 1896 |
| 7 | 9733 | 1.14 | 3.5 % | 83.4 % | 520 | 185 |
| 13 | 14570 | 2.00 | 4.9 % | 76.5 % | 546 | 1386 |
| 28 | 15123 | 4.14 | 5.0 % | 75.5 % | 542 | 1551 |

`ALLOCATED_PHYSICAL_CORES=7`, `LOGICAL_CPUS_IN_AFFINITY=14`, `RECOMMENDED_WORKERS=13`,
`MAIN_LIMITING_FACTOR=storage-or-synchronisation`.

**The involuntary context-switch column is not usable evidence here.** It reads 1024/s at
*zero* workers, where there are no child processes at all, and 185/s at seven workers. An
earlier ladder on the same configuration recorded 4.2/s at zero workers. The metric is
dominated by other jobs sharing the node, not by this pipeline's worker count. It is
reported because it is the right signal in principle; on a shared partition it is too
noisy to support a conclusion, and an earlier version of this tutorial wrongly claimed a
25-fold rise as a finding.

## Part V — storage placement (2 repeats each)

| Placement | Samples/s | Epoch s | Staging s | Validate s | Job s | Break-even |
| --------- | --------- | ------- | --------- | ---------- | ----- | ---------- |
| scratch | 13 480 | 3.712 | 0 | 0 | 6.617 | — (baseline) |
| flash | 14 070 | 3.556 | 0 | 0 | 6.177 | immediate |
| tmp | 13 720 | 3.648 | 4.755 | 0.0026 | 10.84 | **75.1 epochs** |

Run-to-run variation was 4.9–5.1 %, larger than every difference between placements
(1.8–4.4 %). The placements are indistinguishable on speed. Staged dataset occupied 0.7 %
of the 32 GB allocation.

These are predicted wall times, not costs: flash is billed at a higher rate than scratch
and is a much smaller shared resource.

## Part VI — distributed validation (8 ranks, 1 epoch)

| Case | Reads | Unique | Duplicates | Missing | Idle ranks | Throughput spread | Elapsed spread | Aggregate/s | Valid |
| ---- | ----- | ------ | ---------- | ------- | ---------- | ----------------- | -------------- | ----------- | ----- |
| healthy | 50 000 | 50 000 | 0 | 0 | none | 11.3 % | 11.3 % | 16 630 | yes |
| too few shards | 50 000 | 50 000 | 0 | 0 | 2–7 | 100 % | 99.3 % | 6 751 | no |
| duplicate samples | 400 000 | 50 000 | 350 000 | 0 | none | 0.03 % | 0.03 % | 21 030 | no |
| imbalanced shards | 50 000 | 50 000 | 0 | 0 | none | 6.5 % | 33.4 % | 15 530 | yes |

Shard sets: healthy 40 shards of 1250; too-few 2 shards of 25 000; imbalanced 40 shards
ramping 286–1714 samples (5.1× ratio, random sizes so round-robin assignment cannot cancel
the imbalance).

Elapsed times: healthy 2.82–3.18 s; duplicate 19.02–19.03 s for the same epoch.

## Part VII — readiness

```text
DATA_READINESS=READY_WITH_CAUTION       PLANNED_EPOCHS=3
RECOMMENDED_LAYOUT=webdataset           RECOMMENDED_STORAGE=scratch
RECOMMENDED_WORKERS_PER_RANK=13         NODE_LOCAL_STAGING=not-recommended
DISTRIBUTED_PARTITIONING=valid          MAIN_LIMITING_FACTOR=storage-or-synchronisation
```

The single caution: storage placements were indistinguishable within measurement noise.

## Optional tracks (2 repeats, 13 workers, 1000 batches)

| Access | Parquet | Hugging Face (Arrow) |
| ------ | ------- | -------------------- |
| Random (`shuffle: true`) | 983 | 11 290 |
| Sequential (`shuffle: false`) | 20 070 | 17 090 |

Artifacts: Parquet 135 MB in 1 file (40 row groups); Arrow 135 MB in 3 files. Both hold
all 50 000 rows, byte-identical to the loose files by checksum.

These runs used 13 workers and 1000 batches while the Part III table used 4 workers and
200, so the two tables are **not** directly comparable. `compare_layouts.py` reports
`CONTROLLED_COMPARISON=false` and names the differing fields when you mix them.

## Measurement limitations

1. **Repeat counts are modest.** Two or three per configuration shows variability; it does
   not characterise it tightly. For a decision you intend to defend, run five or more and
   report median and range.
2. **The partition is shared.** Other jobs on the same node affect timings, and the
   involuntary context-switch metric demonstrably so.
3. **Execution order was not randomised.** Layouts and placements ran in a fixed order, so
   a systematic drift in system load would not be separated from a real effect.
4. **Page-cache state was not controlled.** The loose-file outlier at 1582 samples/s is
   consistent with page-cache warming from a previous job; that is an interpretation, not
   an established cause.
5. **One dataset, one machine.** Every conclusion is about a metadata-heavy image dataset
   on LUMI scratch. A decode-heavy or large-sample dataset shifts the balance, and the
   method is what transfers, not the numbers.
