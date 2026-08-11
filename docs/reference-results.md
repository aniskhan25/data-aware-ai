# Reference results

Every performance figure quoted in the README, with the environment that produced it and
the limits of what it supports.

> **These are reference measurements, not targets.** Your absolute values will differ with
> system load, software versions, container contents, and the state of your project
> storage. Compare the qualitative trends. If your ratios differ wildly from these, that is
> information about your dataset - not a sign you did it wrong.

## Environment

| | |
| --- | --- |
| System | LUMI, `small` CPU partition (Parts I-VI), login nodes for reporting |
| Container, Parts I-VII | `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif` |
| Python / PyTorch | 3.12.9 / 2.6.0+rocm6.2.4 |
| Container, optional tracks | `lumi-multitorch-u24r70f21m50t210-20260731_122833` (LAIF) |
| Python / PyTorch | 3.12.3 / 2.10.0+rocm7.0, h5py 3.16.0, pyarrow 25.0.0, datasets 5.0.0 |
| Storage | Project scratch (Lustre), project flash, compute-node `/tmp` |
| Allocation per task | `--cpus-per-task=7` (7 physical cores, 14 logical CPUs), `--mem=32G` |
| Dataset | `metadata-heavy`: 50 000 JPEG, 64×64, ~2.7 KB each, 134 MB, 100 classes |
| Repeats | 3 for layouts and formats, 2 for workers/storage, 1 per distributed case |

The two container rows matter: the optional-track numbers were taken in a different image
from Parts I-VII, so they are not directly comparable with the Part III table. The
`sif-images` PyTorch containers do not ship h5py, which is why the HDF5 track moved.

The dataset is generated deterministically from `(seed, index)`, so the same profile and
seed reproduce the same bytes and the same manifest on any machine.

## Part I - inspection

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

## Part III - layouts (3 repeats each)

| Layout | Median | Min | Max | CV | MiB/s | Mean wait | P95 wait | FS objects | Startup (CV) |
| ------ | ------ | --- | --- | - | ----- | --------- | -------- | ---------- | ------------ |
| loose-files | 405.1 | 362.5 | 1581.8 | 0.88 | 1.033 | 0.1572 | 0.7345 | 50 000 | 0.58 s (0.79) |
| squashfs | 3652.1 | 3429.8 | 4047.9 | 0.08 | 9.312 | 0.01665 | 0.05455 | 1 | 2.38 s (0.01) |
| webdataset | 6926.3 | 5571.5 | 7402.9 | 0.14 | 17.66 | 0.008237 | 0.03308 | 51 | 0.90 s (0.21) |

Artifact sizes: SquashFS image 144.6 MB (`SIZE_RATIO=1.005`, genuinely uncompressed with
`-noD -noF`); 40-50 tar shards totalling 232 MB; loose tree 144 MB in 50 002 files.

A `-noD`-only image came out at 114 MB (79 % of source) in 298 s, against 144.6 MB in
1027 s with `-noD -noF`. Small files are stored as fragments, which `-noD` does not cover.

## Part IV - worker ladder (2 repeats, 1000 batches, tar shards)

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

## Part V - storage placement (2 repeats each)

| Placement | Samples/s | Epoch s | Staging s | Validate s | Job s | Break-even |
| --------- | --------- | ------- | --------- | ---------- | ----- | ---------- |
| scratch | 13 480 | 3.712 | 0 | 0 | 6.617 | - (baseline) |
| flash | 14 070 | 3.556 | 0 | 0 | 6.177 | immediate |
| tmp | 13 720 | 3.648 | 4.755 | 0.0026 | 10.84 | **75.1 epochs** |

Run-to-run variation was 4.9-5.1 %, larger than every difference between placements
(1.8-4.4 %). The placements are indistinguishable on speed. Staged dataset occupied 0.7 %
of the 32 GB allocation.

These are predicted wall times, not costs: flash is billed at a higher rate than scratch
and is a much smaller shared resource.

## Part VI - distributed validation (8 ranks, 1 epoch)

| Case | Reads | Unique | Duplicates | Missing | Idle ranks | Throughput spread | Elapsed spread | Aggregate/s | Valid |
| ---- | ----- | ------ | ---------- | ------- | ---------- | ----------------- | -------------- | ----------- | ----- |
| healthy | 50 000 | 50 000 | 0 | 0 | none | 11.3 % | 11.3 % | 16 630 | yes |
| too few shards | 50 000 | 50 000 | 0 | 0 | 2-7 | 100 % | 99.3 % | 6 751 | no |
| duplicate samples | 400 000 | 50 000 | 350 000 | 0 | none | 0.03 % | 0.03 % | 21 030 | no |
| imbalanced shards | 50 000 | 50 000 | 0 | 0 | none | 6.5 % | 33.4 % | 15 530 | yes |

Shard sets: healthy 40 shards of 1250; too-few 2 shards of 25 000; imbalanced 40 shards
ramping 286-1714 samples (5.1× ratio, random sizes so round-robin assignment cannot cancel
the imbalance).

Elapsed times: healthy 2.82-3.18 s; duplicate 19.02-19.03 s for the same epoch.

## Part VII - readiness

```text
DATA_READINESS=READY_WITH_CAUTION       PLANNED_EPOCHS=3
RECOMMENDED_LAYOUT=webdataset           RECOMMENDED_STORAGE=scratch
RECOMMENDED_WORKERS_PER_RANK=13         NODE_LOCAL_STAGING=not-recommended
DISTRIBUTED_PARTITIONING=valid          MAIN_LIMITING_FACTOR=storage-or-synchronisation
```

The single caution: storage placements were indistinguishable within measurement noise.

## All six formats (3 repeats, 13 workers, 1000 batches, LAIF container)

Every representation under identical conditions, `min / median / max` samples per second.
This is the controlled cross-format comparison; the step 3 table above is not comparable
with it, having used 4 workers, 200 batches, and a different container.

Shuffled access (`shuffle: true`), index-addressable formats only:

| Format | min / median / max | vs loose files |
| ------ | ------------------ | -------------: |
| HDF5 | 11 356 / 11 723 / 13 228 | 4.5x |
| Arrow | 8 132 / 8 240 / 12 750 | 3.2x |
| loose files | 2 516 / 2 609 / 2 618 | - |
| SquashFS | 2 260 / 2 303 / 2 640 | 0.9x |
| Parquet | 724 / 902 / 1 345 | 0.3x |

Sequential access (`shuffle: false`):

| Format | min / median / max | vs loose files |
| ------ | ------------------ | -------------: |
| Parquet | 22 076 / 22 469 / 22 914 | 8.6x |
| tar shards | 14 160 / 15 736 / 16 482 | 6.0x |
| Arrow | 13 827 / 15 020 / 18 319 | 5.8x |
| HDF5 | 10 916 / 11 112 / 13 097 | 4.3x |

Tar shards stream and have no index to permute, so they appear only in the second table;
order comes from shard assignment and a 1000-sample shuffle buffer.

All 27 runs reported `FAILED_SAMPLES=0`, `DUPLICATE_SAMPLES=0`, `MISSING_SAMPLES=0`. The
HDF5 runs exercise the fork-safe per-process handle path with 13 workers.

Artifacts: Parquet 135 MB in 1 file (40 row groups); HDF5 138 MB in 1 file (50 chunks of
1000); Arrow 135 MB in 3 files; SquashFS 138 MB in 1 file; shards 138 MB in 41 files. All
hold 50 000 rows, byte-identical to the loose tree of 144 MB in 50 002 files by checksum.

Two results deserve care. HDF5's two ranges overlap almost entirely, so access order does
not measurably matter for it; that is not the same as shuffled being faster. And the
SquashFS row is that layout at its worst rung, not its best: see the ladder below.

## Worker ladder by layout (LAIF container, 1000 batches)

The step 4 ladder was run against tar shards only. Repeating it per layout shows that the
tuned worker count does not transfer. Medians in samples per second, 2 to 8 repeats per
cell.

| Workers | Proc/core | loose files | SquashFS | tar shards |
| ------: | --------: | ----------: | -------: | ---------: |
| 0 | 0.14 | 371 | 2 057 | 2 700 |
| 2 | 0.43 | 892 | 3 449 | 5 237 |
| 4 | 0.71 | 1 108 | 7 073 | 7 895 |
| 7 | 1.14 | 1 327 | 9 934 | 12 696 |
| 8 | 1.29 | | 10 031 | |
| 9 | 1.43 | | 9 223 | |
| 10 | 1.57 | | 9 006 | |
| 11 | 1.71 | | 6 618 | |
| 13 | 2.00 | 2 148 | 2 575 | 14 815 |

SquashFS plateaus over 7 to 10 workers and then degrades, losing about three quarters of
its peak by 13 workers. Loose files and tar shards improve monotonically to 13. CPU
utilisation during the SquashFS decline is 1 to 3 %, so the loss is contention on the
single image mount rather than compute.

The spread widens as the layout passes its contention point: the eight SquashFS runs at 13
workers span 388 to 3 823, against 9 273 to 10 594 at 7 workers. Unpredictability arrives
before, and is a better warning than, the drop in the median.

All 45 ladder runs reported zero failed, duplicate, and missing samples.

### Does the contention point scale with the allocation?

No. Repeating the ladder at `--cpus-per-task=14` (14 physical cores, 28 logical), three
repeats per cell:

| Workers | SquashFS @ 7 cores | SquashFS @ 14 cores | tar shards @ 14 cores |
| ------: | -----------------: | ------------------: | --------------------: |
| 4 | 7 073 | 6 663 | |
| 8 | 10 031 | 7 802 | |
| 13 | 2 575 | 3 497 | 17 917 |
| 16 | | 1 399 | |
| 20 | | 823 | |
| 27 | | 754 | 17 826 |

SquashFS peaks at 8 workers on both allocations, so the ceiling is a count of concurrent
readers against a single image mount, not a proportion of the cores held. Doubling the
allocation did not raise it; throughput at 27 workers is a tenth of the peak. CPU
utilisation stays at or below 1 % throughout, confirming the loss is contention.

Tar shards move the other way, 14 815 at 7 cores to 17 917 at 14, and then flatten between
13 and 27 workers. A layout spread over 41 files has no single point to contend on.

These 24 runs also reported zero failed, duplicate, and missing samples.

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
