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
| Repeats | 5 for the worker ladder, 3 for layouts and formats, 2 for storage, 1 per distributed case |

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

## Part III - layouts (3 repeats each, serialised, 1000 batches, 40 shards)

| Layout | min / median / max | Internal spread | FS objects | Startup | Mean wait |
| ------ | ------------------ | --------------: | ---------- | ------- | --------- |
| loose-files | 1317 / 3813 / 6498 | 4.9x | 50 002 | 0.88 s | 93 % |
| squashfs | 4775 / 5259 / 5810 | 1.2x | 1 | 2.71 s | 93 % |
| webdataset | 6646 / 7465 / 8577 | 1.3x | 41 | 0.61 s | 88 % |

An earlier campaign, 200 batches against 50 shards with the repeats submitted together,
gave loose-files 362/405/1582, squashfs 3430/3652/4048 and webdataset 5571/6926/7403, and
a headline of 9.0x and 17.1x over the baseline. Those ratios are not reproducible: they
divide by a loose-file number that moves by a factor of five with page-cache state. This
table reports no speedup for that reason.

Artifact sizes: SquashFS image 144.6 MB (`SIZE_RATIO=1.005`, genuinely uncompressed with
`-noD -noF`); 40-50 tar shards totalling 232 MB; loose tree 144 MB in 50 002 files.

A `-noD`-only image came out at 114 MB (79 % of source) in 298 s, against 144.6 MB in
1027 s with `-noD -noF`. Small files are stored as fragments, which `-noD` does not cover.

## Part IV - worker ladder (5 repeats, serialised, 1000 batches, tar shards)

| Workers | min / median / max | Spread | Proc/physical core | CPU util | Wait frac | Peak MiB |
| ------- | ------------------ | -----: | ------------------ | -------- | --------- | -------- |
| 0 | 2080 / 2454 / 3363 | 1.6x | 0.14 | 7 % | 99 % | 535 |
| 2 | 3586 / 4223 / 5125 | 1.4x | 0.43 | 1 % | 94 % | 581 |
| 6 | 8663 / 11 558 / 13 377 | 1.5x | 1.00 | 4 % | 81 % | 575 |
| 7 | 9287 / 11 291 / 13 483 | 1.5x | 1.14 | 4 % | 80 % | 563 |
| 13 | 11 257 / 13 062 / 15 727 | 1.4x | 2.00 | 5 % | 77 % | 569 |
| 28 | 14 586 / 15 966 / 16 384 | 1.1x | 4.14 | 5 % | 74 % | 560 |

`RECOMMENDED_WORKERS=13`, `MAIN_LIMITING_FACTOR=not-yet-saturated`. Zero failed,
duplicate, or missing samples across all 30 runs.

Repeats are interleaved a round at a time and the round reverses direction each pass, so
no rung sits permanently at the cold or warm end of the cache.

**The SMT claim did not survive serialisation.** Two earlier campaigns, both with their
repeats submitted concurrently, put 13 workers 53 % and then 23 % above 6 and read that as
evidence that saturating both hardware threads per core helps a blocked loader. Serialised,
the gap is 13 % and the two rungs' ranges overlap across most of their width. What the
ladder shows without ambiguity is the jump to six workers, 4.7x, and that 28 is faster
still while overrunning the allocation.

| Campaign | 0 workers | 13 workers | 13 vs 6 | Verdict |
| -------- | --------: | ---------: | ------: | ------- |
| 2 repeats, concurrent | 2167 | 14 570 | +53 % | storage-or-synchronisation |
| 5 repeats, concurrent | 2766 | 14 945 | +23 % | not-yet-saturated |
| 5 repeats, serialised | 2454 | 13 062 | +13 % | not-yet-saturated |

The recommendation of 13 held across all three. The size of the gain did not.

**The involuntary context-switch column is not usable evidence here.** Across campaigns the
zero-worker rung has read 4.2/s, 1024/s, and 1/s, on a configuration with no child
processes at all. The metric is dominated by other jobs sharing the node rather than by
this pipeline. It is reported because it is the right signal in principle; on a shared
partition it is too noisy to support a conclusion, and an earlier version of this tutorial
wrongly claimed a 25-fold rise as a finding.

## Part V - storage placement (3 repeats each, serialised, 1000 batches)

| Placement | min / median / max | Epoch s | Staging s | Validate s |
| --------- | ------------------ | ------: | --------: | ---------: |
| scratch | 13 336 / 14 058 / 15 065 | 3.56 | 0 | 0 |
| flash | 14 489 / 14 743 / 15 338 | 3.39 | 0 | 0 |
| tmp | 16 058 / 16 241 / 16 554 | 3.08 | 0.156 / 0.171 / 3.443 | 0.0018 |

All three read the same 40-shard artifact; flash is refreshed from scratch before the run
so the placements are not compared across different builds.

`/tmp` is 16 % above scratch and the ranges do not overlap, which is what memory against a
parallel filesystem should look like.

Flash and scratch differ by 4.9 % at the median, smaller than the run-to-run variation, so
three repeats cannot separate them. Two things that comparison does not establish, and
which an earlier version of the README wrongly treated as settled:

* Flash's entire range, 14 489-15 338, sits above scratch's median of 14 058, and flash is
  the steadier placement, 5.9 % spread against 13 %. That is consistent with a real but
  small effect that needs more repeats to resolve, not with no effect.
* Flash against tar shards is 40 large files read sequentially, the workload least
  sensitive to storage speed. `configs/staging/flash.yaml` pins `layout: webdataset`.

### Flash against a loose tree (3 repeats, serialised, 13 workers, 1000 batches)

Same manifest hash on both sides, so this is a controlled placement comparison.

| Placement | min / median / max | Spread | Wait frac | p95 batch wait |
| --------- | ------------------ | -----: | --------: | -------------: |
| scratch | 3575 / 14 843 / 15 276 | 4.3x | 78 % | 0.0144 s |
| flash | 11 023 / 14 273 / 14 443 | 1.3x | 74 % | 0.0067 s |

Flash is 3.8 % below scratch at the median and three repeats cannot separate the two on
that basis either. The difference is in the tails: flash's slowest run was 11 023 against
scratch's 3575, its spread is 1.3x against 4.3x, and its 95th-percentile batch wait is
0.0067 s against 0.0144 s. Faster storage did not raise the ceiling on a loose tree; it
raised the floor.

Copying the tree to flash took 11 min 25 s for 137 MiB in 50 002 files. That cost is per
file, not per byte, and it is paid before any measurement begins. Packaging the dataset
first reduces the same copy to seconds, which is an argument for doing step 3 before
step 5 rather than treating them as independent choices.

**Staging cost is bimodal, not noisy.** The same 220.9 MiB copy took 0.156 s, 0.171 s and
3.443 s. The fast cases follow a job that had just read the shards, so the source is in
page cache; the slow case reads from Lustre. At 0.478 s saved per epoch that is the
difference between repaying in a third of an epoch and repaying in seven.

The comparison tool takes the median of the three, 0.173 s, and reports break-even at 0.36
epochs. That is a correct calculation over a statistic that should not have been taken:
the median of a bimodal quantity describes neither mode. An earlier concurrent campaign
recorded 4.755 s for the same copy and reported 75 epochs, which is the same effect seen
from the other end.

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
DISTRIBUTED_PARTITIONING=valid          MAIN_LIMITING_FACTOR=not-yet-saturated
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
1000); Arrow 135 MB in 3 files; SquashFS 145 MB in 1 file; shards 231 MB in 41 files. All
hold 50 000 rows, byte-identical to the loose tree of 144 MB in 50 002 files by checksum.

Two results deserve care. HDF5's two ranges overlap almost entirely, so access order does
not measurably matter for it; that is not the same as shuffled being faster. And the
SquashFS row is that layout at its worst rung, not its best: see the ladder below.

## Worker ladder by layout (3 repeats, serialised, 1000 batches)

The step 4 ladder was run against tar shards only. Repeating it per layout shows that the
tuned worker count does not transfer. Serialised and interleaved, with the round order
reversing each pass. Medians in samples per second.

| Workers | Proc/core | loose files | SquashFS | tar shards |
| ------: | --------: | ----------: | -------: | ---------: |
| 2 | 0.43 | 1 056 | 3 380 | 4 400 |
| 8 | 1.29 | 3 427 | **7 327** | 12 875 |
| 13 | 2.00 | 6 887 | 1 734 | **14 521** |
| 28 | 4.14 | 13 730 | 621 | 14 156 |

SquashFS peaks at 8 workers and then collapses, to 24 % of its peak at 13 and 8 % at 28.
The image is a single mount and every worker reads through it. CPU utilisation stays low at
every rung, so the loss is contention rather than compute. Tar shards peak at 13 and hold
flat to 28; loose files climb throughout.

Spreads at the rungs that matter: SquashFS at its 8-worker peak spans 6482-8278, a factor
of 1.3, while loose files at 13 workers span 6041-14 942, a factor of 2.5. The loose tree
is the layout whose numbers should be trusted least, here as in Part III.

An earlier concurrent campaign put the SquashFS peak at 10 031 and its 13-worker figure at
2575, that is 26 % of peak against 24 % here. The absolute values moved with the
methodology; the collapse did not. Zero failed, duplicate, or missing samples across all
36 runs.

### Worker ladder by format (3 repeats, sequential access, LAIF container)

Medians in samples per second, with each rung as a percentage of that format's own peak.

| Workers | Parquet | HDF5 | Arrow |
| ------: | ------: | ---: | ----: |
| 2 | 10 738 (47 %) | 2 777 (18 %) | 7 208 (39 %) |
| 6 | 20 155 (87 %) | 8 524 (57 %) | 13 977 (76 %) |
| 8 | **23 075 (100 %)** | 11 619 (77 %) | 15 813 (87 %) |
| 13 | 22 033 (95 %) | **15 018 (100 %)** | 16 883 (92 %) |
| 28 | 20 163 (87 %) | 13 160 (88 %) | **18 271 (100 %)** |

Parquet peaks at 8 workers and gives back 13 % by 28. HDF5 peaks at 13. Arrow was still
rising at 28, which overruns a 7-core allocation, so 13 is the fastest count it can use.

**HDF5's absolute throughput does not hold across campaigns.** Measured at identical
settings, 13 workers and sequential access, the cross-format run gave 10 916-13 097 and
this ladder gave 13 324-16 567: the ranges do not overlap. Parquet and Arrow agreed between
the same two campaigns. Treat HDF5's numbers as an order of magnitude, not a value. The
claim that access order does not change HDF5's speed rests on the shuffled and sequential
columns *within* one campaign, which is a controlled comparison and is unaffected.

The three single-file artifacts, Parquet, HDF5, and the SquashFS image, all reach their
limit at or below the point where a 40-shard directory is still improving. All 45 runs
reported zero failed, duplicate, and missing samples.

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
