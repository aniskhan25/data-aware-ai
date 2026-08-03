# Measurement methodology

This document explains how a run is measured and what its numbers do and do not
support. Read it once before drawing conclusions from any comparison.

## What is measured

The headline metric is **useful sample throughput**: decoded, batched samples per
second reaching the training step. Raw storage bandwidth is reported alongside it
but is never the conclusion, because bandwidth alone hides

- decoding bottlenecks (bytes arrive quickly, samples do not);
- duplicated samples (high aggregate rate, redundant work);
- worker stalls and synchronisation;
- rank imbalance;
- startup cost paid once per job.

Each measured sample performs a manifest lookup, a file open, a byte read, an
image decode, batch assembly, and a small synthetic compute step.

## The synthetic compute step

`loader.compute_steps` controls a small fixed per-batch computation. It exists for
two reasons: it stops the benchmark from being a pure storage microbenchmark, and
it gives the data-wait fraction a meaningful denominator.

It is deliberately far cheaper than real training. Do not compare its absolute
time against a training step, and do not read a high data-wait fraction from this
benchmark as "my training job will be 95 % data-bound" — a real GPU step is much
larger, so the same input pipeline would show a lower fraction. What the fraction
*is* good for is comparing two layouts under an identical compute step.

## Measurement phases

A run is divided into phases, reported separately rather than blended:

| Phase    | What it covers                                          | Field |
| -------- | ------------------------------------------------------- | ----- |
| Startup  | Worker creation and the first batch, which is discarded | `startup_seconds` |
| Warm-up  | `run.warmup_batches` batches, excluded from throughput  | `warmup_seconds` |
| Measured | `run.measured_batches` batches                          | `measured_seconds` |

Startup is kept visible because layouts trade startup cost against steady-state
speed. A packaged dataset that mounts slowly but reads quickly is a different
proposition for a three-minute job than for a three-hour one.

Warm-up absorbs interpreter imports, first-touch page-cache effects, and worker
creation. It is measured but never mixed into the reported throughput.

## Batch wait and the data-wait fraction

For each measured batch the benchmark times how long it blocks waiting for the
loader to hand over a batch. That wait is reported as a mean, a median, a 95th
percentile, and a maximum. The 95th percentile matters more than the mean: a
pipeline that is usually fine but stalls badly one batch in twenty will show a
mean that looks acceptable.

`mean_data_wait_fraction` is total wait divided by total measured wall time. Read
it as "what share of the loop was spent waiting for data".

## A fixed number of batches, not epochs

The measured window is a fixed number of batches, which makes runs directly
comparable and keeps small datasets from running out of samples. When the dataset
is exhausted, the loader restarts. The cost of restarting an epoch stays inside
the batch wait time on purpose: it is a real cost the workload pays.

## Duplicate and missing samples

Correctness is reported separately from performance, because a fast run that reads
the wrong data is not a good result.

- **Duplicates** are counted **per epoch**. A correct sampler visits each sample
  at most once per epoch, so a repeat within one epoch is a defect. Repeats across
  epochs are normal and are not counted.
- **Missing samples** are only evaluated for epochs the measured window observed
  from their very first batch. A partial epoch says nothing about coverage, so it
  is reported as such in the summary's `notes` field rather than as missing data.
  `drop_last` is accounted for: a correctly dropped remainder is not missing.

## Thread counts

Each DataLoader worker is pinned to one thread (`torch.set_num_threads(1)`, plus
`OMP_NUM_THREADS=1` in the job scripts). Without this, every worker tries to use
the whole allocation, and the resulting CPU oversubscription is easily mistaken
for a storage problem. This is a deliberate choice, not a default: it means the
worker count is the variable under test in Part IV, rather than a confound.

## Controlled comparison

Two runs are comparable only when the things that are not under test match. The
comparison tools introduced in Part III check:

- manifest hash (the same samples, in the same manifest order);
- batch size, worker count, and measurement length;
- seed and shuffle setting;
- world size;
- schema version.

When they differ, the comparison says so instead of quietly producing a table.
`config_hash` is recorded in every summary and covers everything that affects what
was measured; it deliberately excludes the output directory, since writing results
elsewhere does not make it a different experiment.

## Cache effects

Repeated reads may be served from page cache, and this tutorial does not attempt
globally clean cold-cache benchmarking on a shared system. What it does instead is
keep first-pass and repeated-pass behaviour separable through the phase reporting
above, and say plainly when a result is warm. On a busy shared filesystem, treat a
single run as an observation, not a measurement.

## Repetition and variability

Results on a shared system vary with load. For any comparison you intend to act
on, run it more than once and look at the spread, not just the central value.
`dataaware.metrics` provides `coefficient_of_variation` and `spread` for this. A
single noisy result is not evidence, and a difference smaller than the run-to-run
variation is not a difference.

## What these numbers do not support

- They are not a benchmark of LUMI storage. They measure one application's input
  path on shared infrastructure under unknown concurrent load.
- They do not rank formats in general. They compare specific layouts under one
  access pattern, on one dataset.
- The tiny smoke-test profile produces numbers that are meaningless as
  performance results. Its purpose is to prove the plumbing works.
