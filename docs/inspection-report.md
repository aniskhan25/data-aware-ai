# The inspection report

`scripts/inspect_dataset.py` writes one JSON report, schema version `1.0`. This
document is that schema.

The report has a deliberate split: everything outside the `provenance` block is a
**deterministic function** of the inspected tree and the configured thresholds.
Inspect the same tree twice with the same settings and those fields are identical.
`provenance` holds the timestamp, hostname, and walk duration, which are not.

## Top level

| Key | Type | Meaning |
| --- | ---- | ------- |
| `schema_version` | string | `"1.0"` |
| `dataset_path` | string | The path that was inspected |
| `settings` | object | Thresholds used, echoed so the report is self-describing |
| `tree` | object | Counts over the whole tree |
| `file_sizes` | object | Size distribution |
| `size_thresholds` | array | Coarse size histogram |
| `small_files` | object | Count and fraction below the small-file threshold |
| `extensions` | array | Extension mix, largest first |
| `directories` | object | Directory shape |
| `packaging` | object | Packaging and sharding arithmetic |
| `memory` | object | Allocated memory and node-local staging headroom |
| `format_hints` | array | Optional-track pointers derived from file names |
| `candidates` | array | Suggested next experiments, each with its reason |
| `limitations` | array | What the tool cannot determine |
| `provenance` | object | Timestamp, hostname, walk duration, Slurm job ID |

## `tree`

| Key | Meaning |
| --- | ------- |
| `total_files` | Regular files found. Symlinks are not counted here |
| `total_bytes` | Sum of file sizes. May double-count hard links |
| `total_gib` | `total_bytes` in GiB |
| `directories` | Directories visited, including the root |
| `symlinks` | Symlinks, counted but never followed |
| `other_entries` | Sockets, FIFOs, devices |
| `hardlinked_files` | Files with a link count above one |
| `unreadable_directories` | Directories that could not be listed |
| `unreadable_files` | Entries whose metadata could not be read |
| `filesystem_objects` | `total_files + directories` — what packaging would collapse |

A non-zero `unreadable_*` count means the report describes only the readable part of
the tree. The tool says so on stderr and adds a `fix-permissions-first` candidate
rather than silently reporting partial totals as if they were complete.

## `file_sizes`

`min_bytes`, `p5_bytes`, `median_bytes`, `mean_bytes`, `p95_bytes`, `max_bytes`,
plus two spread measures:

- **`p95_to_median_ratio`** — robust spread. This is what drives the
  shard-balancing suggestion.
- **`coefficient_of_variation`** — reported for completeness, but outlier-sensitive.
  One stray manifest or checkpoint among otherwise uniform samples pushes it above
  any fixed threshold while the bulk of the distribution is tight. It is not used
  for any suggestion, for exactly that reason.

## `packaging`

| Key | Meaning |
| --- | ------- |
| `filesystem_objects_now` | Objects the dataset presents today |
| `filesystem_objects_as_squashfs` | `1` — a SquashFS image is one file, whatever it contains |
| `already_compressed_byte_fraction` | Share of bytes in already-compressed formats |
| `compression_likely_to_help` | False when most bytes are already compressed |
| `estimated_packaged_bytes` | Approximated as `total_bytes` |
| `suggested_shards` | `ceil(total_bytes / target_shard_bytes)` |
| `suggested_samples_per_shard` | `total_files / suggested_shards` |

`estimated_packaged_bytes` is an approximation, not a measurement: for JPEG or PNG
trees, packaging changes the object count rather than the byte count. For
uncompressed data (`.npy`, `.csv`, `.bin`) the packaged image can be markedly
smaller, which is what `compression_likely_to_help` flags.

The shard suggestion is arithmetic on a target size. It knows nothing about your
reader count — **shards must be at least as numerous as readers**, or ranks sit
idle. Part VI tests that directly.

## `memory`

| Key | Meaning |
| --- | ------- |
| `allocated_bytes` | Memory the job may use, or `null` if unknown |
| `source` | Where that number came from |
| `safety_fraction` | Largest share of memory a staged dataset may occupy |
| `dataset_fraction_of_memory` | `total_bytes / allocated_bytes`, or `null` |
| `tmp_staging_within_safety_margin` | `true`, `false`, or `null` when unknown |
| `note` | Explanation, including how to get an answer when it is `null` |

Detection order: `--memory-bytes`, then `SLURM_MEM_PER_NODE`, then
`SLURM_MEM_PER_CPU × SLURM_CPUS_PER_TASK`. Outside an allocation the answer is
`null`, not a guess — compute-node `/tmp` is memory charged against the job, so
advice based on an invented number would be worse than no advice.

To get meaningful staging advice, run the inspection **inside the allocation you
intend to train with**, or pass `--memory-bytes` explicitly.

## `candidates`

An ordered array of `{experiment, reason}`. `loose-file-baseline` is always first:
every comparison needs a reference point.

| Experiment | Proposed when |
| ---------- | ------------- |
| `loose-file-baseline` | Always |
| `squashfs` | Many files and most of them small |
| `webdataset` | Many files; alongside `squashfs` when they are small |
| `benchmark-native-representation` | Few files — packaging rarely pays |
| `shard-balancing` | `p95_to_median_ratio` at or above 4 |
| `tmp-staging` | Dataset fits within the memory safety margin |
| `avoid-tmp-staging` | Dataset exceeds it |
| `fix-permissions-first` | Any unreadable entries |
| `none` | No files found |

The thresholds behind these are module constants in `dataaware/inspection.py`
(`MANY_FILES_TRIGGER`, `SMALL_FILE_FRACTION_TRIGGER`, `SIZE_RATIO_TRIGGER`,
`DEFAULT_TMP_SAFETY_FRACTION`), each with its rationale. They are starting points
chosen to be defensible, not properties of the filesystem. Change them when your
workload justifies it — and say so when you report results.

## What the report cannot tell you

Repeated in every report's `limitations`, because a suggestion read as a finding is
the main way this tool could mislead:

- It reads metadata only. It never opens a file, so it cannot know decode cost.
- It cannot tell whether the workload needs random access by path, or whether a
  stream of samples would do.
- It cannot tell whether sample order matters, or whether records are mutable.
- It cannot tell whether an ecosystem or library mandates a format.
- Extension hints describe file names, not how data is used.
- Every candidate is a hypothesis to be measured.

A dataset of a million tiny files with `squashfs` and `webdataset` suggested is not
a dataset that should be packaged. It is a dataset worth running Part III on.
