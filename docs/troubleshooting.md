# Troubleshooting

## Submission

| Symptom | Cause and fix |
| ------- | ------------- |
| `AssocMaxSubmitJobLimit` | `SBATCH_ACCOUNT` is unset in this shell. Run `source env.sh` — sbatch resolves the account at submission time, so setting it inside the job is too late |
| `undefined environment variable(s) TUTORIAL_ROOT` | No `env.sh`, or you did not source it. Copy `env.example.sh` and edit it |
| Job produces no log file | `logs/` must exist before submitting. It is in the repository; do not delete it |
| A job "succeeds" having done nothing | Submit from the repository root. Slurm copies the batch script to a spool directory, so the scripts locate `jobs/common.sh` through `SLURM_SUBMIT_DIR` |
| Benchmark starts before its artifact exists | Two bare `sbatch` calls race. Use `./jobs/run_stage.sh <stage>`, which chains them with `afterok` |

## Configuration

| Symptom | Cause and fix |
| ------- | ------------- |
| `unknown option(s) ['num_worker'] in section 'loader'` | A typo. Unknown fields are rejected on purpose, so you never measure a configuration nobody chose |
| `loader.shuffle must be false for layout 'webdataset'` | Streaming layouts have no index to permute. Use `loader.shuffle_buffer` |
| `dataset.adapter is only used with dataset.layout: adapter` | Set both, or neither |
| `storage.location: tmp requires storage.stage_to_tmp: true` | Node-local storage starts empty; something has to put the data there |

## Data and artifacts

| Symptom | Cause and fix |
| ------- | ------------- |
| `manifest not found` | Generate the dataset first: `sbatch jobs/prepare_dataset.sh configs/datasets/metadata_heavy.yaml` |
| `shard index not found` | Build shards: `./jobs/run_stage.sh webdataset` |
| `Parquet/HDF5 artifact not found` | Convert first: `python3 scripts/convert_dataset.py --to <track> ...` |
| `... holds N rows but the manifest has M` | A stale artifact against a newer manifest. Reconvert from the same manifest |
| `... is not empty; pass --overwrite` | Regenerating over existing output needs `--overwrite` |
| `WARNING n sample(s) failed to load` | The run is not a valid comparison. Check the dataset is complete and readable |

## Comparisons

| Symptom | Cause and fix |
| ------- | ------------- |
| `BLOCKING manifest_hash differs across runs` | The runs read different data. Rebuild every layout from one manifest |
| `CONTROLLED_COMPARISON=false` | Something other than the variable under test differed. The named fields tell you what |
| `Single run per group` | Repeat before acting on a small difference |
| `... indistinguishable on steady-state speed` | The difference is inside the measurement noise. Decide on setup cost and operational fit |

## Storage and staging

| Symptom | Cause and fix |
| ------- | ------------- |
| `Refusing to stage: ... above the safety margin` | Working as intended. `/tmp` is memory: request more, stage a packaged form, or read from shared storage |
| `Refusing to stage: ... could not be determined` | Allocated memory is unknown. Run inside a Slurm allocation, or set `storage.memory_bytes` |
| A path that plainly exists is "not found" inside a container | It is not bound. Add it to `TUTORIAL_CONTAINER_BINDS` — project flash is the usual omission |
| `squashfs_mode is 'prebound' but ... is not a directory` | Bind the image there, or switch to `squashfs_mode: squashfuse` |

## Distributed

| Symptom | Cause and fix |
| ------- | ------------- |
| Exit code 4 | A partitioning problem was found. For the deliberately broken challenges this is the expected result |
| `IDLE_RANKS` non-empty | Fewer shards than readers. Rebuild with at least `world_size × num_workers` |
| Duplicates in a *healthy* run | Ranks are cycling into a second epoch. Use `measured_epochs` rather than a batch count |
| A rank dies and the job hangs | `--kill-on-bad-exit=1` should prevent it; without it, surviving ranks block in the correctness gather |

## Environment

| Symptom | Cause and fix |
| ------- | ------------- |
| `is older than Python 3.9` | LUMI's system python is 3.6. `module load cray-python`, or set `TUTORIAL_CONTAINER` |
| `PyTorch is required` | Install `.[loader]`, load a module, or set `TUTORIAL_CONTAINER` |
| `cannot import adapter module` | An optional track's extra is missing: `pip install '.[parquet]'`, `'.[hdf5]'`, `'.[huggingface]'` |
| `mksquashfs was not found` | It is a host binary, absent from PyTorch containers. The build step runs outside the container by design |
| `rclone was not found` | `module load lumio`, then `lumio-conf` |
| `remote ... does not look like a private endpoint` | Use the `-private` remote. The public one serves objects to anyone with the URL |
