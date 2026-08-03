# LUMI storage vocabulary

A short orientation so the tutorial's storage terms are unambiguous. This is not a
substitute for the official documentation, which is the authority on quotas,
lifetimes, and policy: <https://docs.lumi-supercomputer.eu/storage/>.

| Term | What it is | What the tutorial uses it for |
| ---- | ---------- | ----------------------------- |
| Project scratch | Large parallel (Lustre) space for active job I/O | The default location for datasets, results, and the baseline in Part V |
| Project flash | Smaller, faster parallel space for workloads that genuinely need faster disk operations | The optional Part V comparison |
| Compute-node `/tmp` | Node-local space that lives in memory and consumes part of the job's allocated memory | The node-local staging experiment in Part V |
| Home | Small per-user space | Nothing. Do not run jobs against it |
| LUMI-O | S3-compatible object storage | Staging, sharing, and secondary copies (optional track) |

Three points that shape every experiment in this tutorial:

1. **Project scratch is the main location for job input, output, and checkpoint
   I/O.** Experiments treat it as the baseline, not as a fallback.
2. **Compute-node `/tmp` is memory.** Staging a dataset there consumes the job's
   allocated memory, so the staging experiment checks dataset size against the
   allocation and refuses unsafe copies. Data there disappears when the job ends,
   so results must be written back to shared storage before the job exits.
3. **Large numbers of small files create pressure on filesystem metadata
   services.** This is the operational reason behind Part I and Part III, and it
   affects other users of a shared system, not only your own job.

## Terms this tutorial keeps distinct

**Usable** — the application and its libraries can read the format. On LUMI this
is mostly a property of the software environment and container, not the
filesystem.

**Suitable** — the representation matches the workload's access pattern:
path-based random access, sequential sample streaming, column scans,
multidimensional array access, or object staging.

**Scalable** — the representation still behaves acceptably as workers, ranks,
nodes, and repeated epochs increase.

A format can be usable without being suitable, and suitable at one process without
being scalable across many. Keeping these separate is what stops "it reads fine on
my laptop" from being mistaken for "it is ready for distributed training".

## Official references

- Storage options: <https://docs.lumi-supercomputer.eu/storage/>
- Lustre: <https://docs.lumi-supercomputer.eu/storage/parallel-filesystems/lustre/>
- SquashFS and FUSE: <https://docs.lumi-supercomputer.eu/storage/formats/FUSE/>
- LUMI-O: <https://docs.lumi-supercomputer.eu/storage/lumio/>
- Moving data: <https://docs.lumi-supercomputer.eu/firststeps/movingdata/>
