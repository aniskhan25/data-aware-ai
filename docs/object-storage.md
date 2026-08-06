# LUMI-O: object storage

LUMI-O is S3-compatible object storage. It is **not** a mounted filesystem, and treating
it as one is the mistake this page exists to prevent.

Objects are *put* and *got* whole. There is no opening, seeking, appending, or writing in
place. A training job does not read from LUMI-O; it stages data out of LUMI-O onto a
parallel filesystem first, or it reads nothing at all.

Authority on endpoints, quotas, and lifetimes is the official documentation:
<https://docs.lumi-supercomputer.eu/storage/lumio/>.

## What it is for

```text
Local system  →  LUMI-O bucket  →  project scratch  →  training job
                                                          ↓
                       LUMI-O or elsewhere  ←  packaged outputs
```

Three legitimate uses:

- **Staging in.** Bring a dataset to LUMI once, then copy it to scratch for jobs.
- **Sharing.** Give collaborators access without giving them your project's filesystem.
- **Secondary copies.** Move results off scratch, which is working space, not storage.

## What it is not

**LUMI-O is a secondary copy, not an independent backup.** It is a reasonable place to
hold a copy of something that also exists on scratch, and that is a real use. But LUMI
provides no backup service for *any* of its storage systems, nothing versions or
replicates your objects, a deletion is a deletion, and LUMI-O data remains tied to the
project lifecycle like everything else. Anything you cannot regenerate also needs a copy
outside LUMI entirely.

**It is not fast random access.** Reading a million objects one at a time across the
network is far worse than the small-file problem Part III measures on Lustre. Package
first - one SquashFS image or a few dozen tar shards - then transfer.

## Credentials

Authentication is set up by `lumio-conf` on LUMI, which writes rclone remotes for each of
your projects. It creates two per project:

| Remote | Endpoint | Who can read |
| ------ | -------- | ------------ |
| `lumi-<project>-private` | Private | Only authenticated project members |
| `lumi-<project>-public` | Public | **Anyone with the URL, no authentication** |

Use the private one. The tutorial's script refuses a remote whose name does not contain
`private` unless you pass `--allow-public`, because publishing data by accident is not a
recoverable mistake.

Rules that are not negotiable:

- **Never commit an rclone config.** It contains access keys in plain text. `rclone.conf`
  is gitignored in this repository.
- **Never pass keys as command-line arguments.** Arguments are visible to every other
  user on the node through the process table.
- **Never echo keys into a job log.** Slurm logs sit on a shared filesystem.
- Keys are credentials for the whole project, not just for you. Treat a leak as one.

Nothing in this repository reads, prints, accepts, or stores a key.

## Verify the round trip

```bash
module load lumio          # provides rclone and lumio-conf
lumio-conf                 # once per project, interactive

python3 scripts/lumio_roundtrip.py --list-remotes

python3 scripts/lumio_roundtrip.py \
    --remote lumi-46XXXXXXX-private \
    --bucket data-aware-ai \
    --file "$TUTORIAL_ROOT/source.squashfs"
```

The script uploads the file, downloads it again, compares SHA-256 of both, deletes the
remote object, and reports:

```text
LOCAL_SHA256=...
RETURNED_SHA256=...
ROUND_TRIP_VERIFIED=true
REMOTE_CLEANED=true
```

**Checksums are the point.** A transfer that reported no error is not the same as a
transfer that delivered the bytes, and object storage over a network gives you more ways
to get a truncated or partial object than a local copy does. Verify before you delete the
source, never after.

## Ordinary transfers

Once verified, the everyday commands are plain rclone:

```bash
# Package first, then upload. One object beats fifty thousand.
rclone copy "$TUTORIAL_ROOT/source.squashfs" lumi-46XXXXXXX-private:data-aware-ai/

# Bring it back to scratch for a job.
rclone copy lumi-46XXXXXXX-private:data-aware-ai/source.squashfs "$TUTORIAL_ROOT/incoming/"

# Check what is there, and what it is costing.
rclone ls   lumi-46XXXXXXX-private:data-aware-ai/
rclone size lumi-46XXXXXXX-private:data-aware-ai/

# Remove a staged copy you no longer need. Buckets count against the project.
rclone delete lumi-46XXXXXXX-private:data-aware-ai/source.squashfs
```

`rclone --progress --stats=30s` is worth using for anything large, and `rclone check`
compares a local tree against a remote prefix without re-downloading it.

## Cleaning up

Staged copies are easy to forget and they consume the project's allocation until removed.
When a campaign ends:

1. Confirm the results you want to keep exist somewhere that is not scratch and not a
   single LUMI-O bucket.
2. Verify checksums of anything you are about to rely on.
3. Delete the intermediates.

The tutorial's `--keep-remote` flag exists so you can leave an object in place
deliberately; the default is to clean up, because an accidental accumulation of staged
datasets is the common failure here.
