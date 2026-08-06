#!/usr/bin/env python3
"""Verify a LUMI-O round trip: upload, download, compare checksums, clean up.

    python3 scripts/lumio_roundtrip.py \
        --remote lumi-46XXXXXXX-private \
        --bucket data-aware-ai \
        --file "$TUTORIAL_ROOT/source.squashfs"

LUMI-O is object storage, not a mounted filesystem. Objects are put and got; they are not
opened, seeked, or written in place. This script exercises the lifecycle the tutorial
teaches - stage in, verify, clean up - and checks that what came back is what went out.

CREDENTIALS ARE NEVER HANDLED HERE. Authentication lives in your rclone configuration,
created by `lumio-conf` on LUMI. This script does not read, print, accept, or store keys,
and nothing it writes contains them. Never commit an rclone config, and never pass keys
on a command line: arguments are visible to other users through the process table.

LUMI-O is a secondary copy, not an independent backup. LUMI provides no backup service for
any of its storage systems, nothing here versions or replicates anything, and LUMI-O data
is tied to the project lifecycle. Keep a copy outside LUMI of anything you cannot
regenerate.

Needs no PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: A remote whose name does not say "private" may well be the public endpoint, which
#: serves objects to anyone with the URL.
PRIVATE_MARKER = "private"


class LumioError(RuntimeError):
    """Raised when the environment is not ready or a transfer did not verify."""


def run_rclone(*arguments: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Invoke rclone, never echoing its configuration.

    Only the arguments given here are used: no environment is forwarded that could carry
    a key, and the output is returned rather than the config being inspected.
    """
    if shutil.which("rclone") is None:
        raise LumioError(
            "rclone was not found on PATH.\n"
            "On LUMI: module load lumio  (or use rclone from a container), then run "
            "lumio-conf to create your configuration."
        )
    return subprocess.run(
        ["rclone", *arguments], capture_output=True, text=True, timeout=timeout
    )


def configured_remotes() -> list[str]:
    result = run_rclone("listremotes", timeout=60)
    if result.returncode != 0:
        raise LumioError(f"rclone listremotes failed: {result.stderr.strip()}")
    return [line.strip().rstrip(":") for line in result.stdout.splitlines() if line.strip()]


def checksum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remote", help="configured rclone remote, e.g. lumi-46XXXXXXX-private")
    parser.add_argument("--bucket", default="data-aware-ai")
    parser.add_argument("--prefix", default="roundtrip")
    parser.add_argument("--file", type=Path, help="file to upload and verify")
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="leave the uploaded object in place instead of deleting it",
    )
    parser.add_argument(
        "--allow-public",
        action="store_true",
        help="proceed with a remote that does not look private (see the warning)",
    )
    parser.add_argument(
        "--list-remotes", action="store_true", help="list configured remotes and exit"
    )
    args = parser.parse_args(argv)

    try:
        remotes = configured_remotes()
    except LumioError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    if args.list_remotes or not args.remote:
        print("CONFIGURED_REMOTES=" + (",".join(remotes) if remotes else "none"))
        if not remotes:
            print(
                "\nNo remotes are configured. On LUMI, run lumio-conf and follow the\n"
                "browser prompt; it writes an rclone remote per project, in private and\n"
                "public variants. Use the private one.",
                file=sys.stderr,
            )
            return 2
        if not args.remote:
            print("\nRe-run with --remote <name> --file <path>.", file=sys.stderr)
            return 2

    if args.remote not in remotes:
        print(
            f"ERROR remote {args.remote!r} is not configured. Available: "
            f"{','.join(remotes) or 'none'}",
            file=sys.stderr,
        )
        return 2

    if PRIVATE_MARKER not in args.remote and not args.allow_public:
        print(
            f"ERROR remote {args.remote!r} does not look like a private endpoint.\n"
            "The public endpoint serves objects to anyone who has the URL, with no\n"
            "authentication. Use the private remote, or pass --allow-public if you\n"
            "genuinely intend to publish this data.",
            file=sys.stderr,
        )
        return 2

    if not args.file or not args.file.is_file():
        print(f"ERROR --file must point at an existing file, got {args.file}", file=sys.stderr)
        return 2

    target = f"{args.remote}:{args.bucket}/{args.prefix}/{args.file.name}"
    local_digest = checksum(args.file)
    print(f"LOCAL_FILE={args.file}")
    print(f"LOCAL_BYTES={args.file.stat().st_size}")
    print(f"LOCAL_SHA256={local_digest}")
    print(f"REMOTE_TARGET={target}", flush=True)

    upload = run_rclone("copyto", str(args.file), target, "--progress", "--stats=10s")
    if upload.returncode != 0:
        print(f"ERROR upload failed: {upload.stderr.strip()}", file=sys.stderr)
        return 3
    print("UPLOAD=ok", flush=True)

    with tempfile.TemporaryDirectory(prefix="daai-lumio-") as scratch:
        returned = Path(scratch) / args.file.name
        download = run_rclone("copyto", target, str(returned))
        if download.returncode != 0:
            print(f"ERROR download failed: {download.stderr.strip()}", file=sys.stderr)
            return 3

        returned_digest = checksum(returned)
        print(f"RETURNED_BYTES={returned.stat().st_size}")
        print(f"RETURNED_SHA256={returned_digest}")
        verified = returned_digest == local_digest
        print(f"ROUND_TRIP_VERIFIED={str(verified).lower()}")
        if not verified:
            print(
                "ERROR the object that came back differs from the one sent. Do not treat "
                "this transfer as complete.",
                file=sys.stderr,
            )
            return 4

    if not args.keep_remote:
        removed = run_rclone("deletefile", target)
        print(f"REMOTE_CLEANED={str(removed.returncode == 0).lower()}")
    else:
        print("REMOTE_CLEANED=false")
        print(f"NOTE the object remains at {target}")

    print(
        "\nLUMI-O holds a secondary copy, not an independent backup: LUMI provides no "
        "backup service for its storage, and these objects are tied to the project "
        "lifecycle."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
