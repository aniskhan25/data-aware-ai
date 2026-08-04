"""Run context: where and when a measurement happened.

Every run summary records this so that a result can be traced back to a machine,
a job, and a source revision.
"""

from __future__ import annotations

import os
import platform
import resource
import socket
import subprocess
from datetime import datetime, timezone


def timestamp_utc() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hostname() -> str:
    return socket.gethostname()


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> str:
    """Short commit of the working tree, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    commit = out.stdout.strip()
    return commit or "unknown"


def slurm_context() -> dict[str, str]:
    """Slurm allocation variables that are useful when interpreting a result.

    Missing variables are reported as empty strings rather than omitted, so that
    summaries from Slurm and from a laptop have the same shape.
    """
    names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
    )
    return {name: os.environ.get(name, "") for name in names}


def cpus_available() -> int:
    """Logical CPUs this process may actually use.

    Prefers the affinity mask, because that is what Slurm restricts. Falls back to the
    machine CPU count, which overstates the allocation inside a job.

    These are *logical* CPUs, not cores. On LUMI, simultaneous multithreading means an
    allocation of ``--cpus-per-task=7`` reports 14 here. Worker-count advice is
    measured against this number, so a ladder rung that looks like "one worker per
    core" is really one per two hardware threads.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def cpu_seconds(include_children: bool = True) -> tuple[float, float]:
    """CPU time consumed so far as ``(user, system)`` seconds.

    Children are included because DataLoader workers are separate processes, and it
    is precisely their CPU use that worker tuning is about. Unlike peak RSS, rusage
    *sums* CPU time over reaped children, so this total is meaningful.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    user, system = usage.ru_utime, usage.ru_stime
    if include_children:
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        user += children.ru_utime
        system += children.ru_stime
    return user, system


def context_switches(include_children: bool = True) -> tuple[int, int]:
    """``(voluntary, involuntary)`` context switches so far.

    Involuntary switches rising sharply is the signature of oversubscription: more
    runnable processes than cores, so the scheduler preempts them.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    voluntary, involuntary = usage.ru_nvcsw, usage.ru_nivcsw
    if include_children:
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        voluntary += children.ru_nvcsw
        involuntary += children.ru_nivcsw
    return voluntary, involuntary


def child_process_count() -> int:
    """Direct child processes of this one, or ``-1`` where it cannot be determined.

    Used to confirm that the worker count actually in use matches the configured
    one. Reads ``/proc``, so it returns -1 on platforms without it rather than
    guessing.
    """
    proc = "/proc"
    if not os.path.isdir(proc):
        return -1
    own = os.getpid()
    count = 0
    try:
        for entry in os.listdir(proc):
            if not entry.isdigit():
                continue
            try:
                with open(f"{proc}/{entry}/stat") as handle:
                    fields = handle.read().rsplit(")", 1)[-1].split()
                # After the comm field: state, ppid, ...
                if len(fields) >= 2 and int(fields[1]) == own:
                    count += 1
            except (OSError, ValueError):
                continue
    except OSError:
        return -1
    return count


def peak_memory_bytes(include_children: bool = True) -> int:
    """Peak resident set size of the largest single process, in bytes.

    ``ru_maxrss`` is reported in kibibytes on Linux and in bytes on macOS, so the
    unit is normalised here. ``RUSAGE_CHILDREN`` reports the peak of the largest
    reaped child, not the sum over children, so this value is the high-water mark
    of one process rather than total pipeline memory. Read it as "did any worker
    grow unexpectedly large", not as "how much memory did the job use".
    """
    scale = 1 if platform.system() == "Darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if include_children:
        peak = max(peak, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return int(peak) * scale
