#!/bin/bash
# Per-rank entry point, invoked by srun from jobs/run_distributed_loader.sh.
#
# This exists so the launcher does not have to embed a quoted shell command inside
# srun: every rank sources jobs/common.sh here and gets the same container handling as
# every other job in the repository. Not meant to be run directly.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# shellcheck source=/dev/null
source jobs/common.sh

run_python scripts/distributed_loader.py "$@"
