#!/bin/bash
# Run one measured data-loading experiment.
#
#   sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml
#   sbatch jobs/run_loader.sh configs/baseline/loose_files.yaml loader.num_workers=2
#
# Extra arguments are passed through as --set overrides, so one configuration can
# drive a ladder of runs without duplicating files.
#
# --cpus-per-task=7 mirrors the CPU share of a single LUMI-G GCD, which is the
# allocation the worker tuning in Part IV is reasoning about. This benchmark uses
# no GPU, so it runs on the CPU partition; Part VI moves to a GPU node.

#SBATCH --job-name=daai-loader
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "$(dirname "$0")/common.sh"

CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
    echo "usage: sbatch jobs/run_loader.sh CONFIG [section.option=value ...]" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR configuration not found: $CONFIG" >&2
    exit 2
fi
shift

require_vars TUTORIAL_ROOT
report_allocation

# Turn trailing key=value arguments into repeated --set flags.
OVERRIDES=()
for assignment in "$@"; do
    if [[ "$assignment" != *=* ]]; then
        echo "ERROR override must look like section.option=value, got '$assignment'" >&2
        exit 2
    fi
    OVERRIDES+=(--set "$assignment")
done

echo "CONFIG=$CONFIG"
echo "OVERRIDES=${OVERRIDES[*]:-none}"

# The +expansion guard keeps an empty array from tripping 'set -u' on older bash.
run_python scripts/benchmark_loader.py --config "$CONFIG" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}

echo "DONE ${SLURM_JOB_ID:-local}"
