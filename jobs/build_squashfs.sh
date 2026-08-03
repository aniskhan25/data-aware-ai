#!/bin/bash
# Package the dataset into one SquashFS image.
#
#   sbatch jobs/build_squashfs.sh
#
# Packaging is CPU and I/O work with no GPU involved, and it is a one-off: the image
# is read-only, so it only needs rebuilding when the dataset changes.
#
# Runs outside the container: mksquashfs is a host binary and is not present inside
# a PyTorch image.
#
# The default arguments store sample bytes uncompressed (-noD -noF), because the
# tutorial dataset is JPEG and PNG whose bytes are already compressed. Both flags are
# needed: small files become fragments, which -noD alone does not cover. Compressing them again would
# spend CPU on every read for almost no space saved.

#SBATCH --job-name=daai-build-squashfs
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fail before anything else, so a broken setup cannot report success.
set -euo pipefail

# Slurm copies the batch script into a spool directory, so $0 does not point into
# the repository. SLURM_SUBMIT_DIR is where sbatch was invoked, i.e. the repo root.
COMMON="${SLURM_SUBMIT_DIR:-$(dirname "$0")}/jobs/common.sh"
[[ -f "$COMMON" ]] || COMMON="$(dirname "$0")/common.sh"
if [[ ! -f "$COMMON" ]]; then
    echo "ERROR cannot find jobs/common.sh; submit from the repository root" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$COMMON"

require_vars TUTORIAL_ROOT
report_allocation

SOURCE_DIR="${1:-$TUTORIAL_ROOT/source}"
IMAGE_PATH="${2:-$TUTORIAL_ROOT/source.squashfs}"
MOUNT_POINT="$TUTORIAL_ROOT/mnt/source"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR dataset not found: $SOURCE_DIR" >&2
    echo "Generate it first: sbatch jobs/prepare_dataset.sh" >&2
    exit 2
fi

echo "SOURCE_DIR=$SOURCE_DIR"
echo "IMAGE_PATH=$IMAGE_PATH"

# mksquashfs is a host binary, absent from a PyTorch container, so this step runs
# outside the container. It needs no PyTorch.
run_host_python scripts/build_squashfs.py \
    --source "$SOURCE_DIR" \
    --image "$IMAGE_PATH" \
    --overwrite

mkdir -p "$MOUNT_POINT"
cat <<NEXT

The image is built. To measure it, make it readable at
  $MOUNT_POINT
which is what configs/baseline/squashfs.yaml expects (squashfs_mode: prebound).

Two ways, both documented at
https://docs.lumi-supercomputer.eu/storage/formats/FUSE/

  1. Bind it into your container at launch, which needs no privileges. Add to
     env.sh (this is added to the repository bind, not a replacement):
       export TUTORIAL_CONTAINER_BINDS="$IMAGE_PATH:$MOUNT_POINT:image-src=/"

  2. Let the loader mount it with squashfuse, by setting in the config:
       dataset:
         squashfs_mode: squashfuse
         image: $IMAGE_PATH

Then: sbatch jobs/run_loader.sh configs/baseline/squashfs.yaml
NEXT
