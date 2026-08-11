# Site configuration. Copy to env.sh and edit. env.sh is deliberately gitignored:
# it holds your project allocation, which must never be committed.
#
#   cp env.example.sh env.sh
#   $EDITOR env.sh
#
# Job scripts source env.sh automatically when it exists.

# Your LUMI project. Find it with: groups | tr ' ' '\n' | grep project_
export LUMI_PROJECT=project_XXXXXXXXX

# Slurm bills to this account. Setting SBATCH_ACCOUNT means you do not have to
# pass --account on every sbatch command.
export SBATCH_ACCOUNT="$LUMI_PROJECT"

# Where the tutorial keeps datasets, packaged images, and results.
#
# Under $USER, not at the top of the project directory: a LUMI project is shared,
# and two people running this tutorial would otherwise write to the same paths and
# overwrite each other's datasets and results.
export TUTORIAL_ROOT="/scratch/$LUMI_PROJECT/$USER/data-aware-ai"

# Project flash, used only by the optional flash placement in step 5. Comment this
# out if your project has no flash allocation.
export TUTORIAL_FLASH_ROOT="/flash/$LUMI_PROJECT/$USER/data-aware-ai"

# Optional: a container to run the Python workload inside. Leave unset to use
# whatever python is already on PATH (a module, a venv, or a conda environment).
#
# This LUMI AI Factory image carries pyarrow, datasets, and h5py, so all three
# optional format adapters run without installing anything. The sif-images PyTorch
# containers carry the first two but not h5py.
#
# Pinned to a version, not to lumi-multitorch-latest.sif: that symlink moves when a
# new image is published, so a run that used it is not reproducible. This is the
# image the measured results in docs/reference-results.md came from.
# export TUTORIAL_CONTAINER="/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260731_122833/lumi-multitorch-full-u24r70f21m50t210-20260731_122833.sif"
# export TUTORIAL_CONTAINER="/scratch/$LUMI_PROJECT/$USER/containers/pytorch.sif"

# Extra bind mounts for the container, comma separated. These are added to the
# repository bind, not a replacement for it.
#
# Anything the job reads or writes must appear here, or it will be invisible inside
# the container. That includes project flash if you run the flash placement - its
# absence shows up as a confusing "file not found" for a path that plainly exists.
export TUTORIAL_CONTAINER_BINDS="/scratch/$LUMI_PROJECT/$USER,/flash/$LUMI_PROJECT/$USER"
#
# To read a SquashFS image through a container bind (step 3), point one of these
# at the image with image-src, matching dataset.root in the config:
# export TUTORIAL_CONTAINER_BINDS="$TUTORIAL_ROOT/source.squashfs:$TUTORIAL_ROOT/mnt/source:image-src=/"
