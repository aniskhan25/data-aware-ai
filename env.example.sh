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
# Project scratch is the documented location for job input and output I/O.
export TUTORIAL_ROOT="/scratch/$LUMI_PROJECT/data-aware-ai"

# Project flash, used only by the optional Part V flash experiment. Comment this
# out if your project has no flash allocation.
export TUTORIAL_FLASH_ROOT="/flash/$LUMI_PROJECT/data-aware-ai"

# Optional: a container to run the Python workload inside. Leave unset to use
# whatever python is already on PATH (a module, a venv, or a conda environment).
# export TUTORIAL_CONTAINER="/scratch/$LUMI_PROJECT/containers/pytorch.sif"

# Optional: extra bind mounts for the container, comma separated.
# export TUTORIAL_CONTAINER_BINDS="/scratch/$LUMI_PROJECT,/flash/$LUMI_PROJECT"
