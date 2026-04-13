#!/bin/bash

# PBS submission script for generating each shard of the Pk dataset in
# the nuw0waCDM training campaign.  This job array will run 20 tasks,
# one per shard (indexed 0–19), using 8 CPUs and 160 GB per task.  Be sure
# to run the init job first so that the HDF5 files are pre-allocated.

#PBS -N nuw0wa_run_PK
#PBS -o /lustre/work/n2minh/std/cosmopower/nuw0wa/PK/nuw0wa_run_PK.$PBS_JOBID.$PBS_ARRAY_INDEX.out
#PBS -e /lustre/work/n2minh/std/cosmopower/nuw0wa/PK/nuw0wa_run_PK.$PBS_JOBID.$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=48:00:00
#PBS -J 0-19%6
#PBS -M nhat.minh.nguyen@ipmu.jp
#PBS -m ae
#PBS -q mini
#PBS -V

set -euo pipefail

# Set a reasonable default environment.  Override via ENV_NAME when
# submitting.  Avoid referencing a non‑existent environment like "carpile".
ENV_NAME="${ENV_NAME:-cosmopower}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"  # e.g. /home/n2minh/cosmopower/yaml/nuw0wa_camb_PK.yaml
OUTDIR="${OUTDIR:-/lustre/work/n2minh/cosmopower/training_sets/nuw0wa/PK}"
THREADS="${THREADS:-8}"

source ~/.bashrc
micromamba activate "${ENV_NAME}"

# Set thread affinity for CAMB and BLAS libraries
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

cd "$PBS_O_WORKDIR"

# Use PBS_ARRAY_INDEX to select the shard to compute
python -u generate_nuw0wa_training_data.py run-shard "$CFG" \
  --outdir "$OUTDIR" \
  --shard "$PBS_ARRAY_INDEX" \
  --resume