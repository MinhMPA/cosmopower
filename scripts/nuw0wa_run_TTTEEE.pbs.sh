#!/bin/bash

# PBS submission script for generating each shard of the CMB TT/TE/EE/BB/pp dataset
# in the nuw0waCDM training campaign.  This job array will run 15 tasks,
# one per shard (indexed 0–14), using 8 CPUs and 160 GB per task.  Run the
# corresponding init job before launching this array.

#PBS -N nuw0wa_run_TTTEEE
#PBS -o /lustre/work/n2minh/std/cosmopower/nuw0wa/TTTEEE/nuw0wa_run_TTTEEE.$PBS_JOBID.$PBS_ARRAY_INDEX.out
#PBS -e /lustre/work/n2minh/std/cosmopower/nuw0wa/TTTEEE/nuw0wa_run_TTTEEE.$PBS_JOBID.$PBS_ARRAY_INDEX.err
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=48:00:00
#PBS -J 0-14%6
#PBS -M nhat.minh.nguyen@ipmu.jp
#PBS -m ae
#PBS -q mini
#PBS -V

set -euo pipefail

# Set a sensible default environment.  Override via ENV_NAME when
# submitting.  Do not use a non‑existent name like "carpile".
ENV_NAME="${ENV_NAME:-cosmopower}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"  # e.g. /home/n2minh/cosmopower/yaml/nuw0wa_camb_TTTEEE.yaml
OUTDIR="${OUTDIR:-/lustre/work/n2minh/cosmopower/training_sets/nuw0wa/TTTEEE}"
THREADS="${THREADS:-8}"

source ~/.bashrc
micromamba activate "${ENV_NAME}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

cd "$PBS_O_WORKDIR"

python -u generate_nuw0wa_training_data.py run-shard "$CFG" \
  --outdir "$OUTDIR" \
  --shard "$PBS_ARRAY_INDEX" \
  --resume