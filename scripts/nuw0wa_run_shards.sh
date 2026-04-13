#!/bin/bash
#SBATCH --job-name=nuw0wa_run
#SBATCH --output=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_run.out.%A_%a
#SBATCH --error=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_run.err.%A_%a
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --array=0-9

# Shard execution job for the nuw0waCDM training dataset.
#
# Each array task calls generate_nuw0wa_training_data.py run-shard on a
# single shard of the training set.  The array range should match
# generator.n_shards-1 in the YAML (e.g. --array=0-14 for CMB or
# --array=0-19 for Pk).  The initialization step must be completed
# before this job is launched.
#
# Usage:
#   sbatch --array=0-(N_SHARDS-1) \
#          --export=CFG=path/to/package.yaml,OUTDIR=outdir,ENV_NAME=conda_env \
#          nuw0wa_run_shards.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-carpile}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"
OUTDIR="${OUTDIR:-}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

micromamba activate "${ENV_NAME}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

cd "${SLURM_SUBMIT_DIR}"

python -u generate_nuw0wa_training_data.py run-shard "$CFG" \
  ${OUTDIR:+--outdir "$OUTDIR"} \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --resume