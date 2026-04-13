#!/bin/bash

# PBS submission script to report the status of the nuw0waCDM training dataset.
# This script prints how many samples are filled in each quantity after all
# shards have been processed.  The init and run-array jobs must complete
# before you submit this job.

#PBS -N nuw0wa_status
#PBS -o /lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_status.$PBS_JOBID.out
#PBS -e /lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_status.$PBS_JOBID.err
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=02:00:00
#PBS -M nhat.minh.nguyen@ipmu.jp
#PBS -m ae
#PBS -q mini
#PBS -V

set -euo pipefail

# Set a default environment for the status job.  Override this when
# submitting if you use a different conda environment.  Avoid
# referencing a non‑existent environment name like "carpile".
ENV_NAME="${ENV_NAME:-cosmopower}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"
OUTDIR="${OUTDIR:-}"

source ~/.bashrc
micromamba activate "${ENV_NAME}"

cd "$PBS_O_WORKDIR"

python -u generate_nuw0wa_training_data.py status "$CFG" ${OUTDIR:+--outdir "$OUTDIR"}