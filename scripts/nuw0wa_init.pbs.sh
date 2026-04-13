#!/bin/bash

# PBS submission script for initializing the nuw0waCDM training dataset.
# This script creates the Latin hypercube parameter table and pre‑allocates
# HDF5 files for all quantities defined in your YAML package.  Adjust the
# resource requests (walltime, ncpus, mem) to suit your cluster.

#PBS -N nuw0wa_init
#PBS -o /lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_init.$PBS_JOBID.out
#PBS -e /lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_init.$PBS_JOBID.err
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=04:00:00
#PBS -M nhat.minh.nguyen@ipmu.jp
#PBS -m ae
#PBS -q mini

# Inherit the submission environment (helps with modules, etc.)
#PBS -V

set -euo pipefail

# Environment and input variables
# Choose a sensible default conda environment.  Override this by
# passing ENV_NAME=<your_env> via the qsub -v option.  Without an
# explicit override, default to "cosmopower"; this avoids using
# a non‑existent environment name like "carpile".
ENV_NAME="${ENV_NAME:-cosmopower}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"
OUTDIR="${OUTDIR:-}"

# Activate your conda/mamba environment.  Adjust for your setup.
source ~/.bashrc
micromamba activate "${ENV_NAME}"

# Change into the directory from which qsub was invoked.
cd "$PBS_O_WORKDIR"

# Run the initialization command.  If OUTDIR is set, override the YAML path.
python -u generate_nuw0wa_training_data.py init "$CFG" ${OUTDIR:+--outdir "$OUTDIR"}