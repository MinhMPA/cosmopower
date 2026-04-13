#!/bin/bash
#SBATCH --job-name=nuw0wa_init
#SBATCH --output=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_init.out.%j
#SBATCH --error=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_init.err.%j
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08

# Initialization job for the nuw0waCDM training dataset.
#
# Usage:
#   sbatch --export=CFG=path/to/package.yaml,OUTDIR=outdir,ENV_NAME=conda_env nuw0wa_init.sh
#
# This job reads the CosmoPower YAML configuration specified in $CFG and
# pre‑allocates the parameter and spectra files in the output directory
# specified by $OUTDIR.  If $OUTDIR is not provided, the script uses
# the path field from the YAML.

set -euo pipefail

ENV_NAME="${ENV_NAME:-carpile}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"
OUTDIR="${OUTDIR:-}"

micromamba activate "${ENV_NAME}"

cd "${SLURM_SUBMIT_DIR}"

python -u generate_nuw0wa_training_data.py init "$CFG" ${OUTDIR:+--outdir "$OUTDIR"}