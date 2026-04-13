#!/bin/bash
#SBATCH --job-name=nuw0wa_status
#SBATCH --output=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_status.out.%j
#SBATCH --error=/lustre/work/n2minh/std/cosmopower/nuw0wa/nuw0wa_status.err.%j
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08

# Status job for the nuw0waCDM training dataset.
#
# After all shards have been generated, this job prints how many
# samples are filled per quantity and verifies that the dataset is
# complete.  Pass CFG and OUTDIR via --export when submitting.

set -euo pipefail

ENV_NAME="${ENV_NAME:-carpile}"
CFG="${CFG:?Set CFG environment variable to the YAML configuration file.}"
OUTDIR="${OUTDIR:-}"

micromamba activate "${ENV_NAME}"

cd "${SLURM_SUBMIT_DIR}"

python -u generate_nuw0wa_training_data.py status "$CFG" ${OUTDIR:+--outdir "$OUTDIR"}