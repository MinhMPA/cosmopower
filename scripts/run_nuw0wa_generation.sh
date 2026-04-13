#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-nuw0wa_camb_package.yaml}"
OUTDIR="${2:-./mpa_2026_camb_nuw0waCDM}"

python generate_nuw0wa_training_data.py init "$CFG" --outdir "$OUTDIR"

# Sequential shard generation. On a cluster, launch one run-shard job per shard instead.
for shard in $(seq 0 9); do
  python generate_nuw0wa_training_data.py run-shard "$CFG" --outdir "$OUTDIR" --shard "$shard" --resume
done

python generate_nuw0wa_training_data.py status "$CFG" --outdir "$OUTDIR"
