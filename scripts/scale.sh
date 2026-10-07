#!/bin/bash
# Run train.py on the first NODES nodes of the session you are already in.
#
#   bash scripts/scale.sh 1 --limit 60000
#   bash scripts/scale.sh 2 --limit 60000
#   bash scripts/scale.sh 4 --limit 60000
#
# Each node has one GPU, so NODES is also the number of GPUs. Every node gets the same
# script; torchrun connects them so they split the complaints between them and share
# what they learn after every step. Run from a session that holds at least NODES nodes.
# Needs SHARED and PY set (the notebook sets both).

set -euo pipefail

NODES=${1:?usage: scale.sh NODES [train.py args]}
shift

: "${SHARED:?set SHARED to the shared workshop folder}"
: "${PY:?set PY to the workshop python}"
: "${SLURM_JOB_NODELIST:?not inside a compute-node session}"

REPO=$(cd "$(dirname "$0")/.." && pwd)
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( NODES > ${#HOSTS[@]} )); then
  echo "This session has ${#HOSTS[@]} node(s); it cannot run on $NODES."
  exit 1
fi

USE=$(IFS=,; echo "${HOSTS[*]:0:NODES}")
MASTER=${HOSTS[0]}
PORT=$((29500 + RANDOM % 1000))
OUT=${OUT:-$SCRATCH/inspire/scale-${NODES}node}
TORCHRUN=$(dirname "$PY")/torchrun

echo "nodes: $USE"
echo "results -> $OUT"

# --overlap lets this run alongside the Jupyter server already holding the first node.
srun --overlap -N "$NODES" -n "$NODES" --ntasks-per-node=1 --nodelist="$USE" \
  "$TORCHRUN" --nnodes="$NODES" --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER:$PORT" --rdzv_id="$SLURM_JOB_ID-$NODES" \
    "$REPO/scripts/train.py" --shared "$SHARED" --out "$OUT" "$@"
