#!/bin/bash
# Instructor setup, run once inside a gh compute-node session (not a login node):
#
#   idev -p gh -A TRA25008 -r inspireGPU -N 1 -t 02:00:00
#   bash scripts/setup_shared.sh /path/to/shared
#
# Builds the shared Python environment, stages the data and model, and opens read
# access so every participant uses the same copy instead of downloading their own.

set -euo pipefail

SHARED=${1:?usage: setup_shared.sh /path/to/shared}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$SHARED"

module reset
module load gcc cuda python3

if [ ! -x "$SHARED/venv/bin/python" ]; then
  python3 -m venv "$SHARED/venv"
  "$SHARED/venv/bin/pip" install --upgrade pip
  "$SHARED/venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu129
  "$SHARED/venv/bin/pip" install -r "$REPO/requirements.txt"
fi

"$SHARED/venv/bin/python" "$REPO/scripts/stage.py" --shared "$SHARED"

# Quick end-to-end check: 600 complaints, one GPU.
"$SHARED/venv/bin/python" "$REPO/scripts/train.py" --shared "$SHARED" \
  --out "$SCRATCH/inspire/setup-check" --limit 600

# Read access for everyone, plus traverse on each parent folder up to $WORK or $SCRATCH.
chmod -R a+rX "$SHARED"
d=$SHARED
while [ "$d" != "/" ] && [ "$d" != "$(dirname "$WORK")" ] && [ "$d" != "$(dirname "$SCRATCH")" ]; do
  chmod a+x "$d" 2>/dev/null || true
  d=$(dirname "$d")
done
echo "shared folder ready: $SHARED"
