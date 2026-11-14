#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unset PYTHONHOME LD_PRELOAD
export PYTHONPATH="$root/src" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$root/cache" TMPDIR="$root/cache/tmp"
mkdir -p "$TMPDIR"
cd "$root"
exec /usr/bin/python3 -B -s -m as2_motion_imitation.deployment.offline \
  --bundle "$root/bundle" "$@"
