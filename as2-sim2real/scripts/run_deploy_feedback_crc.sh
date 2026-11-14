#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unset PYTHONHOME LD_PRELOAD CYCLONEDDS_URI CYCLONEDDS_PYTHON_NO_IMPORT_LIBS
export PYTHONPATH="$root/src:$root/runtime/python310:$root/vendor/unitree_sdk2_python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CYCLONEDDS_HOME="$root/runtime" LD_LIBRARY_PATH="$root/runtime/lib"
export XDG_CACHE_HOME="$root/cache" TMPDIR="$root/cache/tmp"
mkdir -p "$TMPDIR"
cd "$root"
exec "$root/.venv-sdk/bin/python" -B -s -m as2_motion_imitation.deployment.feedback_crc "$@"
