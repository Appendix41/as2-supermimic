#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "${1:-}" in
  simulation) workspace="$repo/as2-motion-imitation" ;;
  sim2real) workspace="$repo/as2-sim2real" ;;
  *) printf 'Usage: bash tools/run.sh simulation|sim2real PYTHON_ARGS...\n' >&2
     exit 2 ;;
esac
shift
python_bin=$(command -v "${PYTHON_BIN:-python3}")
export PYTHONPATH="$workspace/src:$workspace/baseline/project/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export AS2_MOTION_ROOT="$workspace" AS2_OUTPUT_ROOT="$workspace/outputs"
export XDG_CACHE_HOME="$AS2_OUTPUT_ROOT/cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/huggingface"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$AS2_OUTPUT_ROOT/tmp"
export WANDB_MODE=disabled WANDB_DIR="$AS2_OUTPUT_ROOT/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb" WANDB_CONFIG_DIR="$XDG_CACHE_HOME/wandb-config"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd "$workspace"
exec "$python_bin" "$@"
