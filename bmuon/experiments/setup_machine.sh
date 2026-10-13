#!/usr/bin/env bash
# Install everything needed to run BMuon on this machine, in /tmp/$USER (local disk).
#   bash bmuon/experiments/setup_machine.sh
# Idempotent: rerun it after a reboot that wiped /tmp. Takes a few minutes.
#
# The lab machines have no full CUDA toolkit, which the OpenEquivariance JAX extension needs
# to compile. We assemble a minimal CUDA 12 root (headers + libcudart + libnvrtc) from the
# NVIDIA pip wheels already installed with JAX, plus the driver's libcuda.
set -euo pipefail
USER="${USER:-$(id -un)}"
cd "$(dirname "$0")/../.."

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/$USER/uv-python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/nequix-venv}"
CUDA_ROOT="/tmp/$USER/cuda12"
mkdir -p "/tmp/$USER"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== python environment ($UV_PROJECT_ENVIRONMENT)"
uv sync --extra oeq
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
uv pip install "nvidia-cuda-nvrtc-cu12>=12.8,<13" "nvidia-cuda-nvcc-cu12>=12.8,<13"

echo "== minimal CUDA 12 root ($CUDA_ROOT)"
SITE="$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))')/nvidia"
rm -rf "$CUDA_ROOT"
mkdir -p "$CUDA_ROOT/include" "$CUDA_ROOT/lib64"
for d in cuda_runtime cuda_nvrtc cuda_nvcc; do
  if [ -d "$SITE/$d/include" ]; then
    for f in "$SITE/$d/include"/*; do ln -sfn "$f" "$CUDA_ROOT/include/"; done
  fi
done
ln -sf "$SITE/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so"
ln -sf "$SITE/cuda_runtime/lib/libcudart.so.12" "$CUDA_ROOT/lib64/libcudart.so.12"
ln -sf "$SITE/cuda_nvrtc/lib/libnvrtc.so.12" "$CUDA_ROOT/lib64/libnvrtc.so"
ln -sf "$SITE/cuda_nvrtc/lib/libnvrtc.so.12" "$CUDA_ROOT/lib64/libnvrtc.so.12"
for f in "$SITE"/cuda_nvrtc/lib/libnvrtc-builtins*; do ln -sf "$f" "$CUDA_ROOT/lib64/"; done
DRV="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so(\.1)? /{print $NF; exit}')"
[ -z "$DRV" ] && DRV="$(ls /usr/lib64/libcuda.so* /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null | head -1 || true)"
if [ -z "$DRV" ]; then echo "NVIDIA driver library libcuda not found" >&2; exit 1; fi
ln -sf "$DRV" "$CUDA_ROOT/lib64/libcuda.so"
echo '{"cuda": {"version": "12.8.0"}}' > "$CUDA_ROOT/version.json"

echo "== OpenEquivariance JAX extension"
CUDAToolkit_ROOT="$CUDA_ROOT" CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDA_ROOT" \
  uv pip install --no-build-isolation --reinstall "openequivariance_extjax==0.7.0"

echo "== checks"
python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'JAX does not see the GPU'"
python -c "import openequivariance, openequivariance_extjax; print('OpenEquivariance OK')"
touch "$UV_PROJECT_ENVIRONMENT/.bmuon_ready"
echo "== ready: source $UV_PROJECT_ENVIRONMENT/bin/activate"
