#!/usr/bin/env bash
# Install a CUDA-enabled PyTorch build for Jetson Orin.
# PyPI's aarch64 torch wheels are CPU-only, so NVIDIA's Jetson builds are used instead.
# Wheel locations change between JetPack releases; if a download fails, see
# https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
set -euo pipefail
PY="${1:-python3}"
L4T=$(head -1 /etc/nv_tegra_release 2>/dev/null | sed -n 's/^# R\([0-9]*\).*/\1/p')
echo "L4T major release: ${L4T:-unknown}"
case "$L4T" in
  36)  # JetPack 6.x (Ubuntu 22.04, Python 3.10, CUDA 12.6)
    "$PY" -m pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch ;;
  35)  # JetPack 5.1.x (Ubuntu 20.04, Python 3.8)
    sudo apt-get install -y -qq libopenmpi-dev libomp-dev >/dev/null
    "$PY" -m pip install --no-cache-dir \
      https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl ;;
  *)
    echo "Unknown JetPack - installing the generic wheel (CPU only on aarch64)."
    "$PY" -m pip install torch ;;
esac
"$PY" -m pip install -q onnx || true
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
