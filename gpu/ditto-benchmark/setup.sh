#!/usr/bin/env bash
# VikataKavi · Anvaya AI Lab — Ditto talking-head benchmark: one-time setup
# Target: Ubuntu 20.04/22.04 GPU VM with an NVIDIA driver already installed
# (E2E Networks / Yotta / AWS / any cloud "GPU + CUDA" image).
set -euo pipefail

DITTO_DIR="${DITTO_DIR:-$HOME/ditto-talkinghead}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

echo "== 1/5 GPU check =="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
echo "compute capability: $CC"

echo "== 2/5 System packages =="
if command -v apt-get >/dev/null; then
  sudo apt-get update -y && sudo apt-get install -y git git-lfs ffmpeg wget
fi
git lfs install

echo "== 3/5 Miniconda =="
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
# Newer conda asks to accept the Anaconda channel terms non-interactively.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "== 4/5 Ditto code + Python environment (10-15 min first time) =="
[ -d "$DITTO_DIR/.git" ] || git clone --depth 1 https://github.com/antgroup/ditto-talkinghead "$DITTO_DIR"
if conda env list | grep -qE '^ditto\s'; then
  echo "conda env 'ditto' already exists"
else
  conda env create -f "$DITTO_DIR/environment.yaml" -n ditto
fi
conda activate ditto
pip install -q requests

echo "== 5/5 Model checkpoints (several GB) =="
if [ ! -d "$DITTO_DIR/checkpoints/ditto_cfg" ]; then
  git clone https://huggingface.co/digital-avatar/ditto-talkinghead "$DITTO_DIR/checkpoints"
fi

MAJOR=${CC%%.*}
if [ "$MAJOR" -lt 8 ]; then
  echo "GPU is older than Ampere (compute $CC): building TensorRT engines for it (~10 min)…"
  (cd "$DITTO_DIR" && python scripts/cvt_onnx_to_trt.py --onnx_dir ./checkpoints/ditto_onnx --trt_dir ./checkpoints/ditto_trt_custom)
  echo "TRT_DIR=$DITTO_DIR/checkpoints/ditto_trt_custom" > "$HOME/.ditto_bench_env"
else
  echo "TRT_DIR=$DITTO_DIR/checkpoints/ditto_trt_Ampere_Plus" > "$HOME/.ditto_bench_env"
fi

echo
echo "Setup complete. Next:"
echo "  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate ditto"
echo "  export SARVAM_API_KEY=...   # your key, typed here only"
echo "  python make_test_audio.py && python bench.py"
