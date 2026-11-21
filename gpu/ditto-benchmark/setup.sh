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
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
missing=""
for c in git git-lfs ffmpeg wget; do command -v "$c" >/dev/null || missing="$missing $c"; done
if [ -n "$missing" ] && command -v apt-get >/dev/null; then
  # Unrelated broken repos/packages on the machine must not stop us.
  $SUDO apt-get update -y || echo "warning: apt-get update reported errors; continuing"
  $SUDO apt-get install -y $missing || echo "warning: apt-get install reported errors; checking tools"
fi
for c in git git-lfs ffmpeg wget; do
  command -v "$c" >/dev/null || { echo "ERROR: '$c' is still missing. Install it and re-run."; exit 1; }
done
echo "git, git-lfs, ffmpeg, wget: OK"
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
# TensorRT 8.6.1 wheels now live on NVIDIA's own index, not PyPI.
export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"
if conda env list | grep -qE '^ditto\s'; then
  echo "conda env 'ditto' exists: updating it (repairs a previously failed install)"
  conda env update -n ditto -f "$DITTO_DIR/environment.yaml"
else
  conda env create -f "$DITTO_DIR/environment.yaml" -n ditto
fi
conda activate ditto
pip install -q requests

# TensorRT 8.6 needs cuDNN 8, while this PyTorch uses cuDNN 9. Keep cuDNN 8 in its own
# folder and put it on the library path whenever the 'ditto' env is activated.
CUDNN8_DIR="$HOME/cudnn8"
if [ ! -f "$CUDNN8_DIR/nvidia/cudnn/lib/libcudnn.so.8" ]; then
  pip install -q --no-deps --target "$CUDNN8_DIR" "nvidia-cudnn-cu12==8.9.7.29"
fi
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export LD_LIBRARY_PATH=\"$CUDNN8_DIR/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}\"" \
  > "$CONDA_PREFIX/etc/conda/activate.d/cudnn8.sh"
set +u; conda deactivate; conda activate ditto; set -u   # re-activate to apply the new path
python - <<'PY'
import torch, tensorrt, librosa
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("tensorrt", tensorrt.__version__, "| librosa", librosa.__version__)
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
PY

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
