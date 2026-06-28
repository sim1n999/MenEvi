#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${1:-memlens-qwen25}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env already exists: $ENV_NAME"
else
  if [[ "$ENV_NAME" == "memlens-qwen25" ]]; then
    conda env create -f "${REPRO_ROOT}/environment.yml"
  else
    conda env create -n "$ENV_NAME" -f "${REPRO_ROOT}/environment.yml"
  fi
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "$PYTORCH_CUDA_INDEX"
python -m pip install -r "${REPRO_ROOT}/requirements_pip.txt"

echo "Conda env ready: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"
