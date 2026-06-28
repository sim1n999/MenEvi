#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_DEVICE="${GPU1_DEVICE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export RUN_TAG="${GPU1_RUN_TAG:-gpu1_validation}"
Q_RESULT_ROOT="${ROOT}/efficiency_profile/results/${RUN_TAG}"
export EVIMEM_SHARED_PACKET_ROOT="${GPU1_SHARED_PACKET_ROOT:-${Q_RESULT_ROOT}/fresh_shared_assets}"
export CAPTION_CACHE="${GPU1_CAPTION_CACHE:-${EVIMEM_SHARED_PACKET_ROOT}/caption_cache.jsonl}"

bash "${ROOT}/efficiency_profile/run_all.sh"

echo "GPU1 tasks complete: efficiency_profile (${RUN_TAG})"

