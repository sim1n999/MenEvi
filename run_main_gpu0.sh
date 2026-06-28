#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_DEVICE="${GPU0_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export RUN_TAG="${GPU0_RUN_TAG:-gpu0_main}"
export EVIMEM_SHARED_PACKET_ROOT="${GPU0_SHARED_PACKET_ROOT:-${ROOT}/common/generated/gpu0_main}"
export CAPTION_CACHE="${GPU0_CAPTION_CACHE:-${EVIMEM_SHARED_PACKET_ROOT}/caption_cache_all_lengths.jsonl}"

bash "${ROOT}/length_curve/run_all.sh"
bash "${ROOT}/automatic_routing/run_all.sh"
bash "${ROOT}/component_ablations/run_all.sh"

echo "GPU0 tasks complete: length_curve, automatic_routing, component_ablations (${RUN_TAG})"

