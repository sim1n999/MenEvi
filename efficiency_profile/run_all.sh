#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
RUN_TAG="${RUN_TAG:-efficiency_full}"
LENGTHS="${LENGTHS:-32k 64k 128k 256k}"
OUT="${EXP_ROOT}/results/${RUN_TAG}"
mkdir -p "${OUT}"

PROFILE=1 EVIMEM_SHARED_PACKET_ROOT="${OUT}/fresh_shared_assets" \
  CAPTION_CACHE="${OUT}/fresh_shared_assets/caption_cache.jsonl" \
  RUN_TAG="${RUN_TAG}" LENGTHS="${LENGTHS}" \
  bash "${ROOT}/length_curve/run_all.sh"

PROFILE=1 EVIMEM_SHARED_PACKET_ROOT="${OUT}/fresh_shared_assets" \
  CAPTION_CACHE="${OUT}/fresh_shared_assets/caption_cache.jsonl" \
  RUN_TAG="${RUN_TAG}" \
  bash "${ROOT}/strong_baselines/run_all.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Efficiency profiling dry run complete"
  exit 0
fi

"${PYTHON:-python}" "${EXP_ROOT}/scripts/summarize_efficiency.py" \
  --profile-root "${OUT}" \
  --length-results "${ROOT}/length_curve/results/${RUN_TAG}" \
  --baseline-results "${ROOT}/strong_baselines/results/${RUN_TAG}" \
  --output "${OUT}/paper_efficiency.csv" --storage-output "${OUT}/paper_storage.csv"
echo "Efficiency profiling complete: ${OUT}"


