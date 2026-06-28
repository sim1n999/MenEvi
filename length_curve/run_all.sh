#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
source "${ROOT}/common/common.sh"

LENGTHS="${LENGTHS:-32k 64k 128k 256k}"
RUN_TAG="${RUN_TAG:-default}"
RESULT_ROOT="${EXP_ROOT}/results/${RUN_TAG}"
mkdir -p "${RESULT_ROOT}"

for length in ${LENGTHS}; do
  dataset="$(dataset_path "${length}")"
  assert_full_dataset "${dataset}"
  ensure_core_assets "${length}"
  assets="${SHAREVISUAL_PACKET_ROOT}/assets_${length}_full"
  out="${RESULT_ROOT}/${length}"
  run_typed_base "${dataset}" "${assets}" "${out}/p0_typed_runtime"
  run_visual_gate "${dataset}" "${assets}/kg_memory/graphs" \
    "${out}/p0_typed_runtime/predictions.json" "${out}/p1_full_system"
  compare_predictions "${dataset}" "${out}/p0_typed_runtime/predictions.json" \
    "${out}/p1_full_system/full_predictions.json" "${out}/p1_vs_p0"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" "${EXP_ROOT}/scripts/summarize_length_curve.py" \
    --results-root "${RESULT_ROOT}" --output "${RESULT_ROOT}/paper_length_curve.csv"
fi
echo "Length-curve evaluation complete: ${RESULT_ROOT}"


