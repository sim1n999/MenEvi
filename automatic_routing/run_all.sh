#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
source "${ROOT}/common/common.sh"

METHODS="${METHODS:-gold llm_zero_shot rules universal_c}"
[[ " ${METHODS} " == *" gold "* ]] || { echo "METHODS must include gold" >&2; exit 2; }
RUN_TAG="${RUN_TAG:-default}"
DATASET="$(dataset_path 32k)"
OUT="${EXP_ROOT}/results/${RUN_TAG}"
assert_full_dataset "${DATASET}"
ensure_core_assets 32k
CORE="${SHAREVISUAL_PACKET_ROOT}/assets_32k_full"
mkdir -p "${OUT}"

for method in ${METHODS}; do
  method_out="${OUT}/${method}"
  mkdir -p "${method_out}"
  router_args=()
  if [[ "${method}" == "llm_zero_shot" ]]; then router_args+=(--model "${TEXT_MODEL}" "${qargs[@]}"); fi
  run_step "route-manifest-${method}" "${method_out}/.route_manifest.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/build_route_manifest.py" --dataset "${DATASET}" \
    --method "${method}" --output "${method_out}/route_manifest.json" "${router_args[@]}"
  run_step "routed-dataset-${method}" "${method_out}/.routed_dataset.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/apply_route_manifest.py" --dataset "${DATASET}" \
    --manifest "${method_out}/route_manifest.json" --output "${method_out}/routed_dataset.json"
  prepare_packets "${method_out}/routed_dataset.json" \
    "${CORE}/retrieval_budget120/retrieved_subgraphs" "${method_out}/assets"
  run_typed_base "${method_out}/routed_dataset.json" "${method_out}/assets" "${method_out}/base"
  run_visual_gate "${method_out}/routed_dataset.json" "${CORE}/kg_memory/graphs" \
    "${method_out}/base/predictions.json" "${method_out}/full"
done

for method in ${METHODS}; do
  [[ "${method}" == "gold" ]] && continue
  compare_predictions "${DATASET}" "${OUT}/gold/full/full_predictions.json" \
    "${OUT}/${method}/full/full_predictions.json" "${OUT}/comparisons/${method}_vs_gold"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" "${EXP_ROOT}/scripts/summarize_routes.py" --results-root "${OUT}" \
    --output "${OUT}/paper_automatic_routing.csv"
fi
echo "Automatic routing evaluation complete: ${OUT}"

