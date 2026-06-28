#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
source "${ROOT}/common/common.sh"

RUN_TAG="${RUN_TAG:-default}"
VARIANTS="${VARIANTS:-no_edges no_state no_visual no_temporal}"
DATASET="$(dataset_path 32k)"
OUT="${EXP_ROOT}/results/${RUN_TAG}"
P_OUT="${ROOT}/length_curve/results/${RUN_TAG}/32k"
mkdir -p "${OUT}"
assert_full_dataset "${DATASET}"

LENGTHS=32k RUN_TAG="${RUN_TAG}" bash "${ROOT}/length_curve/run_all.sh"
CORE="${SHAREVISUAL_PACKET_ROOT}/assets_32k_full"

for variant in ${VARIANTS}; do
  variant_out="${OUT}/storage/${variant}"
  run_step "transform-${variant}" "${variant_out}/.transform.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/transform_graphs.py" \
    --input-dir "${CORE}/kg_memory/graphs" --output-dir "${variant_out}/graphs" \
    --variant "${variant}" --expected-count 789
  run_step "retrieve-${variant}" "${variant_out}/.retrieval.complete" \
    "${PYTHON}" "${ROOT}/kg_retrieval/scripts/retrieve_kg_subgraphs.py" \
    --input "${DATASET}" --graph-dir "${variant_out}/graphs" \
    --output-dir "${variant_out}/retrieval" --node-budget 120 "${margs[@]}"
  prepare_packets "${DATASET}" "${variant_out}/retrieval/retrieved_subgraphs" "${variant_out}/assets"
  run_typed_base "${DATASET}" "${variant_out}/assets" "${variant_out}/base"
  run_visual_gate "${DATASET}" "${variant_out}/graphs" \
    "${variant_out}/base/predictions.json" "${variant_out}/full"
  compare_predictions "${DATASET}" "${P_OUT}/p1_full_system/full_predictions.json" \
    "${variant_out}/full/full_predictions.json" "${OUT}/comparisons/storage_${variant}_vs_full"
done

BASE="${P_OUT}/p0_typed_runtime/predictions.json"
VISUAL="${P_OUT}/p1_full_system/visual_specialist/predictions.json"
OBS="${P_OUT}/p1_full_system/visual_inspection/observations"
for gate in always_replace reject_refusal preserve_support full_gate; do
  gate_out="${OUT}/gate/${gate}"
  flags=()
  [[ "${gate}" == "reject_refusal" || "${gate}" == "full_gate" ]] && flags+=(--reject-refusal)
  [[ "${gate}" == "preserve_support" || "${gate}" == "full_gate" ]] && flags+=(--preserve-supported-base)
  run_step "gate-ablation-${gate}" "${gate_out}/.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/apply_gate_variant.py" \
    --dataset "${DATASET}" --base "${BASE}" --visual-predictions "${VISUAL}" \
    --observation-dir "${OBS}" --output "${gate_out}/predictions.json" \
    --decisions-output "${gate_out}/decisions.json" "${flags[@]}"
  compare_predictions "${DATASET}" "${P_OUT}/p1_full_system/full_predictions.json" \
    "${gate_out}/predictions.json" "${OUT}/comparisons/gate_${gate}_vs_full"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" "${EXP_ROOT}/scripts/summarize_ablations.py" --comparison-dir "${OUT}/comparisons" \
    --output "${OUT}/paper_component_ablations.csv"
fi
echo "Component ablation evaluation complete: ${OUT}"


