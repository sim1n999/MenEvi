#!/usr/bin/env bash
set -euo pipefail

if [[ "${FORMAL_HOLDOUT_ACK:-}" != "YES" ]]; then
  echo "Refusing to touch the formal holdout. Set FORMAL_HOLDOUT_ACK=YES only after configuration freeze." >&2
  exit 3
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
TYPED_ROOT="${ROOT}/typed_evidence"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEXT_MODEL="${TEXT_MODEL:-}"
VISION_MODEL="${VISION_MODEL:-}"
[[ -n "${TEXT_MODEL}" ]] || { echo "ERROR: TEXT_MODEL is empty." >&2; exit 2; }
[[ -n "${VISION_MODEL}" ]] || { echo "ERROR: VISION_MODEL is empty." >&2; exit 2; }
FROZEN_CONFIG="${FROZEN_CONFIG:-${HOLDOUT_ROOT}/frozen_config.json}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:?Set HOLDOUT_DATASET to the materialized 594-question JSON}"
GRAPH_DIR="${GRAPH_DIR:?Set GRAPH_DIR to graphs for the 594-question holdout}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:?Set BASELINE_PREDICTIONS to the frozen baseline holdout predictions}"
IMAGES="${IMAGES:-${ROOT}/memlens_repro/data/memlens/release_images}"
RESULTS="${HOLDOUT_ROOT}/results/holdout_594"
MARKER="${RESULTS}/FORMAL_RUN.json"
PACKET_ROOT="${RESULTS}/typed_packets"
PACKETS="${PACKET_ROOT}/packets"
VISUAL_AUDIT="${RESULTS}/visual_prompt_audit"
VISUAL="${RESULTS}/visual_inspection"
SPECIALIST_AUDIT="${RESULTS}/specialist_prompt_audit"
SPECIALISTS="${RESULTS}/typed_specialists"
CANDIDATE_PREDICTIONS="${SPECIALISTS}/predictions.json"

for required in "${FROZEN_CONFIG}" "${HOLDOUT_DATASET}" "${GRAPH_DIR}" \
  "${BASELINE_PREDICTIONS}" "${IMAGES}" "${TEXT_MODEL}" "${VISION_MODEL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required formal-run path: ${required}" >&2
    exit 2
  fi
done

QUANT_ARGS=()
if [[ "${NO_4BIT:-0}" == "1" ]]; then
  QUANT_ARGS+=(--no-4bit)
fi
RESUME_ARGS=()
if [[ "${RESUME_FORMAL_RUN:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/freeze_config.py" verify \
  --frozen-config "${FROZEN_CONFIG}"
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/holdout_guard.py" begin \
  --dataset "${HOLDOUT_DATASET}" \
  --base "${BASELINE_PREDICTIONS}" \
  --frozen-config "${FROZEN_CONFIG}" \
  --marker "${MARKER}" \
  "${RESUME_ARGS[@]}"

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/build_typed_evidence_v2.py" \
  --dataset "${HOLDOUT_DATASET}" \
  --graph-dir "${GRAPH_DIR}" \
  --image-dir "${IMAGES}" \
  --output-dir "${PACKET_ROOT}"

"${PYTHON}" "${TYPED_ROOT}/scripts/run_visual_inspection.py" \
  --packet-dir "${PACKETS}" \
  --output-dir "${VISUAL_AUDIT}" \
  --top-k 3 \
  --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${VISUAL_AUDIT}/requests" \
  --report "${VISUAL_AUDIT}/leakage_audit.json"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${TYPED_ROOT}/scripts/run_visual_inspection.py" \
  --packet-dir "${PACKETS}" \
  --output-dir "${VISUAL}" \
  --model "${VISION_MODEL}" \
  --top-k 3 \
  "${QUANT_ARGS[@]}"

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py" \
  --dataset "${HOLDOUT_DATASET}" \
  --packet-dir "${PACKETS}" \
  --visual-observation-dir "${VISUAL}/observations" \
  --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md" \
  --output-dir "${SPECIALIST_AUDIT}" \
  --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${SPECIALIST_AUDIT}/prompts" \
  --report "${SPECIALIST_AUDIT}/leakage_audit.json"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py" \
  --dataset "${HOLDOUT_DATASET}" \
  --packet-dir "${PACKETS}" \
  --visual-observation-dir "${VISUAL}/observations" \
  --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md" \
  --output-dir "${SPECIALISTS}" \
  --model "${TEXT_MODEL}" \
  "${QUANT_ARGS[@]}"

run_condition() {
  local name="$1"
  shift
  local out_dir="${RESULTS}/${name}"
  "${PYTHON}" "${HOLDOUT_ROOT}/scripts/merge_with_base.py" \
    --dataset "${HOLDOUT_DATASET}" \
    --base "${BASELINE_PREDICTIONS}" \
    --candidate "${CANDIDATE_PREDICTIONS}" \
    --output "${out_dir}/predictions.json" \
    --require-all-targets \
    --include-subtypes "$@"
  "${PYTHON}" "${TYPED_ROOT}/scripts/compare_predictions.py" \
    --dataset "${HOLDOUT_DATASET}" \
    --baseline "${BASELINE_PREDICTIONS}" \
    --candidate "${out_dir}/predictions.json" \
    --output-json "${out_dir}/comparison.json" \
    --output-md "${out_dir}/COMPARISON.md"
}

run_condition holdout_arithmetic arithmetic
run_condition holdout_duration duration_comparison
run_condition holdout_visual entity previnfo
run_condition holdout_merged arithmetic duration_comparison entity previnfo

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/holdout_guard.py" finish \
  --marker "${MARKER}" \
  --result "${RESULTS}/holdout_merged/predictions.json" \
  --comparison "${RESULTS}/holdout_merged/comparison.json"

echo "Formal holdout run completed: ${RESULTS}/holdout_merged"



