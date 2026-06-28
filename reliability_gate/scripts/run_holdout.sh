#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_ROOT="${ROOT}/reliability_gate"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
TYPED_ROOT="${ROOT}/typed_evidence"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

: "${HOLDOUT_DATASET:?Set HOLDOUT_DATASET to the materialized 594-row JSON}"
: "${HOLDOUT_GRAPH_DIR:?Set HOLDOUT_GRAPH_DIR to the 594 holdout graph directory}"
: "${HOLDOUT_IMAGE_DIR:?Set HOLDOUT_IMAGE_DIR to the image root}"
: "${BASELINE_PREDICTIONS:?Set BASELINE_PREDICTIONS to the frozen 594-row baseline}"
if [[ "${FORMAL_HOLDOUT_ACK:-}" != "YES" ]]; then
  echo "Set FORMAL_HOLDOUT_ACK=YES only after the reliability-gate protocol is frozen and inputs are ready." >&2
  exit 2
fi

FROZEN="${GATE_ROOT}/frozen_config.json"
RESULTS="${GATE_ROOT}/results/formal_holdout_594"
MARKER="${RESULTS}/formal_run_marker.json"
PACKETS="${RESULTS}/visual_packets"
VISUAL_AUDIT="${RESULTS}/visual_prompt_audit"
VISUAL="${RESULTS}/visual_inspection"
SPECIALIST_AUDIT="${RESULTS}/specialist_prompt_audit"
SPECIALIST="${RESULTS}/visual_specialist"
GATED_PREDICTIONS="${RESULTS}/gated_predictions.json"
DECISIONS="${RESULTS}/gated_decisions.json"
COMPARISON="${RESULTS}/comparison.json"
BOOTSTRAP="${RESULTS}/paired_bootstrap.json"
POLICY="${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md"

"${PYTHON}" "${GATE_ROOT}/scripts/freeze_config.py" verify   --frozen-config "${FROZEN}"

TEXT_MODEL="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["models"]["text"])' "${FROZEN}")"
VISION_MODEL="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["models"]["vision"])' "${FROZEN}")"
LOAD_4BIT="$("${PYTHON}" -c 'import json,sys; print("1" if json.load(open(sys.argv[1], encoding="utf-8"))["generation"]["load_in_4bit"] else "0")' "${FROZEN}")"
QUANT_ARGS=()
if [[ "${LOAD_4BIT}" == "0" ]]; then
  QUANT_ARGS+=(--no-4bit)
fi
RESUME_ARGS=()
if [[ "${RESUME_FORMAL_GATE:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
for required in "${HOLDOUT_DATASET}" "${HOLDOUT_GRAPH_DIR}" \
  "${HOLDOUT_IMAGE_DIR}" "${BASELINE_PREDICTIONS}" "${TEXT_MODEL}" \
  "${VISION_MODEL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing formal reliability-gate input: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" "${GATE_ROOT}/scripts/holdout_guard.py" begin   --dataset "${HOLDOUT_DATASET}"   --base "${BASELINE_PREDICTIONS}"   --frozen-config "${FROZEN}"   --marker "${MARKER}"   "${RESUME_ARGS[@]}"

"${PYTHON}" "${GATE_ROOT}/scripts/build_visual_packets.py"   --dataset "${HOLDOUT_DATASET}"   --graph-dir "${HOLDOUT_GRAPH_DIR}"   --image-dir "${HOLDOUT_IMAGE_DIR}"   --output-dir "${PACKETS}"

"${PYTHON}" "${TYPED_ROOT}/scripts/run_visual_inspection.py"   --packet-dir "${PACKETS}/packets"   --output-dir "${VISUAL_AUDIT}"   --top-k 3   --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${VISUAL_AUDIT}/requests"   --report "${VISUAL_AUDIT}/leakage_audit.json"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" "${TYPED_ROOT}/scripts/run_visual_inspection.py"   --packet-dir "${PACKETS}/packets"   --output-dir "${VISUAL}"   --model "${VISION_MODEL}"   --top-k 3   "${QUANT_ARGS[@]}"

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py"   --dataset "${HOLDOUT_DATASET}"   --packet-dir "${PACKETS}/packets"   --policy "${POLICY}"   --visual-observation-dir "${VISUAL}/observations"   --output-dir "${SPECIALIST_AUDIT}"   --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${SPECIALIST_AUDIT}/prompts"   --report "${SPECIALIST_AUDIT}/leakage_audit.json"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py"   --dataset "${HOLDOUT_DATASET}"   --packet-dir "${PACKETS}/packets"   --policy "${POLICY}"   --visual-observation-dir "${VISUAL}/observations"   --output-dir "${SPECIALIST}"   --model "${TEXT_MODEL}"   "${QUANT_ARGS[@]}"

"${PYTHON}" "${GATE_ROOT}/scripts/apply_reliability_gate.py"   --dataset "${HOLDOUT_DATASET}"   --base "${BASELINE_PREDICTIONS}"   --visual-predictions "${SPECIALIST}/predictions.json"   --observation-dir "${VISUAL}/observations"   --output "${GATED_PREDICTIONS}"   --decisions-output "${DECISIONS}"   --require-all-visual-targets

"${PYTHON}" "${TYPED_ROOT}/scripts/compare_predictions.py"   --dataset "${HOLDOUT_DATASET}"   --baseline "${BASELINE_PREDICTIONS}"   --candidate "${GATED_PREDICTIONS}"   --output-json "${COMPARISON}"   --output-md "${RESULTS}/COMPARISON.md"

"${PYTHON}" "${GATE_ROOT}/scripts/paired_stats.py"   --dataset "${HOLDOUT_DATASET}"   --baseline "${BASELINE_PREDICTIONS}"   --candidate "${GATED_PREDICTIONS}"   --output "${BOOTSTRAP}"

"${PYTHON}" "${GATE_ROOT}/scripts/holdout_guard.py" finish   --marker "${MARKER}"   --result "${GATED_PREDICTIONS}"   --comparison "${COMPARISON}"   --bootstrap "${BOOTSTRAP}"

echo "Formal reliability-gate holdout completed once: ${RESULTS}"




