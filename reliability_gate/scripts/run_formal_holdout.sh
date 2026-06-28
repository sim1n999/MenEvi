#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_ROOT="${ROOT}/reliability_gate"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
TYPED_ROOT="${ROOT}/typed_evidence"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
IMAGE_DIR="${IMAGE_DIR:-}"
ASSETS="${GATE_ROOT}/assets/holdout_594"
DATASET="${HOLDOUT_ROOT}/protocol/holdout_32k.json"
FROZEN="${GATE_ROOT}/frozen_config.json"
RESULTS="${GATE_ROOT}/results/formal_holdout_594"
MARKER="${RESULTS}/formal_run_marker.json"
BASELINE_DIR="${RESULTS}/baseline_predictions"
BASELINE_PREDICTIONS="${BASELINE_DIR}/predictions.json"

if [[ "${FORMAL_HOLDOUT_ACK:-}" != "YES" ]]; then
  echo "Set FORMAL_HOLDOUT_ACK=YES only for the one frozen formal execution." >&2
  exit 2
fi
[[ -n "${IMAGE_DIR}" ]] || { echo "ERROR: IMAGE_DIR is empty. Set it to the MemLens release_images directory." >&2; exit 2; }
for path in "${FROZEN}" "${DATASET}" "${ASSETS}/asset_manifest.json"; do
  [[ -e "${path}" ]] || { echo "Missing formal prerequisite: ${path}" >&2; exit 2; }
done
if [[ ! -e "${MARKER}" && -e "${BASELINE_PREDICTIONS}" ]]; then
  echo "Refusing unguarded pre-existing baseline predictions." >&2
  exit 2
fi

"${PYTHON}" "${GATE_ROOT}/scripts/freeze_config.py" verify --frozen-config "${FROZEN}"
TEXT_MODEL="$("${PYTHON}" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["models"]["text"])' "${FROZEN}")"
VISION_MODEL="$("${PYTHON}" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["models"]["vision"])' "${FROZEN}")"
LOAD_4BIT="$("${PYTHON}" -c 'import json,sys;print("1" if json.load(open(sys.argv[1],encoding="utf-8"))["generation"]["load_in_4bit"] else "0")' "${FROZEN}")"
Q=()
if [[ "${LOAD_4BIT}" != "1" ]]; then Q+=(--no-4bit); fi
R=()
if [[ "${RESUME_FORMAL_GATE:-0}" == "1" ]]; then R+=(--resume); fi

"${PYTHON}" "${GATE_ROOT}/scripts/formal_run_guard.py" begin   --dataset "${DATASET}" --frozen-config "${FROZEN}"   --marker "${MARKER}" "${R[@]}"

if [[ ! -e "${BASELINE_PREDICTIONS}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}"     "${ROOT}/runtime_routing/scripts/runtime_hybrid_answering.py"     --dataset "${DATASET}"     --c-packet-dir "${ASSETS}/c_packets/packets"     --d-packet-dir "${ASSETS}/d_packets/packets"     --specialist-packet-dir "${ASSETS}/specialist_packets/packets"     --policy-c "${ROOT}/answer_evidence/prompts/answer_focused_policy.md"     --policy-d "${ROOT}/visual_evidence/prompts/type_aware_visual_ocr_policy.md"     --policy-duration "${ROOT}/runtime_routing/prompts/duration_specialist_policy.md"     --policy-arithmetic "${ROOT}/runtime_routing/prompts/arithmetic_specialist_policy.md"     --variant runtime_specialists --output-dir "${BASELINE_DIR}"     --model "${TEXT_MODEL}" "${Q[@]}"
fi
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${BASELINE_DIR}/prompts" --report "${BASELINE_DIR}/leakage_audit.json"
"${PYTHON}" "${GATE_ROOT}/scripts/formal_run_guard.py" register-base   --marker "${MARKER}" --base "${BASELINE_PREDICTIONS}"

PACKETS="${RESULTS}/visual_packets"
VISUAL="${RESULTS}/visual_inspection"
SPECIALIST="${RESULTS}/visual_specialist"
"${PYTHON}" "${GATE_ROOT}/scripts/build_visual_packets.py"   --dataset "${DATASET}" --graph-dir "${ASSETS}/kg_memory/graphs"   --image-dir "${IMAGE_DIR}"   --output-dir "${PACKETS}"
"${PYTHON}" "${TYPED_ROOT}/scripts/run_visual_inspection.py"   --packet-dir "${PACKETS}/packets"   --output-dir "${RESULTS}/visual_prompt_audit" --top-k 3 --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${RESULTS}/visual_prompt_audit/requests"   --report "${RESULTS}/visual_prompt_audit/leakage_audit.json"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}"   "${TYPED_ROOT}/scripts/run_visual_inspection.py"   --packet-dir "${PACKETS}/packets" --output-dir "${VISUAL}"   --model "${VISION_MODEL}" --top-k 3 "${Q[@]}"

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py"   --dataset "${DATASET}" --packet-dir "${PACKETS}/packets"   --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md"   --visual-observation-dir "${VISUAL}/observations"   --output-dir "${RESULTS}/specialist_prompt_audit" --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${RESULTS}/specialist_prompt_audit/prompts"   --report "${RESULTS}/specialist_prompt_audit/leakage_audit.json"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}"   "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py"   --dataset "${DATASET}" --packet-dir "${PACKETS}/packets"   --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md"   --visual-observation-dir "${VISUAL}/observations"   --output-dir "${SPECIALIST}" --model "${TEXT_MODEL}" "${Q[@]}"

GATED_PREDICTIONS="${RESULTS}/gated_predictions.json"
COMPARE="${RESULTS}/comparison.json"
BOOT="${RESULTS}/paired_bootstrap.json"
"${PYTHON}" "${GATE_ROOT}/scripts/apply_reliability_gate.py"   --dataset "${DATASET}" --base "${BASELINE_PREDICTIONS}"   --visual-predictions "${SPECIALIST}/predictions.json"   --observation-dir "${VISUAL}/observations" --output "${GATED_PREDICTIONS}"   --decisions-output "${RESULTS}/gated_decisions.json" --require-all-visual-targets
"${PYTHON}" "${TYPED_ROOT}/scripts/compare_predictions.py"   --dataset "${DATASET}" --baseline "${BASELINE_PREDICTIONS}" --candidate "${GATED_PREDICTIONS}"   --output-json "${COMPARE}" --output-md "${RESULTS}/COMPARISON.md"
"${PYTHON}" "${GATE_ROOT}/scripts/paired_stats.py"   --dataset "${DATASET}" --baseline "${BASELINE_PREDICTIONS}" --candidate "${GATED_PREDICTIONS}"   --output "${BOOT}"
"${PYTHON}" "${GATE_ROOT}/scripts/formal_run_guard.py" finish   --marker "${MARKER}" --result "${GATED_PREDICTIONS}"   --comparison "${COMPARE}" --bootstrap "${BOOT}"
echo "One formal reliability-gate holdout execution completed: ${RESULTS}"




