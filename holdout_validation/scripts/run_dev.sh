#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
TYPED_ROOT="${ROOT}/typed_evidence"
PYTHON="${PYTHON:-python}"
TEXT_MODEL="${TEXT_MODEL:-}"
VISION_MODEL="${VISION_MODEL:-}"
[[ -n "${TEXT_MODEL}" ]] || { echo "ERROR: TEXT_MODEL is empty." >&2; exit 2; }
[[ -n "${VISION_MODEL}" ]] || { echo "ERROR: VISION_MODEL is empty." >&2; exit 2; }
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PROMPT_ONLY="${PROMPT_ONLY:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

DATASET="${ROOT}/memlens_repro/data/memlens_agent_subset/dataset_32k.json"
GRAPHS="${ROOT}/memlens_repro/outputs/kg_memory_32k_agent/graphs"
IMAGES="${ROOT}/memlens_repro/data/memlens/release_images"
BASE="${ROOT}/runtime_routing/results/runtime_specialists/predictions.json"
RESULTS="${HOLDOUT_ROOT}/results/dev_195"
PACKET_ROOT="${RESULTS}/typed_packets"
PACKETS="${PACKET_ROOT}/packets"
VISUAL_AUDIT="${RESULTS}/visual_prompt_audit"
VISUAL="${RESULTS}/visual_inspection"
SPECIALIST_AUDIT="${RESULTS}/specialist_prompt_audit"
SPECIALISTS="${RESULTS}/typed_specialists"
CANDIDATE_PREDICTIONS="${SPECIALISTS}/predictions.json"

for required in "${DATASET}" "${GRAPHS}" "${IMAGES}" "${BASE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
QUANT_ARGS=()
if [[ "${NO_4BIT:-0}" == "1" ]]; then
  QUANT_ARGS+=(--no-4bit)
fi

"${PYTHON}" -m pytest "${HOLDOUT_ROOT}/tests" -q
"${PYTHON}" "${TYPED_ROOT}/scripts/eval_v21.py" --self-test

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/build_typed_evidence_v2.py" \
  --dataset "${DATASET}" \
  --graph-dir "${GRAPHS}" \
  --image-dir "${IMAGES}" \
  --output-dir "${PACKET_ROOT}" \
  "${SAMPLE_ARGS[@]}"

"${PYTHON}" "${TYPED_ROOT}/scripts/run_visual_inspection.py" \
  --packet-dir "${PACKETS}" \
  --output-dir "${VISUAL_AUDIT}" \
  --top-k 3 \
  --prompt-only \
  "${SAMPLE_ARGS[@]}"
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${VISUAL_AUDIT}/requests" \
  --report "${VISUAL_AUDIT}/leakage_audit.json"

if [[ "${PROMPT_ONLY}" == "1" ]]; then
  "${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py" \
    --dataset "${DATASET}" \
    --packet-dir "${PACKETS}" \
    --visual-observation-dir "${VISUAL_AUDIT}/observations" \
    --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md" \
    --output-dir "${SPECIALIST_AUDIT}" \
    --prompt-only \
    "${SAMPLE_ARGS[@]}"
  "${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
    --prompt-dir "${SPECIALIST_AUDIT}/prompts" \
    --report "${SPECIALIST_AUDIT}/leakage_audit.json"
  echo "H development prompt-only audit completed: ${RESULTS}"
  exit 0
fi

for required in "${TEXT_MODEL}" "${VISION_MODEL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required model path: ${required}" >&2
    exit 2
  fi
done

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${TYPED_ROOT}/scripts/run_visual_inspection.py" \
  --packet-dir "${PACKETS}" \
  --output-dir "${VISUAL}" \
  --model "${VISION_MODEL}" \
  --top-k 3 \
  "${QUANT_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py" \
  --dataset "${DATASET}" \
  --packet-dir "${PACKETS}" \
  --visual-observation-dir "${VISUAL}/observations" \
  --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md" \
  --output-dir "${SPECIALIST_AUDIT}" \
  --prompt-only \
  "${SAMPLE_ARGS[@]}"
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${SPECIALIST_AUDIT}/prompts" \
  --report "${SPECIALIST_AUDIT}/leakage_audit.json"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${HOLDOUT_ROOT}/scripts/run_typed_specialists_v2.py" \
  --dataset "${DATASET}" \
  --packet-dir "${PACKETS}" \
  --visual-observation-dir "${VISUAL}/observations" \
  --policy "${HOLDOUT_ROOT}/prompts/typed_specialist_policy_v2.md" \
  --output-dir "${SPECIALISTS}" \
  --model "${TEXT_MODEL}" \
  "${QUANT_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "Holdout development smoke run completed; merge skipped because MAX_SAMPLES is set."
  exit 0
fi

run_condition() {
  local name="$1"
  shift
  local out_dir="${RESULTS}/${name}"
  "${PYTHON}" "${HOLDOUT_ROOT}/scripts/merge_with_base.py" \
    --dataset "${DATASET}" \
    --base "${BASE}" \
    --candidate "${CANDIDATE_PREDICTIONS}" \
    --output "${out_dir}/predictions.json" \
    --require-all-targets \
    --include-subtypes "$@"
  "${PYTHON}" "${TYPED_ROOT}/scripts/compare_predictions.py" \
    --dataset "${DATASET}" \
    --baseline "${BASE}" \
    --candidate "${out_dir}/predictions.json" \
    --output-json "${out_dir}/comparison.json" \
    --output-md "${out_dir}/COMPARISON.md"
}

run_condition holdout_arithmetic arithmetic
run_condition holdout_duration duration_comparison
run_condition holdout_visual entity previnfo
run_condition holdout_merged arithmetic duration_comparison entity previnfo

echo "H development run completed: ${RESULTS}/holdout_merged"



