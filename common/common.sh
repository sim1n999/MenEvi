#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${COMMON_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEXT_MODEL="${TEXT_MODEL:-}"
VISION_MODEL="${VISION_MODEL:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
DATA_ROOT="${DATA_ROOT:-}"
SHAREVISUAL_PACKET_ROOT="${EVIMEM_SHARED_PACKET_ROOT:-${ROOT}/common/generated}"
CAPTION_CACHE="${CAPTION_CACHE:-${SHAREVISUAL_PACKET_ROOT}/caption_cache_all_lengths.jsonl}"
DRY_RUN="${DRY_RUN:-0}"
PROFILE="${PROFILE:-0}"
NO_4BIT="${NO_4BIT:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

require_path_var() {
  local name="$1" value="$2" kind="${3:-path}"
  if [[ -z "${value}" ]]; then
    echo "Required ${kind} variable ${name} is empty. Set it in configs/local_paths.env or export it before running." >&2
    exit 2
  fi
}

require_runtime_paths() {
  require_path_var TEXT_MODEL "${TEXT_MODEL}" "model"
  require_path_var VISION_MODEL "${VISION_MODEL}" "model"
  require_path_var DATA_ROOT "${DATA_ROOT}" "data"
  require_path_var IMAGE_DIR "${IMAGE_DIR}" "data"
}

if [[ -n "${MAX_SAMPLES}" && "${DRY_RUN}" != "1" ]]; then
  echo "MAX_SAMPLES is allowed only with DRY_RUN=1; published runs must cover all 789 questions." >&2
  exit 2
fi

mkdir -p "${SHAREVISUAL_PACKET_ROOT}"
require_runtime_paths
qargs=()
if [[ "${NO_4BIT}" == "1" ]]; then qargs+=(--no-4bit); fi
margs=()
if [[ -n "${MAX_SAMPLES}" ]]; then margs+=(--max-samples "${MAX_SAMPLES}"); fi

dataset_path() {
  printf '%s/dataset_%s.json' "${DATA_ROOT}" "$1"
}

assert_full_dataset() {
  "${PYTHON}" "${COMMON_DIR}/scripts/assert_dataset.py" --dataset "$1" --expected-count "${2:-789}"
}

run_step() {
  local label="$1" marker="$2"
  shift 2
  if [[ -e "${marker}" ]]; then echo "[skip] ${label}: ${marker}"; return 0; fi
  mkdir -p "$(dirname "${marker}")"
  echo "[run] ${label}"
  if [[ "${DRY_RUN}" == "1" ]]; then printf '  %q' "$@"; echo; return 0; fi
  if [[ "${PROFILE}" == "1" ]]; then
    "${PYTHON}" "${COMMON_DIR}/scripts/run_with_profile.py" \
      --label "${label}" --output "${marker}.profile.json" -- "$@"
  else
    "$@"
  fi
  "${PYTHON}" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("complete\n", encoding="utf-8")' "${marker}"
}

ensure_caption_cache() {
  local union="${SHAREVISUAL_PACKET_ROOT}/caption_union_all_lengths.json"
  run_step "caption-union-all-lengths" "${SHAREVISUAL_PACKET_ROOT}/.caption_union.complete" \
    "${PYTHON}" "${COMMON_DIR}/scripts/build_caption_union.py" \
    --dataset "$(dataset_path 32k)" --dataset "$(dataset_path 64k)" \
    --dataset "$(dataset_path 128k)" --dataset "$(dataset_path 256k)" --output "${union}"
  run_step "caption-cache-all-lengths" "${SHAREVISUAL_PACKET_ROOT}/.caption_cache.complete" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    "${ROOT}/reliability_gate/scripts/build_caption_cache_resumable.py" \
    --input "${union}" --image-dir "${IMAGE_DIR}" --model "${VISION_MODEL}" \
    --output "${CAPTION_CACHE}" "${qargs[@]}"
}

prepare_packets() {
  local dataset="$1" subgraphs="$2" out="$3"
  run_step "packets-c-$(basename "${out}")" "${out}/.c_packets.complete" \
    "${PYTHON}" "${ROOT}/answer_evidence/scripts/build_answer_focused_packets.py" \
    --dataset "${dataset}" --subgraph-dir "${subgraphs}" --output-dir "${out}/c_packets" \
    --packet-budget 80 "${margs[@]}"
  run_step "packets-d-$(basename "${out}")" "${out}/.d_packets.complete" \
    "${PYTHON}" "${ROOT}/visual_evidence/scripts/build_visual_ocr_packets.py" \
    --dataset "${dataset}" --subgraph-dir "${subgraphs}" --output-dir "${out}/d_packets" \
    --packet-budget 80 "${margs[@]}"
  run_step "packets-specialist-$(basename "${out}")" "${out}/.specialist_packets.complete" \
    "${PYTHON}" "${ROOT}/runtime_routing/scripts/build_specialist_packets.py" \
    --dataset "${dataset}" --subgraph-dir "${subgraphs}" --output-dir "${out}/specialist_packets" "${margs[@]}"
  run_step "sanitize-packets-$(basename "${out}")" "${out}/.sanitize.complete" \
    "${PYTHON}" "${COMMON_DIR}/scripts/sanitize_full_packets.py" \
    --packet-dir "${out}/c_packets/packets" --packet-dir "${out}/d_packets/packets" \
    --packet-dir "${out}/specialist_packets/packets"
}

ensure_core_assets() {
  local length="$1" dataset out
  dataset="$(dataset_path "${length}")"
  out="${SHAREVISUAL_PACKET_ROOT}/assets_${length}_full"
  assert_full_dataset "${dataset}"
  ensure_caption_cache
  run_step "${length}-label-blind-graphs" "${out}/.graphs.complete" \
    "${PYTHON}" "${ROOT}/reliability_gate/scripts/build_label_blind_graphs.py" \
    --dataset "${dataset}" --caption-cache "${CAPTION_CACHE}" --output-dir "${out}/kg_memory/graphs"
  run_step "${length}-subgraph-retrieval" "${out}/.retrieval.complete" \
    "${PYTHON}" "${ROOT}/kg_retrieval/scripts/retrieve_kg_subgraphs.py" \
    --input "${dataset}" --graph-dir "${out}/kg_memory/graphs" \
    --output-dir "${out}/retrieval_budget120" --node-budget 120 "${margs[@]}"
  prepare_packets "${dataset}" "${out}/retrieval_budget120/retrieved_subgraphs" "${out}"
}

run_typed_base() {
  local dataset="$1" assets="$2" out="$3" variant="${4:-runtime_specialists}"
  run_step "typed-base-$(basename "${out}")" "${out}/.base.complete" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    "${ROOT}/runtime_routing/scripts/runtime_hybrid_answering.py" \
    --dataset "${dataset}" --c-packet-dir "${assets}/c_packets/packets" \
    --d-packet-dir "${assets}/d_packets/packets" --specialist-packet-dir "${assets}/specialist_packets/packets" \
    --policy-c "${ROOT}/answer_evidence/prompts/answer_focused_policy.md" \
    --policy-d "${ROOT}/visual_evidence/prompts/type_aware_visual_ocr_policy.md" \
    --policy-duration "${ROOT}/runtime_routing/prompts/duration_specialist_policy.md" \
    --policy-arithmetic "${ROOT}/runtime_routing/prompts/arithmetic_specialist_policy.md" \
    --variant "${variant}" --output-dir "${out}" --model "${TEXT_MODEL}" "${qargs[@]}" "${margs[@]}"
  run_step "audit-base-$(basename "${out}")" "${out}/.audit.complete" \
    "${PYTHON}" "${ROOT}/holdout_validation/scripts/audit_prompt_leakage.py" \
    --prompt-dir "${out}/prompts" --report "${out}/leakage_audit.json"
}

run_visual_gate() {
  local dataset="$1" graph_dir="$2" base_predictions="$3" out="$4"
  run_step "visual-packets-$(basename "${out}")" "${out}/.visual_packets.complete" \
    "${PYTHON}" "${ROOT}/reliability_gate/scripts/build_visual_packets.py" \
    --dataset "${dataset}" --graph-dir "${graph_dir}" --image-dir "${IMAGE_DIR}" \
    --output-dir "${out}/visual_packets" "${margs[@]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    local visual_count
    visual_count="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["count"])' "${out}/visual_packets/manifest.json")"
    if [[ "${visual_count}" == "0" ]]; then
      echo "[info] No visual targets for $(basename "${out}"); using an empty candidate payload."
      run_step "empty-visual-outputs-$(basename "${out}")" "${out}/.empty_visual.complete" \
        "${PYTHON}" "${COMMON_DIR}/scripts/create_empty_visual_outputs.py" \
        --output-dir "${out}" --dataset "${dataset}"
      run_step "reliability-gate-$(basename "${out}")" "${out}/.gate.complete" \
        "${PYTHON}" "${ROOT}/reliability_gate/scripts/apply_reliability_gate.py" \
        --dataset "${dataset}" --base "${base_predictions}" \
        --visual-predictions "${out}/visual_specialist/predictions.json" \
        --observation-dir "${out}/visual_inspection/observations" \
        --output "${out}/full_predictions.json" --decisions-output "${out}/gate_decisions.json" \
        --require-all-visual-targets
      return 0
    fi
  fi
  run_step "visual-inspection-prompts-$(basename "${out}")" "${out}/.visual_prompt.complete" \
    "${PYTHON}" "${ROOT}/typed_evidence/scripts/run_visual_inspection.py" \
    --packet-dir "${out}/visual_packets/packets" --output-dir "${out}/visual_prompt_audit" \
    --top-k 3 --prompt-only "${margs[@]}"
  run_step "audit-visual-$(basename "${out}")" "${out}/.visual_audit.complete" \
    "${PYTHON}" "${ROOT}/holdout_validation/scripts/audit_prompt_leakage.py" \
    --prompt-dir "${out}/visual_prompt_audit/requests" --report "${out}/visual_prompt_audit/leakage_audit.json"
  run_step "visual-inspection-$(basename "${out}")" "${out}/.visual.complete" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    "${ROOT}/typed_evidence/scripts/run_visual_inspection.py" \
    --packet-dir "${out}/visual_packets/packets" --output-dir "${out}/visual_inspection" \
    --model "${VISION_MODEL}" --top-k 3 "${qargs[@]}" "${margs[@]}"
  run_step "visual-specialist-prompts-$(basename "${out}")" "${out}/.specialist_prompt.complete" \
    "${PYTHON}" "${ROOT}/holdout_validation/scripts/run_typed_specialists_v2.py" \
    --dataset "${dataset}" --packet-dir "${out}/visual_packets/packets" \
    --policy "${ROOT}/holdout_validation/prompts/typed_specialist_policy_v2.md" \
    --visual-observation-dir "${out}/visual_inspection/observations" \
    --output-dir "${out}/specialist_prompt_audit" --prompt-only "${margs[@]}"
  run_step "audit-specialist-$(basename "${out}")" "${out}/.specialist_audit.complete" \
    "${PYTHON}" "${ROOT}/holdout_validation/scripts/audit_prompt_leakage.py" \
    --prompt-dir "${out}/specialist_prompt_audit/prompts" \
    --report "${out}/specialist_prompt_audit/leakage_audit.json"
  run_step "visual-specialist-$(basename "${out}")" "${out}/.specialist.complete" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
    "${ROOT}/holdout_validation/scripts/run_typed_specialists_v2.py" \
    --dataset "${dataset}" --packet-dir "${out}/visual_packets/packets" \
    --policy "${ROOT}/holdout_validation/prompts/typed_specialist_policy_v2.md" \
    --visual-observation-dir "${out}/visual_inspection/observations" \
    --output-dir "${out}/visual_specialist" --model "${TEXT_MODEL}" "${qargs[@]}" "${margs[@]}"
  run_step "reliability-gate-$(basename "${out}")" "${out}/.gate.complete" \
    "${PYTHON}" "${ROOT}/reliability_gate/scripts/apply_reliability_gate.py" \
    --dataset "${dataset}" --base "${base_predictions}" \
    --visual-predictions "${out}/visual_specialist/predictions.json" \
    --observation-dir "${out}/visual_inspection/observations" \
    --output "${out}/full_predictions.json" --decisions-output "${out}/gate_decisions.json" \
    --require-all-visual-targets
}

compare_predictions() {
  local dataset="$1" baseline="$2" candidate="$3" out_prefix="$4"
  run_step "compare-$(basename "${out_prefix}")" "${out_prefix}.complete" \
    "${PYTHON}" "${ROOT}/typed_evidence/scripts/compare_predictions.py" \
    --dataset "${dataset}" --baseline "${baseline}" --candidate "${candidate}" \
    --output-json "${out_prefix}.json" --output-md "${out_prefix}.md"
}


