#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TYPED_ROOT="${ROOT}/typed_evidence"
TEXT_MODEL="${TEXT_MODEL:-}"
VISION_MODEL="${VISION_MODEL:-}"
[[ -n "${TEXT_MODEL}" ]] || { echo "ERROR: TEXT_MODEL is empty." >&2; exit 2; }
[[ -n "${VISION_MODEL}" ]] || { echo "ERROR: VISION_MODEL is empty." >&2; exit 2; }
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VISUAL_TOP_K="${VISUAL_TOP_K:-3}"

DATASET="${ROOT}/memlens_repro/data/memlens_agent_subset/dataset_32k.json"
GRAPHS="${ROOT}/memlens_repro/outputs/kg_memory_32k_agent/graphs"
IMAGES="${ROOT}/memlens_repro/data/memlens/release_images"
PACKET_ROOT="${TYPED_ROOT}/results/typed_packets"
PACKETS="${PACKET_ROOT}/packets"
VISUAL="${TYPED_ROOT}/results/visual_inspection"
SPECIALISTS="${TYPED_ROOT}/results/typed_specialists"
CANDIDATE_PREDICTIONS="${SPECIALISTS}/predictions.json"
BASELINE_PREDICTIONS="${ROOT}/runtime_routing/results/runtime_specialists/predictions.json"

for required in "${TEXT_MODEL}" "${VISION_MODEL}" "${DATASET}" "${GRAPHS}" "${IMAGES}" "${BASELINE_PREDICTIONS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

EXTRA_ARGS=()
if [[ "${NO_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-4bit)
fi

python "${TYPED_ROOT}/scripts/eval_v21.py" --self-test
python -m unittest discover -s "${TYPED_ROOT}/tests" -p 'test_*.py' -v
python "${TYPED_ROOT}/scripts/build_typed_evidence.py" \
  --dataset "${DATASET}" --graph-dir "${GRAPHS}" --image-dir "${IMAGES}" --output-dir "${PACKET_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${TYPED_ROOT}/scripts/run_visual_inspection.py" \
  --packet-dir "${PACKETS}" --output-dir "${VISUAL}" --model "${VISION_MODEL}" \
  --top-k "${VISUAL_TOP_K}" "${EXTRA_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${TYPED_ROOT}/scripts/run_typed_specialists.py" \
  --dataset "${DATASET}" --packet-dir "${PACKETS}" \
  --visual-observation-dir "${VISUAL}/observations" \
  --policy "${TYPED_ROOT}/prompts/typed_specialist_policy.md" \
  --output-dir "${SPECIALISTS}" --model "${TEXT_MODEL}" "${EXTRA_ARGS[@]}"

run_condition() {
  local name="$1"
  shift
  local out_dir="${TYPED_ROOT}/results/${name}"
  python "${TYPED_ROOT}/scripts/merge_with_baseline.py" \
    --dataset "${DATASET}" --baseline "${BASELINE_PREDICTIONS}" --candidate "${CANDIDATE_PREDICTIONS}" \
    --output "${out_dir}/predictions.json" --require-all-targets --include-subtypes "$@"
  python "${TYPED_ROOT}/scripts/compare_predictions.py" \
    --dataset "${DATASET}" --baseline "${BASELINE_PREDICTIONS}" --candidate "${out_dir}/predictions.json" \
    --output-json "${out_dir}/comparison.json" --output-md "${out_dir}/COMPARISON.md"
}

run_condition typed_arithmetic arithmetic
run_condition typed_duration duration_comparison
run_condition typed_visual entity previnfo
run_condition typed_merged arithmetic duration_comparison entity previnfo

echo "Typed-evidence evaluation completed: ${TYPED_ROOT}/results/typed_merged"



