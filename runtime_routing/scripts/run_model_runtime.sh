#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
[[ -n "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH is empty." >&2; exit 2; }
VARIANT="${VARIANT:-runtime_cd}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

DATASET="${ROOT}/memlens_repro/data/memlens_agent_subset/dataset_32k.json"
SUBGRAPHS="${ROOT}/kg_retrieval/results/retrieval_budget120/retrieved_subgraphs"
C_PACKETS="${ROOT}/answer_evidence/results/packets_packet80/packets"
D_PACKETS="${ROOT}/visual_evidence/results/visual_ocr_packets_packet80/packets"
SPECIALIST_ROOT="${ROOT}/runtime_routing/results/specialist_packets"
OUT_DIR="${ROOT}/runtime_routing/results/${VARIANT}"

python "${ROOT}/runtime_routing/scripts/build_specialist_packets.py" \
  --dataset "${DATASET}" \
  --subgraph-dir "${SUBGRAPHS}" \
  --output-dir "${SPECIALIST_ROOT}"

EXTRA_ARGS=()
if [[ "${NO_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-4bit)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${ROOT}/runtime_routing/scripts/runtime_hybrid_answering.py" \
  --dataset "${DATASET}" \
  --c-packet-dir "${C_PACKETS}" \
  --d-packet-dir "${D_PACKETS}" \
  --specialist-packet-dir "${SPECIALIST_ROOT}/packets" \
  --policy-c "${ROOT}/answer_evidence/prompts/answer_focused_policy.md" \
  --policy-d "${ROOT}/visual_evidence/prompts/type_aware_visual_ocr_policy.md" \
  --policy-duration "${ROOT}/runtime_routing/prompts/duration_specialist_policy.md" \
  --policy-arithmetic "${ROOT}/runtime_routing/prompts/arithmetic_specialist_policy.md" \
  --variant "${VARIANT}" \
  --output-dir "${OUT_DIR}" \
  --model "${MODEL_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "Runtime-routing evaluation ${VARIANT} completed: ${OUT_DIR}"

