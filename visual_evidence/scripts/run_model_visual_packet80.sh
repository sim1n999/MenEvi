#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="visual_evidence"
DATASET="${DATASET:-memlens_repro/data/memlens_agent_subset/dataset_32k.json}"
SUBGRAPH_DIR="${SUBGRAPH_DIR:-kg_retrieval/results/retrieval_budget120/retrieved_subgraphs}"
MODEL_PATH="${MODEL_PATH:-}"
[[ -n "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH is empty." >&2; exit 2; }
PACKET_BUDGET="${PACKET_BUDGET:-80}"
GEN_MAX="${GEN_MAX:-64}"

PACKET_OUT="${EXPERIMENT_ROOT}/results/visual_ocr_packets_packet${PACKET_BUDGET}"
RUN_OUT="${EXPERIMENT_ROOT}/results/kg_visual_ocr_packet${PACKET_BUDGET}"
POLICY="${EXPERIMENT_ROOT}/prompts/type_aware_visual_ocr_policy.md"

python "${EXPERIMENT_ROOT}/scripts/build_visual_ocr_packets.py" \
  --dataset "${DATASET}" \
  --subgraph-dir "${SUBGRAPH_DIR}" \
  --output-dir "${PACKET_OUT}" \
  --packet-budget "${PACKET_BUDGET}"

python "${EXPERIMENT_ROOT}/scripts/type_aware_packet_answering.py" \
  --dataset "${DATASET}" \
  --packet-dir "${PACKET_OUT}/packets" \
  --policy "${POLICY}" \
  --output-dir "${RUN_OUT}" \
  --model "${MODEL_PATH}" \
  --generation-max-length "${GEN_MAX}"

python "answer_evidence/scripts/aggregate_eval_v2.py" \
  --output-dir "${EXPERIMENT_ROOT}/results/final_eval_v2" \
  --result-dirs \
    "answer_evidence/results/rescore_existing_v2/rescored_predictions/kg_memory_32k_agent" \
    "answer_evidence/results/rescore_existing_v2/rescored_predictions/qwen_vl_caption_rag_32k_agent" \
    "answer_evidence/results/rescore_existing_v2/rescored_predictions/oracle_evidence_qwen25vl_32k_agent" \
    "answer_evidence/results/rescore_existing_v2/rescored_predictions/kg_soft_refusal_budget120" \
    "answer_evidence/results/kg_answer_focused_packet80" \
    "${RUN_OUT}"

echo "Visual-evidence evaluation complete. Results: ${RUN_OUT}"
