#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
DEFAULT_MODEL=""
MODEL="${MODEL_PATH:-$DEFAULT_MODEL}"
[[ -n "${MODEL}" ]] || { echo "ERROR: MODEL_PATH is empty." >&2; exit 2; }
DATASET="$REPO/memlens_repro/data/memlens_agent_subset/dataset_32k.json"
SUBGRAPH_DIR="$REPO/kg_retrieval/results/retrieval_budget120/retrieved_subgraphs"
PACKET_ROOT="$ROOT/results/packets_packet80"
PACKET_DIR="$PACKET_ROOT/packets"

if [[ ! -f "$PACKET_DIR/q_00a973a8.json" ]]; then
  python "$ROOT/scripts/build_answer_focused_packets.py" \
    --dataset "$DATASET" \
    --subgraph-dir "$SUBGRAPH_DIR" \
    --output-dir "$PACKET_ROOT" \
    --packet-budget 80
fi

python "$ROOT/scripts/answer_focused_kg_answering.py" \
  --dataset "$DATASET" \
  --packet-dir "$PACKET_DIR" \
  --policy "$ROOT/prompts/answer_focused_policy.md" \
  --output-dir "$ROOT/results/kg_answer_focused_packet80" \
  --model "$MODEL" \
  --generation-max-length 64

python "$ROOT/scripts/aggregate_eval_v2.py" \
  --output-dir "$ROOT/results/final_eval_v2" \
  --result-dirs \
    "$REPO/memlens_repro/outputs/kg_memory_32k_agent" \
    "$REPO/memlens_repro/outputs/qwen_vl_caption_rag_32k_agent" \
    "$REPO/memlens_repro/outputs/oracle_evidence_qwen25vl_32k_agent" \
    "$REPO/kg_retrieval/results/kg_soft_refusal_budget120" \
    "$ROOT/results/kg_answer_focused_packet80"

