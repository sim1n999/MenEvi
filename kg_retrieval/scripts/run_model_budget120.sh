#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
DEFAULT_MODEL=""
MODEL="${MODEL_PATH:-$DEFAULT_MODEL}"
[[ -n "${MODEL}" ]] || { echo "ERROR: MODEL_PATH is empty." >&2; exit 2; }
RETRIEVAL_DIR="$ROOT/results/retrieval_budget120"
SUBGRAPH_DIR="$RETRIEVAL_DIR/retrieved_subgraphs"

if [[ ! -f "$SUBGRAPH_DIR/q_00a973a8.json" ]]; then
  python "$ROOT/scripts/retrieve_kg_subgraphs.py" \
    --input "$REPO/memlens_repro/data/memlens_agent_subset/dataset_32k.json" \
    --graph-dir "$REPO/memlens_repro/outputs/kg_memory_32k_agent/graphs" \
    --output-dir "$RETRIEVAL_DIR" \
    --node-budget 120
fi

python "$ROOT/scripts/soft_refusal_kg_answering.py" \
  --input "$REPO/memlens_repro/data/memlens_agent_subset/dataset_32k.json" \
  --subgraph-dir "$SUBGRAPH_DIR" \
  --output-dir "$ROOT/results/kg_soft_refusal_budget120" \
  --policy "$ROOT/prompts/soft_refusal_policy.md" \
  --model "$MODEL" \
  --generation-max-length 128
