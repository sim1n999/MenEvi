#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_ROOT="${ROOT}/reliability_gate"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
PYTHON="${PYTHON:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VISION_MODEL="${VISION_MODEL:-}"
TEXT_MODEL="${TEXT_MODEL:-}"
IMAGES="${IMAGES:-}"
ASSETS="${GATE_ROOT}/assets/holdout_594"
DATASET="${HOLDOUT_ROOT}/protocol/holdout_32k.json"
CAPTIONS="${ASSETS}/captions/qwen25vl_holdout.jsonl"
GRAPHS="${ASSETS}/kg_memory/graphs"
RETRIEVAL="${ASSETS}/retrieval_budget120"
SUBGRAPHS="${RETRIEVAL}/retrieved_subgraphs"
ANSWER_PACKET_ROOT="${ASSETS}/c_packets"
VISUAL_PACKET_ROOT="${ASSETS}/d_packets"
SPECIALIST_PACKET_ROOT="${ASSETS}/specialist_packets"
BASELINE_PROMPT_AUDIT="${ASSETS}/baseline_prompt_audit"

if [[ -e "${GATE_ROOT}/frozen_config.json" ]]; then
  echo "Assets must be completed before I is frozen." >&2
  exit 2
fi
[[ -n "${VISION_MODEL}" ]] || { echo "ERROR: VISION_MODEL is empty." >&2; exit 2; }
[[ -n "${TEXT_MODEL}" ]] || { echo "ERROR: TEXT_MODEL is empty." >&2; exit 2; }
[[ -n "${IMAGES}" ]] || { echo "ERROR: IMAGES is empty. Set it to the MemLens release_images directory." >&2; exit 2; }

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/build_holdout_split.py"   --full-dataset "${ROOT}/memlens_repro/data/memlens/dataset_32k.json"   --touched-dataset "${ROOT}/memlens_repro/data/memlens_agent_subset/dataset_32k.json"   --output-dir "${HOLDOUT_ROOT}/protocol"   --write-datasets

"${PYTHON}" "${GATE_ROOT}/scripts/build_caption_cache_resumable.py"   --input "${DATASET}" --image-dir "${IMAGES}"   --output "${CAPTIONS}"   --request-dir "${ASSETS}/caption_prompt_audit/requests"   --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${ASSETS}/caption_prompt_audit/requests"   --report "${ASSETS}/caption_prompt_audit/leakage_audit.json"

if [[ "${PROMPT_ONLY:-0}" == "1" ]]; then
  echo "Holdout asset prompt audit complete; no model was loaded."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}"   "${GATE_ROOT}/scripts/build_caption_cache_resumable.py"   --input "${DATASET}" --image-dir "${IMAGES}"   --model "${VISION_MODEL}" --output "${CAPTIONS}"

"${PYTHON}" "${GATE_ROOT}/scripts/build_label_blind_graphs.py"   --dataset "${DATASET}" --caption-cache "${CAPTIONS}"   --output-dir "${GRAPHS}"

"${PYTHON}" "${ROOT}/kg_retrieval/scripts/retrieve_kg_subgraphs.py"   --input "${DATASET}" --graph-dir "${GRAPHS}"   --output-dir "${RETRIEVAL}" --node-budget 120

"${PYTHON}" "${ROOT}/answer_evidence/scripts/build_answer_focused_packets.py"   --dataset "${DATASET}" --subgraph-dir "${SUBGRAPHS}"   --output-dir "${ANSWER_PACKET_ROOT}" --packet-budget 80
"${PYTHON}" "${ROOT}/visual_evidence/scripts/build_visual_ocr_packets.py"   --dataset "${DATASET}" --subgraph-dir "${SUBGRAPHS}"   --output-dir "${VISUAL_PACKET_ROOT}" --packet-budget 80
"${PYTHON}" "${ROOT}/runtime_routing/scripts/build_specialist_packets.py"   --dataset "${DATASET}" --subgraph-dir "${SUBGRAPHS}"   --output-dir "${SPECIALIST_PACKET_ROOT}"

"${PYTHON}" "${GATE_ROOT}/scripts/sanitize_packets.py"   --packet-dir "${ANSWER_PACKET_ROOT}/packets"   --packet-dir "${VISUAL_PACKET_ROOT}/packets"   --packet-dir "${SPECIALIST_PACKET_ROOT}/packets"

"${PYTHON}" "${ROOT}/runtime_routing/scripts/runtime_hybrid_answering.py"   --dataset "${DATASET}"   --c-packet-dir "${ANSWER_PACKET_ROOT}/packets"   --d-packet-dir "${VISUAL_PACKET_ROOT}/packets"   --specialist-packet-dir "${SPECIALIST_PACKET_ROOT}/packets"   --policy-c "${ROOT}/answer_evidence/prompts/answer_focused_policy.md"   --policy-d "${ROOT}/visual_evidence/prompts/type_aware_visual_ocr_policy.md"   --policy-duration "${ROOT}/runtime_routing/prompts/duration_specialist_policy.md"   --policy-arithmetic "${ROOT}/runtime_routing/prompts/arithmetic_specialist_policy.md"   --variant runtime_specialists   --output-dir "${BASELINE_PROMPT_AUDIT}"   --prompt-only
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py"   --prompt-dir "${BASELINE_PROMPT_AUDIT}/prompts"   --report "${BASELINE_PROMPT_AUDIT}/leakage_audit.json"

"${PYTHON}" "${GATE_ROOT}/scripts/validate_holdout_assets.py" \
  --dataset "${DATASET}" --caption-cache "${CAPTIONS}" \
  --asset-root "${ASSETS}" \
  --audit "${BASELINE_PROMPT_AUDIT}/leakage_audit.json" \
  --output "${ASSETS}/asset_manifest.json"

echo "I holdout assets ready for freeze: ${ASSETS}"


