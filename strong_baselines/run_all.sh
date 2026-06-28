#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${EXP_ROOT}/.." && pwd)"
source "${ROOT}/common/common.sh"

RUN_TAG="${RUN_TAG:-default}"
TOP_K="${TOP_K:-5}"
DATASET="$(dataset_path 32k)"
OUT="${EXP_ROOT}/results/${RUN_TAG}"
P_OUT="${ROOT}/length_curve/results/${RUN_TAG}/32k/p1_full_system/full_predictions.json"
mkdir -p "${OUT}"
vlm_quant=()
if [[ "${NO_4BIT}" != "1" ]]; then vlm_quant+=(--load_in_4bit); fi
assert_full_dataset "${DATASET}"
ensure_caption_cache
LENGTHS=32k RUN_TAG="${RUN_TAG}" bash "${ROOT}/length_curve/run_all.sh"

run_step "baseline-bm25-text" "${OUT}/bm25_text/.run.complete" \
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${ROOT}/memlens_repro/scripts/run_bm25_text_rag.py" --input "${DATASET}" \
  --output-dir "${OUT}/bm25_text" --model "${TEXT_MODEL}" --top-k "${TOP_K}" "${qargs[@]}"
run_step "baseline-caption-rag" "${OUT}/caption_rag/.run.complete" \
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" \
  "${ROOT}/memlens_repro/scripts/run_caption_rag.py" --input "${DATASET}" --image-dir "${IMAGE_DIR}" \
  --caption-source qwen_vl --caption-cache "${CAPTION_CACHE}" --output-dir "${OUT}/caption_rag" \
  --model "${TEXT_MODEL}" --top-k "${TOP_K}" "${qargs[@]}"

run_step "baseline-direct-lvlm" "${OUT}/direct_lvlm/.run.complete" \
  bash "${ROOT}/memlens_repro/scripts/run_qwen25vl_memlens.sh" --dataset 32k --split full \
  --mode direct --model "${VISION_MODEL}" --output-root "${OUT}/direct_lvlm/raw" \
  --max-image-size 512 --gen-max 128

for mode in text caption; do
  build_args=()
  [[ "${mode}" == "caption" ]] && build_args+=(--include-captions)
  reduced="${OUT}/flat_mm_${mode}/retrieved_dataset.json"
  run_step "build-flat-mm-${mode}" "${OUT}/flat_mm_${mode}/.dataset.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/build_multimodal_rag_dataset.py" --dataset "${DATASET}" \
    --output "${reduced}" --trace "${OUT}/flat_mm_${mode}/retrieval_trace.json" \
    --top-k "${TOP_K}" "${build_args[@]}"
  run_step "baseline-flat-mm-${mode}" "${OUT}/flat_mm_${mode}/.run.complete" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${ROOT}/memlens_repro/MEMLENS:${PYTHONPATH:-}" \
    "${PYTHON}" "${ROOT}/memlens_repro/MEMLENS/eval.py" --model_name_or_path "${VISION_MODEL}" \
    --input_file "${reduced}" --image_dir "${IMAGE_DIR}" --output_dir "${OUT}/flat_mm_${mode}/raw" \
    --input_max_length 32768 --generation_max_length 128 --attn_implementation sdpa \
    --dtype bfloat16 --device_map auto --max_image_size 512 --clear_cache_every 1 --overwrite \
    "${vlm_quant[@]}"
done

for method in direct_lvlm flat_mm_text flat_mm_caption; do
  run_step "normalize-${method}" "${OUT}/${method}/.normalize.complete" \
    "${PYTHON}" "${EXP_ROOT}/scripts/normalize_memlens_output.py" \
    --input-dir "${OUT}/${method}/raw" --output "${OUT}/${method}/predictions.json" --expected-count 789
done

for method in bm25_text caption_rag direct_lvlm flat_mm_text flat_mm_caption; do
  run_step "rescore-${method}" "${OUT}/${method}/.eval_v21.complete" \
    "${PYTHON}" "${ROOT}/typed_evidence/scripts/eval_v21.py" \
    --dataset "${DATASET}" --predictions "${OUT}/${method}/predictions.json" \
    --output "${OUT}/${method}/predictions_v21.json"
  compare_predictions "${DATASET}" "${OUT}/${method}/predictions_v21.json" "${P_OUT}" \
    "${OUT}/comparisons/p1_vs_${method}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" "${EXP_ROOT}/scripts/summarize_baselines.py" --results-root "${OUT}" \
    --ours "${P_OUT}" --output "${OUT}/paper_strong_baselines.csv"
fi
echo "Strong-baseline evaluation complete: ${OUT}"
