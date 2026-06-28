#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPRO_ROOT}/configs/paths.env" ]]; then
  # shellcheck source=/dev/null
  source "${REPRO_ROOT}/configs/paths.env"
elif [[ -f "${REPRO_ROOT}/configs/paths.env.example" ]]; then
  # shellcheck source=/dev/null
  source "${REPRO_ROOT}/configs/paths.env.example"
fi

CODE_DIR="$(cd "${REPRO_ROOT}/${MEMLENS_CODE_DIR:-./MEMLENS}" && pwd)"
DATA_ROOT="$(cd "${REPRO_ROOT}/${MEMLENS_DATA_ROOT:-./data/memlens}" && pwd)"
AGENT_DATA_ROOT="${REPRO_ROOT}/${MEMLENS_AGENT_DATA_ROOT:-./data/memlens_agent_subset}"
IMAGE_DIR="$(cd "${REPRO_ROOT}/${MEMLENS_IMAGE_DIR:-./data/memlens/release_images}" && pwd)"
OUTPUT_ROOT="${REPRO_ROOT}/${MEMLENS_OUTPUT_ROOT:-./outputs}"
QWEN25_VL_PATH="${QWEN25_VL_PATH:-}"
[[ -n "${QWEN25_VL_PATH}" ]] || { echo "ERROR: QWEN25_VL_PATH is empty. Pass --model or export QWEN25_VL_PATH." >&2; exit 2; }
QWEN25_VL_PATH="$(cd "${QWEN25_VL_PATH}" && pwd)"

DATASET="32k"
SPLIT="agent"
MODE="direct"
MAX_SAMPLES=""
DRY_RUN=false
LOAD_IN_4BIT=true
MAX_IMAGE_SIZE="${MEMLENS_MAX_IMAGE_SIZE:-512}"
GEN_MAX="${MEMLENS_GENERATION_MAX_LENGTH:-128}"
ATTN_IMPL="${MEMLENS_ATTN_IMPL:-sdpa}"
DTYPE="${MEMLENS_DTYPE:-bfloat16}"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Run Qwen2.5-VL-7B-Instruct on MEMLENS with local HuggingFace Transformers.

Examples:
  bash scripts/run_qwen25vl_memlens.sh --dataset 32k --split agent --mode direct --max-samples 1
  bash scripts/run_qwen25vl_memlens.sh --dataset 32k --split agent --mode text_only
  bash scripts/run_qwen25vl_memlens.sh --dataset 32k --split agent --mode no_context

Options:
  --dataset 32k|64k|128k|256k
  --split agent|full              default: agent
  --mode direct|text_only|no_context|label_images
  --max-samples N                 smoke test / partial run
  --model PATH                    override Qwen2.5-VL path
  --output-root DIR
  --max-image-size N              default: 512
  --gen-max N                     default: 128
  --attn-impl NAME                default: sdpa
  --dtype bfloat16|float16        default: bfloat16
  --no-4bit                       disable 4-bit loading
  --extra "ARGS"                  extra eval.py args, appended at the end
  --dry-run                       print command only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --model) QWEN25_VL_PATH="$(cd "$2" && pwd)"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --max-image-size) MAX_IMAGE_SIZE="$2"; shift 2 ;;
    --gen-max) GEN_MAX="$2"; shift 2 ;;
    --attn-impl) ATTN_IMPL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --no-4bit) LOAD_IN_4BIT=false; shift ;;
    --extra) EXTRA_ARGS+=($2); shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

case "$DATASET" in
  32k) INPUT_MAX=32768 ;;
  64k) INPUT_MAX=65536 ;;
  128k) INPUT_MAX=131072 ;;
  256k) INPUT_MAX=262144 ;;
  *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

if [[ "$SPLIT" == "agent" ]]; then
  if [[ ! -f "${AGENT_DATA_ROOT}/dataset_${DATASET}.json" ]]; then
    python "${SCRIPT_DIR}/filter_agent_subset.py" \
      --data-root "$DATA_ROOT" \
      --out-root "$AGENT_DATA_ROOT"
  fi
  INPUT_FILE="${AGENT_DATA_ROOT}/dataset_${DATASET}.json"
elif [[ "$SPLIT" == "full" ]]; then
  INPUT_FILE="${DATA_ROOT}/dataset_${DATASET}.json"
else
  echo "Unknown split: $SPLIT"; exit 1
fi

RUN_ID="qwen25vl7b_${DATASET}_${SPLIT}_${MODE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"

CMD=(
  python "${CODE_DIR}/eval.py"
  --model_name_or_path "$QWEN25_VL_PATH"
  --input_file "$INPUT_FILE"
  --image_dir "$IMAGE_DIR"
  --output_dir "$OUTPUT_DIR"
  --input_max_length "$INPUT_MAX"
  --generation_max_length "$GEN_MAX"
  --attn_implementation "$ATTN_IMPL"
  --dtype "$DTYPE"
  --device_map auto
  --max_image_size "$MAX_IMAGE_SIZE"
  --clear_cache_every 1
  --overwrite
)

if [[ "$LOAD_IN_4BIT" == true ]]; then
  CMD+=(--load_in_4bit)
fi

case "$MODE" in
  direct) ;;
  text_only) CMD+=(--text_only) ;;
  no_context) CMD+=(--no_context) ;;
  label_images) CMD+=(--label_images) ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max_test_samples "$MAX_SAMPLES")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Repro root : $REPRO_ROOT"
echo "Code dir   : $CODE_DIR"
echo "Input file : $INPUT_FILE"
echo "Image dir  : $IMAGE_DIR"
echo "Model      : $QWEN25_VL_PATH"
echo "Output dir : $OUTPUT_DIR"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
"${CMD[@]}"
