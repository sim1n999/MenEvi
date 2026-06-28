#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_ROOT="${ROOT}/reliability_gate"
HOLDOUT_ROOT="${ROOT}/holdout_validation"
TYPED_ROOT="${ROOT}/typed_evidence"
PYTHON="${PYTHON:-python}"

DATASET="${ROOT}/memlens_repro/data/memlens_agent_subset/dataset_32k.json"
BASE="${ROOT}/runtime_routing/results/runtime_specialists/predictions.json"
HOLDOUT_RESULTS="${HOLDOUT_ROOT}/results/dev_195"
VISUAL_PREDICTIONS="${HOLDOUT_RESULTS}/typed_specialists/predictions.json"
OBSERVATIONS="${HOLDOUT_RESULTS}/visual_inspection/observations"
RESULTS="${GATE_ROOT}/results/dev_195/visual_reliability_gate"
OUTPUT="${RESULTS}/predictions.json"
DECISIONS="${RESULTS}/decisions.json"

for required in "${DATASET}" "${BASE}" "${VISUAL_PREDICTIONS}" "${OBSERVATIONS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing reliability-gate development prerequisite: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" -m pytest "${GATE_ROOT}/tests" -q

"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${HOLDOUT_RESULTS}/visual_prompt_audit/requests" \
  --report "${RESULTS}/source_visual_leakage_audit.json"
"${PYTHON}" "${HOLDOUT_ROOT}/scripts/audit_prompt_leakage.py" \
  --prompt-dir "${HOLDOUT_RESULTS}/specialist_prompt_audit/prompts" \
  --report "${RESULTS}/source_specialist_leakage_audit.json"

"${PYTHON}" "${GATE_ROOT}/scripts/apply_reliability_gate.py" \
  --dataset "${DATASET}" \
  --base "${BASE}" \
  --visual-predictions "${VISUAL_PREDICTIONS}" \
  --observation-dir "${OBSERVATIONS}" \
  --output "${OUTPUT}" \
  --decisions-output "${DECISIONS}" \
  --require-all-visual-targets

"${PYTHON}" "${TYPED_ROOT}/scripts/compare_predictions.py" \
  --dataset "${DATASET}" \
  --baseline "${BASE}" \
  --candidate "${OUTPUT}" \
  --output-json "${RESULTS}/comparison.json" \
  --output-md "${RESULTS}/COMPARISON.md"

"${PYTHON}" "${GATE_ROOT}/scripts/paired_stats.py" \
  --dataset "${DATASET}" \
  --baseline "${BASE}" \
  --candidate "${OUTPUT}" \
  --output "${RESULTS}/paired_bootstrap.json"

echo "Reliability-gate development diagnostic completed: ${RESULTS}"



