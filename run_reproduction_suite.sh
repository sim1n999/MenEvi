#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="${TASKS:-length_curve strong_baselines automatic_routing component_ablations efficiency_profile}"

for task in ${TASKS}; do
  case "${task}" in
    length_curve) script="length_curve/run_all.sh" ;;
    strong_baselines) script="strong_baselines/run_all.sh" ;;
    automatic_routing) script="automatic_routing/run_all.sh" ;;
    component_ablations) script="component_ablations/run_all.sh" ;;
    efficiency_profile) script="efficiency_profile/run_all.sh" ;;
    *) echo "Unknown reproduction task: ${task}" >&2; exit 2 ;;
  esac
  echo "===== ${task} ====="
  bash "${ROOT}/${script}"
done