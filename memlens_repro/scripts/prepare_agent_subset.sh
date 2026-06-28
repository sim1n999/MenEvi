#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/filter_agent_subset.py" \
  --data-root "${REPRO_ROOT}/data/memlens" \
  --out-root "${REPRO_ROOT}/data/memlens_agent_subset"
