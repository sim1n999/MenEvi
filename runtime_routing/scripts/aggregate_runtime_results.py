"""Aggregate Runtime-routing evaluation offline and runtime variants with evaluator v2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from aggregate_eval_v2 import add_rows, metrics_from_payload
from eval_v2 import load_prediction_payload, save_json, write_csv


METHODS = [
    ("Offline hybrid routing (post-hoc)", "hybrid_subtype_routing"),
    ("Cross-validated hybrid routing", "hybrid_cross_validated_routing"),
    ("Runtime answer/visual routing", "runtime_cd"),
    ("Runtime specialists", "runtime_specialists"),
    ("Runtime tools", "runtime_specialists_override"),
]


def main() -> None:
    result_root = PROJECT_ROOT / "runtime_routing" / "results"
    out_dir = result_root / "runtime_comparison_v2"
    summary_rows: List[Dict[str, Any]] = []
    type_rows: List[Dict[str, Any]] = []
    subtype_rows: List[Dict[str, Any]] = []
    included = []

    for name, directory in METHODS:
        result_dir = result_root / directory
        if not (result_dir / "predictions.json").is_file():
            continue
        payload = load_prediction_payload(result_dir)
        payload["source_file"] = str(result_dir / "predictions.json")
        add_rows(name, metrics_from_payload(payload), summary_rows, type_rows, subtype_rows, payload)
        included.append(str(result_dir.relative_to(PROJECT_ROOT)))

    write_csv(out_dir / "method_summary.csv", summary_rows)
    write_csv(out_dir / "by_question_type.csv", type_rows)
    write_csv(out_dir / "by_question_subtype.csv", subtype_rows)
    save_json(out_dir / "aggregate_manifest.json", {"included": included})
    print(f"Wrote {len(included)} Runtime-routing evaluation variants to {out_dir}")


if __name__ == "__main__":
    main()
