"""Apply configurable, label-blind reliability-gate ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPTS = PROJECT_ROOT / "reliability_gate" / "scripts"
TYPED_SCRIPTS = PROJECT_ROOT / "typed_evidence" / "scripts"
for path in (GATE_SCRIPTS, TYPED_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apply_reliability_gate import directly_supported, is_refusal, load_rank1_observation, normalize, payload_rows  # noqa: E402
from eval_v21 import compute_metrics, load_items, load_json, save_json, score_row  # noqa: E402


def decide(base: Any, candidate: Dict[str, Any], observation: Dict[str, Any], reject_refusal: bool, preserve_support: bool) -> Dict[str, Any]:
    candidate_prediction = str(candidate.get("raw_prediction", "")).strip()
    base_support = directly_supported(base, observation)
    candidate_support = directly_supported(candidate_prediction, observation)
    if not candidate.get("eligible_for_merge", True):
        replace, reason = False, "candidate_marked_ineligible"
    elif not normalize(candidate_prediction):
        replace, reason = False, "empty_candidate"
    elif reject_refusal and is_refusal(candidate_prediction):
        replace, reason = False, "normalized_refusal"
    elif preserve_support and base_support and not candidate_support:
        replace, reason = False, "preserve_rank1_supported_base"
    else:
        replace, reason = True, "replace_with_visual_candidate"
    return {
        "replace": replace, "reason": reason,
        "base_prediction": str(base or ""), "candidate_prediction": candidate_prediction,
        "base_rank1_support": base_support, "candidate_rank1_support": candidate_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--visual-predictions", required=True)
    parser.add_argument("--observation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions-output", required=True)
    parser.add_argument("--reject-refusal", action="store_true")
    parser.add_argument("--preserve-supported-base", action="store_true")
    args = parser.parse_args()

    items = {str(x["question_id"]): x for x in load_items(args.dataset)}
    base_rows = payload_rows(load_json(args.base))
    candidates = {str(x["question_id"]): x for x in payload_rows(load_json(args.visual_predictions))}
    if {str(x["question_id"]) for x in base_rows} != set(items):
        raise ValueError("Base predictions do not exactly cover the dataset")

    merged: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for old in base_rows:
        qid = str(old["question_id"])
        subtype = str(items[qid].get("question_subtype"))
        candidate = candidates.get(qid) if subtype in {"entity", "previnfo"} else None
        if candidate is None:
            merged.append(score_row(dict(old), items[qid]))
            continue
        observation = load_rank1_observation(Path(args.observation_dir), qid)
        decision = decide(old.get("raw_prediction"), candidate, observation, args.reject_refusal, args.preserve_supported_base)
        decision.update({"question_id": qid, "question_subtype": subtype})
        decisions.append(decision)
        row = dict(old)
        if decision["replace"]:
            row.update({
                "raw_prediction": candidate.get("raw_prediction", ""),
                "input_len": candidate.get("input_len", old.get("input_len")),
                "output_len": candidate.get("output_len", old.get("output_len")),
                "route_source": "gate_ablation",
                "route_reason": decision["reason"],
            })
        merged.append(score_row(row, items[qid]))

    save_json(args.output, {
        "args": vars(args), "data": merged, "metrics_v21": compute_metrics(merged),
        "replacement_count": sum(x["replace"] for x in decisions),
        "uses_reference_labels_for_decisions": False,
    })
    save_json(args.decisions_output, {"args": vars(args), "decisions": decisions})


if __name__ == "__main__":
    main()



