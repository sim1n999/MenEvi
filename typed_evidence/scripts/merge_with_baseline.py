"""Merge successful candidate predictions into a frozen baseline payload."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_v21 import compute_metrics, load_items, load_json, save_json, score_row

TARGET_SUBTYPES = {"arithmetic", "duration_comparison", "entity", "previnfo"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-all-targets", action="store_true")
    parser.add_argument(
        "--include-subtypes",
        nargs="+",
        choices=sorted(TARGET_SUBTYPES),
        default=sorted(TARGET_SUBTYPES),
        help="Only these target subtypes may replace frozen baseline predictions.",
    )
    args = parser.parse_args()

    items = {str(item["question_id"]): item for item in load_items(args.dataset)}
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    candidate_data = candidate.get("data", candidate if isinstance(candidate, list) else [])
    candidate_rows = {str(row["question_id"]): row for row in candidate_data}
    target_ids = {
        qid for qid, item in items.items()
        if item.get("question_subtype") in TARGET_SUBTYPES
    }
    extra = sorted(set(candidate_rows) - target_ids)
    included_subtypes = set(args.include_subtypes)
    included_target_ids = {
        qid for qid, item in items.items()
        if item.get("question_subtype") in included_subtypes
    }
    if extra:
        raise ValueError(f"Candidate payload contains non-target questions: {extra}")
    if args.require_all_targets and not included_target_ids.issubset(candidate_rows):
        missing = sorted(included_target_ids - set(candidate_rows))
        raise ValueError(f"Candidate target coverage mismatch: missing={missing}")

    baseline_data = baseline.get("data", baseline if isinstance(baseline, list) else [])
    if {str(row["question_id"]) for row in baseline_data} != set(items):
        raise ValueError("Frozen baseline payload does not cover the dataset exactly")

    merged, replacements = [], []
    for old in baseline_data:
        qid = str(old["question_id"])
        new = candidate_rows.get(qid)
        eligible = bool(
            qid in included_target_ids
            and new
            and new.get("eligible_for_merge", True)
            and str(new.get("raw_prediction", "")).strip()
        )
        if eligible:
            row = {
                **old,
                "raw_prediction": new["raw_prediction"],
                "input_len": new.get("input_len", old.get("input_len")),
                "output_len": new.get("output_len", old.get("output_len")),
                "route_source": "typed_candidate",
                "route_reason": "typed specialist replacement",
                "route_policy": "typed_specialist_override",
                "execution_mode": new.get("execution_mode"),
                "candidate_trace": new.get("tool_trace"),
            }
            replacements.append({
                "question_id": qid,
                "question_subtype": old.get("question_subtype"),
                "baseline_prediction": old.get("raw_prediction"),
                "candidate_prediction": new.get("raw_prediction"),
            })
        else:
            row = dict(old)
        merged.append(score_row(row, items[qid]))

    payload = {
        "args": vars(args),
        "data": merged,
        "metrics_v21": compute_metrics(merged),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "included_subtypes": sorted(included_subtypes),
        "frozen_base": str(Path(args.baseline)),
        "evaluator": "eval_v2.1",
    }
    save_json(args.output, payload)
    print(f"Merged {len(replacements)} candidate predictions; wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
