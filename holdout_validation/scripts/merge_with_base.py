"""Merge eligible candidate predictions into a frozen baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPED_SCRIPTS = (
    PROJECT_ROOT
    / "typed_evidence"
    / "scripts"
)
if str(TYPED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TYPED_SCRIPTS))

from eval_v21 import compute_metrics, load_items, load_json, save_json, score_row  # noqa: E402


TARGET_SUBTYPES = {"arithmetic", "duration_comparison", "entity", "previnfo"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-all-targets", action="store_true")
    parser.add_argument(
        "--include-subtypes",
        nargs="+",
        choices=sorted(TARGET_SUBTYPES),
        default=sorted(TARGET_SUBTYPES),
    )
    args = parser.parse_args()

    items = {
        str(item["question_id"]): item
        for item in load_items(args.dataset)
    }
    base_payload = load_json(args.base)
    candidate_payload = load_json(args.candidate)
    base_rows = base_payload.get(
        "data",
        base_payload if isinstance(base_payload, list) else [],
    )
    candidate_data = candidate_payload.get(
        "data",
        candidate_payload if isinstance(candidate_payload, list) else [],
    )
    candidate_rows = {str(row["question_id"]): row for row in candidate_data}

    included = set(args.include_subtypes)
    all_target_ids = {
        qid
        for qid, item in items.items()
        if item.get("question_subtype") in TARGET_SUBTYPES
    }
    included_ids = {
        qid
        for qid, item in items.items()
        if item.get("question_subtype") in included
    }
    extra = sorted(set(candidate_rows) - all_target_ids)
    if extra:
        raise ValueError(f"Candidate payload contains non-target questions: {extra}")
    if args.require_all_targets and not included_ids.issubset(candidate_rows):
        missing = sorted(included_ids - set(candidate_rows))
        raise ValueError(f"Candidate target coverage mismatch: {missing}")

    base_ids = {str(row["question_id"]) for row in base_rows}
    if base_ids != set(items):
        raise ValueError("Frozen baseline does not cover the dataset exactly")

    merged = []
    replacements = []
    for old in base_rows:
        qid = str(old["question_id"])
        new = candidate_rows.get(qid)
        eligible = bool(
            qid in included_ids
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
                "route_source": "holdout_candidate",
                "route_reason": "Leakage-safe typed specialist replacement",
                "route_policy": "holdout_typed_override",
                "execution_mode": new.get("execution_mode"),
                "candidate_trace": new.get("tool_trace"),
            }
            replacements.append(
                {
                    "question_id": qid,
                    "question_subtype": old.get("question_subtype"),
                    "base_prediction": old.get("raw_prediction"),
                    "candidate_prediction": new.get("raw_prediction"),
                }
            )
        else:
            row = dict(old)
        merged.append(score_row(row, items[qid]))

    save_json(
        args.output,
        {
            "args": vars(args),
            "data": merged,
            "metrics_v21": compute_metrics(merged),
            "replacement_count": len(replacements),
            "replacements": replacements,
            "included_subtypes": sorted(included),
            "frozen_base": str(Path(args.base)),
            "evaluator": "eval_v2.1",
        },
    )
    print(
        f"Merged {len(replacements)} candidate predictions; "
        f"wrote {len(merged)} rows to {args.output}"
    )


if __name__ == "__main__":
    main()

