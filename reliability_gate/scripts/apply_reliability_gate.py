"""Apply the preregistered Reliability-gate validation visual reliability gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPED_SCRIPTS = (
    PROJECT_ROOT
    / "typed_evidence"
    / "scripts"
)
if str(TYPED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TYPED_SCRIPTS))

from eval_v21 import compute_metrics, load_items, load_json, save_json, score_row  # noqa: arithmetic repair02


VISUAL_SUBTYPES = {"entity", "previnfo"}
REFUSAL = "insufficient information"
TOKEN_RE = re.compile(r"\w+|[$£€]+", flags=re.UNICODE)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(TOKEN_RE.findall(text)).replace("_", " ").strip()


def is_refusal(value: Any) -> bool:
    return normalize(value) == REFUSAL


def serialize_observation(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def directly_supported(answer: Any, observation: Any) -> bool:
    answer_norm = normalize(answer)
    observation_norm = normalize(serialize_observation(observation))
    if not answer_norm or not observation_norm:
        return False
    return f" {answer_norm} " in f" {observation_norm} "


def payload_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("data", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    if not isinstance(rows, list):
        raise TypeError("Prediction payload data must be a list")
    return rows


def load_rank1_observation(directory: Path, qid: str) -> Dict[str, Any]:
    path = directory / f"{qid}.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        return {}
    first = rows[0]
    return first.get("observation", {}) if isinstance(first, dict) else {}


def gate_decision(
    base_prediction: Any,
    candidate: Dict[str, Any],
    rank1_observation: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_prediction = str(candidate.get("raw_prediction", "")).strip()
    base_support = directly_supported(base_prediction, rank1_observation)
    candidate_support = directly_supported(
        candidate_prediction,
        rank1_observation,
    )

    if not candidate.get("eligible_for_merge", True):
        reason = "candidate_marked_ineligible"
        replace = False
    elif not normalize(candidate_prediction):
        reason = "empty_candidate"
        replace = False
    elif is_refusal(candidate_prediction):
        reason = "normalized_refusal"
        replace = False
    elif base_support and not candidate_support:
        reason = "preserve_rank1_supported_base"
        replace = False
    else:
        reason = "replace_with_visual_candidate"
        replace = True

    return {
        "replace": replace,
        "reason": reason,
        "base_prediction": str(base_prediction or ""),
        "candidate_prediction": candidate_prediction,
        "base_rank1_support": base_support,
        "candidate_rank1_support": candidate_support,
        "rank1_observation_normalized": normalize(
            serialize_observation(rank1_observation)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--visual-predictions", required=True)
    parser.add_argument("--observation-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions-output", required=True)
    parser.add_argument("--require-all-visual-targets", action="store_true")
    args = parser.parse_args()

    items = {
        str(item["question_id"]): item
        for item in load_items(args.dataset)
    }
    base_payload = load_json(args.base)
    candidate_payload = load_json(args.visual_predictions)
    base_rows = payload_rows(base_payload)
    candidate_rows = payload_rows(candidate_payload)
    base_ids = {str(row["question_id"]) for row in base_rows}
    if base_ids != set(items):
        raise ValueError("Base predictions do not cover the dataset exactly")

    visual_ids = {
        qid
        for qid, item in items.items()
        if item.get("question_subtype") in VISUAL_SUBTYPES
    }
    candidates = {
        str(row["question_id"]): row
        for row in candidate_rows
        if row.get("question_subtype") in VISUAL_SUBTYPES
    }
    extra = sorted(set(candidates) - visual_ids)
    if extra:
        raise ValueError(f"Unexpected visual prediction IDs: {extra}")
    if args.require_all_visual_targets and set(candidates) != visual_ids:
        missing = sorted(visual_ids - set(candidates))
        raise ValueError(f"Missing visual prediction IDs: {missing}")

    observation_dir = Path(args.observation_dir)
    merged: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    replacements = []

    for old in base_rows:
        qid = str(old["question_id"])
        subtype = str(items[qid].get("question_subtype"))
        new = candidates.get(qid)
        if subtype not in VISUAL_SUBTYPES or new is None:
            merged.append(score_row(dict(old), items[qid]))
            continue

        rank1 = load_rank1_observation(observation_dir, qid)
        decision = gate_decision(old.get("raw_prediction"), new, rank1)
        decision.update(
            {
                "question_id": qid,
                "question_subtype": subtype,
            }
        )
        decisions.append(decision)

        if decision["replace"]:
            row = {
                **old,
                "raw_prediction": new["raw_prediction"],
                "input_len": new.get("input_len", old.get("input_len")),
                "output_len": new.get("output_len", old.get("output_len")),
                "route_source": "I",
                "route_reason": decision["reason"],
                "route_policy": "visual_reliability_gate_v1",
                "execution_mode": new.get("execution_mode"),
                "i_trace": {
                    key: value
                    for key, value in decision.items()
                    if key not in {"question_id", "question_subtype"}
                },
            }
            replacements.append(
                {
                    "question_id": qid,
                    "question_subtype": subtype,
                    "base_prediction": old.get("raw_prediction"),
                    "i_prediction": new.get("raw_prediction"),
                    "gate_reason": decision["reason"],
                }
            )
        else:
            row = dict(old)
        merged.append(score_row(row, items[qid]))

    non_visual_changes = []
    base_by_id = {str(row["question_id"]): row for row in base_rows}
    for row in merged:
        qid = str(row["question_id"])
        if (
            items[qid].get("question_subtype") not in VISUAL_SUBTYPES
            and str(row.get("raw_prediction", ""))
            != str(base_by_id[qid].get("raw_prediction", ""))
        ):
            non_visual_changes.append(qid)
    if non_visual_changes:
        raise AssertionError(
            f"Non-visual predictions changed: {non_visual_changes}"
        )

    save_json(
        args.output,
        {
            "args": vars(args),
            "data": merged,
            "metrics_v21": compute_metrics(merged),
            "replacement_count": len(replacements),
            "replacements": replacements,
            "gate": {
                "normalized_refusal": REFUSAL,
                "preserve_base_when_rank1_supports_base_only": True,
                "uses_reference_labels": False,
            },
            "frozen_base": str(Path(args.base)),
            "evaluator": "eval_v2.1",
        },
    )
    save_json(
        args.decisions_output,
        {
            "args": vars(args),
            "decision_count": len(decisions),
            "replacement_count": len(replacements),
            "decisions": decisions,
            "uses_reference_labels": False,
        },
    )
    print(
        f"Applied I gate to {len(decisions)} visual targets; "
        f"merged {len(replacements)} candidates into {len(merged)} rows"
    )


if __name__ == "__main__":
    main()

