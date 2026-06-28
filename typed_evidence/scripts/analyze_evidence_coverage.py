"""Post-hoc evidence coverage audit.

This script may read references and answer_session_ids, so its output is
diagnostic only and must never be consumed by an inference runner.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

CURRENCY_RE = re.compile(r"([$£€])\s*(\d+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def items(path: str | Path) -> List[Dict[str, Any]]:
    value = load(path)
    return value.get("data", value) if isinstance(value, dict) else value


def reference_amount(value: Any) -> Tuple[Optional[str], Optional[int]]:
    text = str(value or "")
    match = CURRENCY_RE.search(text)
    number = match.group(2) if match else (NUMBER_RE.search(text).group(0) if NUMBER_RE.search(text) else None)
    if number is None:
        return None, None
    return (match.group(1) if match else None), int(Decimal(number) * 100)


def subset_sum(values: List[int], target: int) -> bool:
    reachable: Set[int] = {0}
    for value in values:
        reachable |= {subtotal + value for subtotal in tuple(reachable) if subtotal + value <= target}
        if target in reachable:
            return True
    return target in reachable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--visual-k", type=int, default=3)
    parser.add_argument("--duration-limit", type=int, default=18)
    args = parser.parse_args()

    rows = []
    for item in items(args.dataset):
        subtype = item.get("question_subtype")
        if subtype not in {"arithmetic", "duration_comparison", "entity", "previnfo"}:
            continue
        qid = str(item["question_id"])
        packet = load(Path(args.packet_dir) / f"{qid}.json")
        evidence = packet["typed_evidence"]
        expected_sessions = set(map(str, item.get("answer_session_ids", [])))
        row = {"question_id": qid, "question_subtype": subtype}

        if subtype == "arithmetic":
            currency, target = reference_amount(item.get("answer"))
            candidates = evidence.get("events", [])[:args.candidate_limit]
            values = []
            for event in candidates:
                if currency and event.get("currency") != currency:
                    continue
                values.append(int(Decimal(str(event["amount"])) * 100))
            row.update({
                "reference_candidate_sum_reachable": bool(target is not None and subset_sum(values, target)),
                "candidate_count": len(candidates),
                "completed_candidate_count": sum(event.get("status") == "completed" for event in candidates),
            })
        elif subtype in {"entity", "previnfo"}:
            targets = evidence.get("targets", [])
            top_one = {str(row.get("session_id")) for row in targets[:1]}
            top_k = {str(row.get("session_id")) for row in targets[:args.visual_k]}
            row.update({
                "reference_session_count": len(expected_sessions),
                "visual_session_recall_at_1": bool(expected_sessions & top_one),
                "visual_session_recall_at_k": bool(expected_sessions & top_k),
            })
        else:
            observed = set()
            for duration in evidence.get("durations", []):
                observed |= {
                    str(boundary.get("session_id"))
                    for boundary in duration.get("boundaries", [])[:args.duration_limit]
                }
            hit = len(expected_sessions & observed)
            row.update({
                "reference_session_count": len(expected_sessions),
                "recalled_reference_sessions": hit,
                "reference_session_recall": hit / len(expected_sessions) if expected_sessions else None,
            })
        rows.append(row)

    arithmetic = [row for row in rows if row["question_subtype"] == "arithmetic"]
    visual = [row for row in rows if row["question_subtype"] in {"entity", "previnfo"}]
    duration = [row for row in rows if row["question_subtype"] == "duration_comparison"]
    summary = {
        "diagnostic_only": True,
        "must_not_feed_inference": True,
        "arithmetic": {
            "count": len(arithmetic),
            "candidate_sum_recall_count": sum(row["reference_candidate_sum_reachable"] for row in arithmetic),
            "candidate_sum_recall": sum(row["reference_candidate_sum_reachable"] for row in arithmetic) / len(arithmetic) if arithmetic else 0,
        },
        "visual": {
            "count": len(visual),
            "reference_session_recall_at_1": sum(row["visual_session_recall_at_1"] for row in visual) / len(visual) if visual else 0,
            "reference_session_recall_at_k": sum(row["visual_session_recall_at_k"] for row in visual) / len(visual) if visual else 0,
            "k": args.visual_k,
        },
        "duration": {
            "count": len(duration),
            "mean_reference_session_recall": (
                sum(row["reference_session_recall"] or 0 for row in duration) / len(duration) if duration else 0
            ),
        },
    }
    output = {"args": vars(args), "summary": summary, "data": rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
