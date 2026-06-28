"""Create a paired eval_v2.1 comparison of two full prediction payloads."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from eval_v21 import load_items, load_json, score_row


def exact_mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value conditional on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def payload_rows(path: str | Path, items: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    data = payload.get("data", payload if isinstance(payload, list) else [])
    rows = {str(row["question_id"]): score_row(row, items[str(row["question_id"])]) for row in data}
    if set(rows) != set(items):
        missing = sorted(set(items) - set(rows))
        extra = sorted(set(rows) - set(items))
        raise ValueError(f"Prediction coverage mismatch: missing={missing}, extra={extra}")
    return rows


def summarize(qids: Iterable[str], baseline: Dict[str, Dict[str, Any]], candidate: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ids = list(qids)
    wins = sum(not baseline[qid]["sub_em_v21"] and candidate[qid]["sub_em_v21"] for qid in ids)
    losses = sum(baseline[qid]["sub_em_v21"] and not candidate[qid]["sub_em_v21"] for qid in ids)
    both_correct = sum(baseline[qid]["sub_em_v21"] and candidate[qid]["sub_em_v21"] for qid in ids)
    both_wrong = len(ids) - wins - losses - both_correct
    base_correct = losses + both_correct
    candidate_correct = wins + both_correct
    base_f1 = sum(float(baseline[qid]["f1_v21"]) for qid in ids) / len(ids) if ids else 0.0
    candidate_f1 = sum(float(candidate[qid]["f1_v21"]) for qid in ids) / len(ids) if ids else 0.0
    return {
        "count": len(ids), "baseline_correct": base_correct, "candidate_correct": candidate_correct,
        "baseline_sub_em_v21": base_correct / len(ids) if ids else 0.0,
        "candidate_sub_em_v21": candidate_correct / len(ids) if ids else 0.0,
        "delta_sub_em_points": 100.0 * (candidate_correct - base_correct) / len(ids) if ids else 0.0,
        "baseline_f1_v21": base_f1,
        "candidate_f1_v21": candidate_f1,
        "delta_f1_points": 100.0 * (candidate_f1 - base_f1),
        "wins": wins, "losses": losses, "net_wins": wins - losses,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "discordant_pairs": wins + losses,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(wins, losses),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Paired candidate comparison", "",
        f"Baseline: `{report['baseline']}`  ", f"Candidate: `{report['candidate']}`", "",
        "| Scope | n | Baseline correct | Candidate correct | EM delta (pp) | F1 delta (pp) | Wins | Losses | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("overall", report["overall"]), *report["by_subtype"].items()]:
        lines.append(
            f"| {name} | {row['count']} | {row['baseline_correct']} | {row['candidate_correct']} | "
            f"{row['delta_sub_em_points']:+.2f} | {row['delta_f1_points']:+.2f} | "
            f"{row['wins']} | {row['losses']} | {row['mcnemar_exact_two_sided_p']:.4f} |"
        )
    lines += ["", "Subtype p-values are descriptive and uncorrected; the pre-registered primary test is overall.",
              "A win is baseline-wrong/candidate-correct; a loss is baseline-correct/candidate-wrong under eval_v2.1.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    items = {str(item["question_id"]): item for item in load_items(args.dataset)}
    baseline, candidate = payload_rows(args.baseline, items), payload_rows(args.candidate, items)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for qid, item in items.items():
        grouped[str(item.get("question_subtype") or "unknown")].append(qid)
    report = {
        "evaluator": "eval_v2.1", "primary_test": "overall paired exact McNemar",
        "baseline": str(Path(args.baseline)), "candidate": str(Path(args.candidate)),
        "overall": summarize(items, baseline, candidate),
        "by_subtype": {name: summarize(qids, baseline, candidate) for name, qids in sorted(grouped.items())},
    }
    output_json, output_md = Path(args.output_json), Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()

