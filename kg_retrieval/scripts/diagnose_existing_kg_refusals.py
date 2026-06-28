"""Diagnose current hard-refusal KG outputs and write KG retrieval baseline targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from local_memlens_utils import (
    is_refusal,
    load_items,
    load_prediction_payload,
    read_jsonl,
    save_json,
    write_csv,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--kg-output-dir", required=True)
    parser.add_argument("--retrieval-log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = {x.get("question_id"): x for x in load_items(args.input)}
    payload = load_prediction_payload(args.kg_output_dir)
    predictions = payload.get("data", [])
    retrieval = {x.get("question_id"): x for x in read_jsonl(args.retrieval_log)}
    out_dir = Path(args.output_dir)

    rows = []
    by_type = defaultdict(lambda: {"count": 0, "refused": 0, "wrong_refused": 0, "hit_refused": 0})
    for pred in predictions:
        qid = pred.get("question_id")
        item = dataset.get(qid, {})
        log = retrieval.get(qid, {})
        qtype = pred.get("question_type") or item.get("question_type")
        refused = bool(pred.get("is_refusal", is_refusal(pred.get("prediction", pred.get("raw_prediction", "")))))
        correct = bool(pred.get("sub_em"))
        answerable = qtype != "answer_refusal"
        hit = log.get("session_hit")
        wrong_refusal = answerable and refused and not correct
        if answerable:
            by_type[qtype]["count"] += 1
            by_type[qtype]["refused"] += int(refused)
            by_type[qtype]["wrong_refused"] += int(wrong_refusal)
            by_type[qtype]["hit_refused"] += int(bool(hit) and wrong_refusal)
        if wrong_refusal:
            rows.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "question": pred.get("question"),
                    "reference_answer": pred.get("reference_answer"),
                    "prediction": pred.get("prediction"),
                    "answer_session_ids": item.get("answer_session_ids", []),
                    "retrieved_session_ids": log.get("retrieved_session_ids", []),
                    "session_hit": hit,
                    "session_all_hit": log.get("session_all_hit"),
                    "node_count": log.get("node_count"),
                    "edge_count": log.get("edge_count"),
                }
            )

    summary = []
    total = {"count": 0, "refused": 0, "wrong_refused": 0, "hit_refused": 0}
    for qtype, stats in sorted(by_type.items()):
        for key, value in stats.items():
            total[key] += value
        summary.append(
            {
                "question_type": qtype,
                **stats,
                "wrong_refusal_rate": stats["wrong_refused"] / stats["count"] if stats["count"] else 0.0,
                "hit_wrong_refusal_rate": stats["hit_refused"] / stats["count"] if stats["count"] else 0.0,
            }
        )
    summary.append(
        {
            "question_type": "ALL_ANSWERABLE",
            **total,
            "wrong_refusal_rate": total["wrong_refused"] / total["count"] if total["count"] else 0.0,
            "hit_wrong_refusal_rate": total["hit_refused"] / total["count"] if total["count"] else 0.0,
        }
    )

    write_csv(out_dir / "wrong_refusal_cases.csv", rows)
    write_jsonl(out_dir / "wrong_refusal_cases.jsonl", rows)
    write_csv(out_dir / "wrong_refusal_summary.csv", summary)
    save_json(
        out_dir / "diagnosis.json",
        {
            "kg_output_dir": args.kg_output_dir,
            "retrieval_log": args.retrieval_log,
            "wrong_refusal_cases": len(rows),
            "summary": summary,
        },
    )
    print(f"Wrote KG refusal diagnosis to {out_dir}")


if __name__ == "__main__":
    main()
