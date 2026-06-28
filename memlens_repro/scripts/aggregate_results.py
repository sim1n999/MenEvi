"""Aggregate MemLens experiment outputs into CSV tables and error cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import load_items, result_file_from_dir, save_json, write_csv, write_jsonl
import json


def method_name(path: Path) -> str:
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-dirs", nargs="+", required=True)
    args = parser.parse_args()

    dataset = {x.get("question_id"): x for x in load_items(args.input)}
    summary_rows = []
    type_rows = []
    error_rows = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rd in args.result_dirs:
        rd_path = Path(rd)
        rf = result_file_from_dir(rd_path)
        if not rf:
            continue
        payload = json.loads(rf.read_text(encoding="utf-8"))
        results = payload.get("data", payload if isinstance(payload, list) else [])
        metrics = payload.get("metrics", {})
        name = method_name(rd_path)
        overall = metrics.get("overall", {})
        ans = metrics.get("answerable", {})
        abst = metrics.get("abstention", {})
        avg = payload.get("averaged_metrics", {})
        summary_rows.append(
            {
                "method": name,
                "n": overall.get("count", len(results)),
                "overall_subem": overall.get("sub_em"),
                "overall_f1": overall.get("f1"),
                "answerable_subem": ans.get("sub_em"),
                "abstention_accuracy": abst.get("accuracy"),
                "refusal_rate": overall.get("refusal_rate"),
                "avg_input_len": avg.get("input_len"),
                "avg_output_len": avg.get("output_len"),
                "source_file": str(rf),
            }
        )
        for qtype, row in metrics.get("by_question_type", {}).items():
            type_rows.append(
                {
                    "method": name,
                    "question_type": qtype,
                    "subem": row.get("sub_em"),
                    "f1": row.get("f1"),
                    "refusal_rate": row.get("refusal_rate"),
                    "count": row.get("count"),
                    "subem_count": row.get("sub_em_count"),
                }
            )
        for r in results:
            if not r.get("sub_em"):
                qid = r.get("question_id")
                src = dataset.get(qid, {})
                error_rows.append(
                    {
                        "method": name,
                        "question_id": qid,
                        "question_type": r.get("question_type"),
                        "question": r.get("question"),
                        "reference_answer": r.get("reference_answer"),
                        "prediction": r.get("prediction"),
                        "answer_session_ids": src.get("answer_session_ids", []),
                    }
                )

    write_csv(out_dir / "summary_metrics.csv", summary_rows)
    write_csv(out_dir / "per_type_metrics.csv", type_rows)
    write_jsonl(out_dir / "error_cases.jsonl", error_rows)
    save_json(out_dir / "aggregate_manifest.json", {"result_dirs": args.result_dirs})
    print(f"Wrote aggregate tables to {out_dir}")


if __name__ == "__main__":
    main()
