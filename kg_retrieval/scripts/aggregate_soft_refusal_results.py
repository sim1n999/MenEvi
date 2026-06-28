"""Aggregate KG retrieval baseline and baseline result directories into CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_memlens_utils import compute_metrics, load_prediction_payload, write_csv, save_json


SHORT_NAMES = {
    "kg_memory_32k_agent": "KG hard-refusal budget80",
    "qwen_vl_caption_rag_32k_agent": "Qwen-VL caption RAG",
    "oracle_evidence_qwen25vl_32k_agent": "Oracle evidence Qwen2.5-VL-7B",
    "kg_soft_refusal_budget80": "KG soft-refusal budget80",
    "kg_soft_refusal_budget120": "KG soft-refusal budget120",
}


def pct(x):
    return None if x is None else 100 * float(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-dirs", nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_rows = []
    type_rows = []
    for result_dir in args.result_dirs:
        path = Path(result_dir)
        payload = load_prediction_payload(path)
        results = payload.get("data", [])
        metrics = payload.get("metrics") or compute_metrics(results)
        local_metrics = compute_metrics(results)
        overall = metrics.get("overall", {})
        answerable = metrics.get("answerable", {})
        local_answerable = local_metrics.get("answerable", {})
        abstention = metrics.get("abstention", {})
        avg = payload.get("averaged_metrics", {})
        name = SHORT_NAMES.get(path.name, path.name)
        summary_rows.append(
            {
                "method": name,
                "count": overall.get("count"),
                "overall_sub_em": pct(overall.get("sub_em")),
                "overall_f1": pct(overall.get("f1")),
                "answerable_sub_em": pct(answerable.get("sub_em")),
                "answerable_refusal_rate": pct(local_answerable.get("refusal_rate")),
                "abstention_acc": pct(abstention.get("accuracy")),
                "overall_refusal_rate": pct(overall.get("refusal_rate")),
                "avg_input_len": avg.get("input_len"),
                "avg_output_len": avg.get("output_len"),
                "source_file": payload.get("source_file"),
            }
        )
        for qtype, row in metrics.get("by_question_type", {}).items():
            type_rows.append(
                {
                    "method": name,
                    "question_type": qtype,
                    "sub_em": pct(row.get("sub_em")),
                    "f1": pct(row.get("f1")),
                    "refusal_rate": pct(row.get("refusal_rate")),
                    "count": row.get("count"),
                    "sub_em_count": row.get("sub_em_count"),
                }
            )

    write_csv(out_dir / "method_summary.csv", summary_rows)
    write_csv(out_dir / "by_question_type.csv", type_rows)
    save_json(out_dir / "aggregate_manifest.json", {"result_dirs": args.result_dirs})
    print(f"Wrote aggregate outputs to {out_dir}")


if __name__ == "__main__":
    main()
