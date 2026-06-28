"""Aggregate Answer-evidence evaluation evaluator-v2 results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from eval_v2 import compute_metrics_v2, load_prediction_payload, save_json, write_csv


SHORT_NAMES = {
    "kg_memory_32k_agent": "KG hard-refusal budget80",
    "qwen_vl_caption_rag_32k_agent": "Qwen-VL caption RAG",
    "oracle_evidence_qwen25vl_32k_agent": "Oracle evidence Qwen2.5-VL-7B",
    "kg_soft_refusal_budget120": "KG soft-refusal budget120",
    "kg_answer_focused_packet80": "KG answer-focused packet80",
    "kg_answer_focused_packet40": "KG answer-focused packet40",
}


def pct(x: Any) -> Any:
    return None if x is None else 100 * float(x)


def metrics_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("metrics_v2") or compute_metrics_v2(payload.get("data", []))


def add_rows(name: str, metrics: Dict[str, Any], summary_rows: List[Dict[str, Any]], type_rows: List[Dict[str, Any]], subtype_rows: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    overall = metrics.get("overall", {})
    answerable = metrics.get("answerable", {})
    abstention = metrics.get("abstention", {})
    avg = payload.get("averaged_metrics", {})
    summary_rows.append(
        {
            "method": name,
            "count": overall.get("count"),
            "overall_sub_em_v2": pct(overall.get("sub_em_v2")),
            "overall_f1_v2": pct(overall.get("f1_v2")),
            "answerable_sub_em_v2": pct(answerable.get("sub_em_v2")),
            "answerable_refusal_rate_v2": pct(answerable.get("refusal_rate_v2")),
            "abstention_acc_v2": pct(abstention.get("accuracy_v2")),
            "overall_refusal_rate_v2": pct(overall.get("refusal_rate_v2")),
            "avg_input_len": avg.get("input_len"),
            "avg_output_len": avg.get("output_len"),
            "source_file": payload.get("source_file"),
        }
    )
    for qtype, row in metrics.get("by_question_type", {}).items():
        type_rows.append({"method": name, "question_type": qtype, **{k: pct(v) if k.endswith("_v2") and k != "sub_em_count_v2" else v for k, v in row.items()}})
    for subtype, row in metrics.get("by_question_subtype", {}).items():
        subtype_rows.append({"method": name, "question_subtype": subtype, **{k: pct(v) if k.endswith("_v2") and k != "sub_em_count_v2" else v for k, v in row.items()}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-dirs", nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_rows: List[Dict[str, Any]] = []
    type_rows: List[Dict[str, Any]] = []
    subtype_rows: List[Dict[str, Any]] = []
    for result_dir in args.result_dirs:
        path = Path(result_dir)
        payload = load_prediction_payload(path)
        name = SHORT_NAMES.get(path.name, path.name)
        add_rows(name, metrics_from_payload(payload), summary_rows, type_rows, subtype_rows, payload)

    write_csv(out_dir / "method_summary.csv", summary_rows)
    write_csv(out_dir / "by_question_type.csv", type_rows)
    write_csv(out_dir / "by_question_subtype.csv", subtype_rows)
    save_json(out_dir / "aggregate_manifest.json", {"result_dirs": args.result_dirs})
    print(f"Wrote aggregate outputs to {out_dir}")


if __name__ == "__main__":
    main()

