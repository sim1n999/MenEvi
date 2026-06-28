"""Integrate MenLens baselines with Runtime-routing evaluation under evaluator-v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from eval_v2 import compute_metrics_v2, load_items, save_json, score_row_v2, write_csv  # noqa: arithmetic repair02


DATASET = PROJECT_ROOT / "memlens_repro" / "data" / "memlens_agent_subset" / "dataset_32k.json"
OUT_DIR = PROJECT_ROOT / "runtime_routing" / "results" / "baseline_integrated_v2"
RESCORED_DIR = OUT_DIR / "rescored_predictions"


METHODS = [
    {
        "group": "MenLens baseline",
        "slug": "qwen25vl7b_no_context",
        "method": "Qwen2.5-VL-7B no-context",
        "path": "memlens_repro/outputs/qwen25vl7b_32k_agent_no_context",
    },
    {
        "group": "MenLens baseline",
        "slug": "qwen25vl7b_text_only",
        "method": "Qwen2.5-VL-7B text-only",
        "path": "memlens_repro/outputs/qwen25vl7b_32k_agent_text_only",
    },
    {
        "group": "MenLens baseline",
        "slug": "qwen25vl7b_direct_32k",
        "method": "Qwen2.5-VL-7B direct 32K",
        "path": "memlens_repro/outputs/qwen25vl7b_32k_agent_direct",
    },
    {
        "group": "MenLens baseline",
        "slug": "last3_qwen25vl",
        "method": "Last-3 sessions Qwen2.5-VL",
        "path": "memlens_repro/outputs/last3_qwen25vl_32k_agent",
    },
    {
        "group": "MenLens baseline",
        "slug": "bm25_text_rag",
        "method": "BM25 text RAG",
        "path": "memlens_repro/outputs/bm25_text_rag_32k_agent",
    },
    {
        "group": "MenLens baseline",
        "slug": "blip_caption_rag",
        "method": "BLIP caption RAG",
        "path": "memlens_repro/outputs/blip_caption_rag_32k_agent",
    },
    {
        "group": "MenLens baseline",
        "slug": "qwen_vl_caption_rag",
        "method": "Qwen-VL caption RAG",
        "path": "memlens_repro/outputs/qwen_vl_caption_rag_32k_agent",
    },
    {
        "group": "MenLens baseline",
        "slug": "kg_memory_32k_agent",
        "method": "KG memory 32K agent",
        "path": "memlens_repro/outputs/kg_memory_32k_agent",
    },
    {
        "group": "MenLens upper bound",
        "slug": "oracle_evidence_qwen25vl",
        "method": "Oracle evidence Qwen2.5-VL",
        "path": "memlens_repro/outputs/oracle_evidence_qwen25vl_32k_agent",
    },
    {
        "group": "Ours B",
        "slug": "kg_soft_refusal_budget120",
        "method": "KG soft-refusal budget120",
        "path": "answer_evidence/results/rescore_existing_v2/rescored_predictions/kg_soft_refusal_budget120/predictions.json",
    },
    {
        "group": "Ours C",
        "slug": "kg_answer_focused_packet80",
        "method": "KG answer-focused packet80",
        "path": "answer_evidence/results/kg_answer_focused_packet80/predictions.json",
    },
    {
        "group": "Ours D",
        "slug": "kg_visual_ocr_packet80",
        "method": "KG visual-OCR packet80 type-aware",
        "path": "visual_evidence/results/kg_visual_ocr_packet80/predictions.json",
    },
    {
        "group": "Ours E",
        "slug": "hybrid_subtype_routing",
        "method": "Hybrid subtype routing",
        "path": "runtime_routing/results/hybrid_subtype_routing/predictions.json",
    },
]


EXCLUDE_JSON_NAMES = {
    "aggregate_manifest.json",
    "graph_stats.json",
    "metrics.json",
    "run_config.json",
    "summary_metrics.json",
}


def pct(x: Any) -> Any:
    return None if x is None else 100 * float(x)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prediction_file(path_value: str) -> Path:
    path = PROJECT_ROOT / path_value
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    predictions = path / "predictions.json"
    if predictions.is_file():
        return predictions
    candidates = [
        p
        for p in path.glob("*.json")
        if p.name not in EXCLUDE_JSON_NAMES
        and not p.name.endswith(".metrics")
        and "manifest" not in p.name
        and "error_cases" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No prediction JSON found in {path}")
    # Timestamped Qwen-VL output dirs can contain an early failed partial JSON.
    # The complete prediction file is the largest regular JSON.
    return sorted(candidates, key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)[0]


def payload_data(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise TypeError("Prediction payload must be a list or a dict with a data list")


def average_lengths(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"input_len": 0.0, "output_len": 0.0}
    return {
        "input_len": sum(float(x.get("input_len", 0) or 0) for x in rows) / len(rows),
        "output_len": sum(float(x.get("output_len", 0) or 0) for x in rows) / len(rows),
    }


def rescore_method(spec: Dict[str, str], items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source = resolve_prediction_file(spec["path"])
    raw_payload = load_json(source)
    rows = payload_data(raw_payload)
    rescored = []
    for row in rows:
        qid = row.get("question_id")
        if qid not in items:
            raise KeyError(f"Question not found in dataset for {spec['method']}: {qid}")
        rescored.append(score_row_v2(dict(row), items[qid]))
    payload = {
        "group": spec["group"],
        "method": spec["method"],
        "source_file": str(source.relative_to(PROJECT_ROOT)),
        "data": rescored,
        "metrics_v2": compute_metrics_v2(rescored),
        "averaged_metrics": average_lengths(rescored),
    }
    method_dir = RESCORED_DIR / spec["slug"]
    save_json(method_dir / "predictions.json", payload)
    save_json(method_dir / "metrics_v2.json", payload["metrics_v2"])
    return payload


def summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload["metrics_v2"]
    overall = metrics.get("overall", {})
    answerable = metrics.get("answerable", {})
    abstention = metrics.get("abstention", {})
    avg = payload.get("averaged_metrics", {})
    return {
        "group": payload["group"],
        "method": payload["method"],
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


def metric_rows(payload: Dict[str, Any], metric_key: str, label_key: str) -> List[Dict[str, Any]]:
    out = []
    for label, row in payload["metrics_v2"].get(metric_key, {}).items():
        out.append(
            {
                "group": payload["group"],
                "method": payload["method"],
                label_key: label,
                **{
                    k: pct(v) if k.endswith("_v2") and k != "sub_em_count_v2" else v
                    for k, v in row.items()
                },
            }
        )
    return out


def main() -> None:
    items = {item.get("question_id"): item for item in load_items(DATASET)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESCORED_DIR.mkdir(parents=True, exist_ok=True)

    payloads = [rescore_method(spec, items) for spec in METHODS]

    summary_rows = [summary_row(payload) for payload in payloads]
    type_rows: List[Dict[str, Any]] = []
    subtype_rows: List[Dict[str, Any]] = []
    for payload in payloads:
        type_rows.extend(metric_rows(payload, "by_question_type", "question_type"))
        subtype_rows.extend(metric_rows(payload, "by_question_subtype", "question_subtype"))

    ranking_rows = sorted(summary_rows, key=lambda x: float(x.get("overall_sub_em_v2") or 0), reverse=True)

    write_csv(OUT_DIR / "method_summary.csv", summary_rows)
    write_csv(OUT_DIR / "method_ranking.csv", ranking_rows)
    write_csv(OUT_DIR / "by_question_type.csv", type_rows)
    write_csv(OUT_DIR / "by_question_subtype.csv", subtype_rows)
    save_json(
        OUT_DIR / "aggregate_manifest.json",
        {
            "dataset": str(DATASET.relative_to(PROJECT_ROOT)),
            "evaluator": "answer_evidence/scripts/eval_v2.py",
            "methods": [
                {
                    "group": spec["group"],
                    "method": spec["method"],
                    "source": str(resolve_prediction_file(spec["path"]).relative_to(PROJECT_ROOT)),
                }
                for spec in METHODS
            ],
        },
    )

    best_non_oracle = next(row for row in ranking_rows if row["group"] != "MenLens upper bound")
    print(f"Wrote {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(
        "Best non-oracle: "
        f"{best_non_oracle['method']} "
        f"EM={float(best_non_oracle['overall_sub_em_v2']):.2f}%"
    )


if __name__ == "__main__":
    main()

