"""Build runtime-routing evaluation predictions from answer and visual evidence outputs."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from aggregate_eval_v2 import add_rows  # noqa: E402
from eval_v2 import compute_metrics_v2, load_items, save_json, score_row_v2, write_csv  # noqa: E402


DATASET = PROJECT_ROOT / "memlens_repro" / "data" / "memlens_agent_subset" / "dataset_32k.json"
ANSWER_PREDICTIONS = PROJECT_ROOT / "answer_evidence" / "results" / "kg_answer_focused_packet80" / "predictions.json"
VISUAL_PREDICTIONS = PROJECT_ROOT / "visual_evidence" / "results" / "kg_visual_ocr_packet80" / "predictions.json"
OUT_DIR = PROJECT_ROOT / "runtime_routing" / "results" / "hybrid_subtype_routing"
FINAL_DIR = PROJECT_ROOT / "runtime_routing" / "results" / "final_eval_v2"


USE_VISUAL_EVIDENCE_SUBTYPES = {
    "entity",
    "previnfo",
    "knowledge_update",
    "temporal_info_extraction",
}


def load_payload(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        payload = {"data": payload}
    payload["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return payload


def payload_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise TypeError("Prediction payload data must be a list")
    return data


def average_lengths(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"input_len": 0.0, "output_len": 0.0}
    return {
        "input_len": sum(float(x.get("input_len", 0) or 0) for x in rows) / len(rows),
        "output_len": sum(float(x.get("output_len", 0) or 0) for x in rows) / len(rows),
    }


def choose_source(subtype: str) -> str:
    return "visual_evidence" if subtype in USE_VISUAL_EVIDENCE_SUBTYPES else "answer_evidence"


def route_reason(subtype: str, source: str) -> str:
    if source == "visual_evidence":
        return "Visual evidence outperformed answer evidence for this subtype in validation analysis"
    if subtype in {"duration_comparison", "entity_resolution"}:
        return "Answer evidence protects this subtype from visual-evidence regressions"
    return "Answer evidence is the conservative default for tied or unrepaired subtypes"


def write_route_summary(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("question_subtype", "unknown")].append(row)

    summary = []
    for subtype, group in sorted(grouped.items()):
        n = len(group)
        correct = sum(int(x.get("sub_em_v2", 0)) for x in group)
        refusal = sum(int(x.get("is_refusal_v2", 0)) for x in group)
        source_counts: Dict[str, int] = defaultdict(int)
        for row in group:
            source_counts[row.get("route_source", "unknown")] += 1
        summary.append(
            {
                "question_subtype": subtype,
                "count": n,
                "route_answer_evidence": source_counts.get("answer_evidence", 0),
                "route_visual_evidence": source_counts.get("visual_evidence", 0),
                "sub_em_count_v2": correct,
                "sub_em_v2": correct / n if n else 0.0,
                "refusal_rate_v2": refusal / n if n else 0.0,
            }
        )
    write_csv(OUT_DIR / "route_summary.csv", summary)


def write_flip_analysis(answer_rows: Dict[str, Dict[str, Any]], visual_rows: Dict[str, Dict[str, Any]], hybrid_rows: List[Dict[str, Any]]) -> None:
    rows = []
    for h in hybrid_rows:
        qid = h["question_id"]
        answer = answer_rows[qid]
        visual = visual_rows[qid]
        rows.append(
            {
                "question_id": qid,
                "question_type": h.get("question_type"),
                "question_subtype": h.get("question_subtype"),
                "route_source": h.get("route_source"),
                "reference_answer": h.get("reference_answer"),
                "answer_prediction_v2": answer.get("prediction_v2"),
                "visual_prediction_v2": visual.get("prediction_v2"),
                "hybrid_prediction_v2": h.get("prediction_v2"),
                "answer_sub_em_v2": answer.get("sub_em_v2"),
                "visual_sub_em_v2": visual.get("sub_em_v2"),
                "hybrid_sub_em_v2": h.get("sub_em_v2"),
                "answer_is_refusal_v2": answer.get("is_refusal_v2"),
                "visual_is_refusal_v2": visual.get("is_refusal_v2"),
                "hybrid_is_refusal_v2": h.get("is_refusal_v2"),
            }
        )
    write_csv(OUT_DIR / "flip_analysis.csv", rows)


def write_aggregate(answer_payload: Dict[str, Any], visual_payload: Dict[str, Any], hybrid_payload: Dict[str, Any]) -> None:
    summary_rows: List[Dict[str, Any]] = []
    type_rows: List[Dict[str, Any]] = []
    subtype_rows: List[Dict[str, Any]] = []

    add_rows("KG answer-focused packet80", answer_payload["metrics_v2"], summary_rows, type_rows, subtype_rows, answer_payload)
    add_rows("kg_visual_ocr_packet80", visual_payload["metrics_v2"], summary_rows, type_rows, subtype_rows, visual_payload)
    add_rows("hybrid_subtype_routing", hybrid_payload["metrics_v2"], summary_rows, type_rows, subtype_rows, hybrid_payload)

    write_csv(FINAL_DIR / "method_summary.csv", summary_rows)
    write_csv(FINAL_DIR / "by_question_type.csv", type_rows)
    write_csv(FINAL_DIR / "by_question_subtype.csv", subtype_rows)
    save_json(
        FINAL_DIR / "aggregate_manifest.json",
        {
            "result_dirs": [
                str(ANSWER_PREDICTIONS.parent.relative_to(PROJECT_ROOT)),
                str(VISUAL_PREDICTIONS.parent.relative_to(PROJECT_ROOT)),
                str(OUT_DIR.relative_to(PROJECT_ROOT)),
            ],
            "routing_policy": {
                "visual_evidence": sorted(USE_VISUAL_EVIDENCE_SUBTYPES),
                "answer_evidence_default": "all other subtypes",
            },
        },
    )


def main() -> None:
    items = {item.get("question_id"): item for item in load_items(DATASET)}
    answer_payload = load_payload(ANSWER_PREDICTIONS)
    visual_payload = load_payload(VISUAL_PREDICTIONS)
    answer_map = {row["question_id"]: row for row in payload_rows(answer_payload)}
    visual_map = {row["question_id"]: row for row in payload_rows(visual_payload)}

    missing = sorted(set(answer_map) ^ set(visual_map))
    if missing:
        raise ValueError(f"Answer/visual prediction IDs do not match: {missing[:5]}")

    hybrid_rows = []
    for qid in answer_map:
        if qid not in items:
            raise KeyError(f"Question not found in dataset: {qid}")
        subtype = str(answer_map[qid].get("question_subtype") or visual_map[qid].get("question_subtype") or "unknown")
        source = choose_source(subtype)
        base = visual_map[qid] if source == "visual_evidence" else answer_map[qid]
        row = dict(base)
        row["route_source"] = source
        row["route_reason"] = route_reason(subtype, source)
        row["route_policy"] = "hybrid_subtype_routing_v1"
        hybrid_rows.append(score_row_v2(row, items[qid]))

    hybrid_payload = {
        "args": {
            "dataset": str(DATASET.relative_to(PROJECT_ROOT)),
            "answer_predictions": str(ANSWER_PREDICTIONS.relative_to(PROJECT_ROOT)),
            "visual_predictions": str(VISUAL_PREDICTIONS.relative_to(PROJECT_ROOT)),
            "routing_policy": "hybrid_subtype_routing_v1",
        },
        "data": hybrid_rows,
        "metrics_v2": compute_metrics_v2(hybrid_rows),
        "averaged_metrics": average_lengths(hybrid_rows),
        "source_file": str((OUT_DIR / "predictions.json").relative_to(PROJECT_ROOT)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUT_DIR / "predictions.json", hybrid_payload)
    save_json(OUT_DIR / "metrics_v2.json", hybrid_payload["metrics_v2"])
    write_route_summary(hybrid_rows)
    write_flip_analysis(answer_map, visual_map, hybrid_rows)
    write_aggregate(answer_payload, visual_payload, hybrid_payload)

    overall = hybrid_payload["metrics_v2"]["overall"]
    print(
        "hybrid_subtype_routing: "
        f"{overall['sub_em_count_v2']}/{overall['count']} "
        f"EM={100 * overall['sub_em_v2']:.2f}% "
        f"F1={100 * overall['f1_v2']:.2f}%"
    )
    print(f"Wrote {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {FINAL_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()



