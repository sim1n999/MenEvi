"""Estimate subtype routing without choosing the route on the evaluated fold."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from eval_v2 import compute_metrics_v2, load_items, save_json, score_row_v2, write_csv


DATASET = PROJECT_ROOT / "memlens_repro" / "data" / "memlens_agent_subset" / "dataset_32k.json"
ANSWER_PREDICTIONS = PROJECT_ROOT / "answer_evidence" / "results" / "kg_answer_focused_packet80" / "predictions.json"
VISUAL_PREDICTIONS = PROJECT_ROOT / "visual_evidence" / "results" / "kg_visual_ocr_packet80" / "predictions.json"
OUT_DIR = PROJECT_ROOT / "runtime_routing" / "results" / "hybrid_cross_validated_routing"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def assign_folds(items: Dict[str, Dict[str, Any]], folds: int) -> Dict[str, int]:
    by_subtype: Dict[str, List[str]] = defaultdict(list)
    for qid, item in items.items():
        by_subtype[str(item.get("question_subtype") or "unknown")].append(qid)
    assignment = {}
    for qids in by_subtype.values():
        for index, qid in enumerate(sorted(qids)):
            assignment[qid] = index % folds
    return assignment


def choose_from_training(
    subtype: str,
    held_out_fold: int,
    items: Dict[str, Dict[str, Any]],
    fold_map: Dict[str, int],
    answer_map: Dict[str, Dict[str, Any]],
    visual_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    train_ids = [
        qid
        for qid, item in items.items()
        if str(item.get("question_subtype") or "unknown") == subtype and fold_map[qid] != held_out_fold
    ]
    answer_correct = sum(int(answer_map[qid].get("sub_em_v2", 0)) for qid in train_ids)
    visual_correct = sum(int(visual_map[qid].get("sub_em_v2", 0)) for qid in train_ids)
    source = "visual_evidence" if visual_correct > answer_correct else "answer_evidence"
    return {
        "source": source,
        "train_count": len(train_ids),
        "train_answer_correct": answer_correct,
        "train_visual_correct": visual_correct,
    }


def main() -> None:
    folds = 5
    items = {str(item.get("question_id")): item for item in load_items(DATASET)}
    answer_raw = {str(row["question_id"]): row for row in load_rows(ANSWER_PREDICTIONS)}
    visual_raw = {str(row["question_id"]): row for row in load_rows(VISUAL_PREDICTIONS)}
    answer_map = {qid: score_row_v2(row, items[qid]) for qid, row in answer_raw.items()}
    visual_map = {qid: score_row_v2(row, items[qid]) for qid, row in visual_raw.items()}
    if set(items) != set(answer_map) or set(items) != set(visual_map):
        raise ValueError("Dataset and answer/visual prediction IDs must match")

    fold_map = assign_folds(items, folds)
    rows = []
    trace = []
    for qid in sorted(items):
        item = items[qid]
        subtype = str(item.get("question_subtype") or "unknown")
        fold = fold_map[qid]
        decision = choose_from_training(subtype, fold, items, fold_map, answer_map, visual_map)
        source = decision["source"]
        base = visual_map[qid] if source == "visual_evidence" else answer_map[qid]
        row = dict(base)
        row.update(
            {
                "route_source": source,
                "route_policy": "stratified_5fold_subtype_routing",
                "route_fold": fold,
                "route_train_count": decision["train_count"],
                "route_train_answer_correct": decision["train_answer_correct"],
                "route_train_visual_correct": decision["train_visual_correct"],
            }
        )
        rows.append(score_row_v2(row, item))
        trace.append(
            {
                "question_id": qid,
                "question_subtype": subtype,
                "fold": fold,
                **decision,
            }
        )

    payload = {
        "args": {
            "folds": folds,
            "dataset": str(DATASET.relative_to(PROJECT_ROOT)),
            "answer_predictions": str(ANSWER_PREDICTIONS.relative_to(PROJECT_ROOT)),
            "visual_predictions": str(VISUAL_PREDICTIONS.relative_to(PROJECT_ROOT)),
        },
        "data": rows,
        "metrics_v2": compute_metrics_v2(rows),
        "averaged_metrics": {
            "input_len": sum(float(row.get("input_len", 0)) for row in rows) / len(rows),
            "output_len": sum(float(row.get("output_len", 0)) for row in rows) / len(rows),
        },
        "source_file": str((OUT_DIR / "predictions.json").relative_to(PROJECT_ROOT)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUT_DIR / "predictions.json", payload)
    save_json(OUT_DIR / "metrics_v2.json", payload["metrics_v2"])
    write_csv(OUT_DIR / "route_trace.csv", trace)
    overall = payload["metrics_v2"]["overall"]
    print(
        f"cross_validated_routing: {overall['sub_em_count_v2']}/{overall['count']} "
        f"EM={100 * overall['sub_em_v2']:.2f}% F1={100 * overall['f1_v2']:.2f}%"
    )


if __name__ == "__main__":
    main()
