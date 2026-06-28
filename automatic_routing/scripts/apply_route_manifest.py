from __future__ import annotations

import argparse
import json
from pathlib import Path

TYPE_BY_SUBTYPE = {
    "answer_refusal": "answer_refusal",
    "arithmetic": "multi_session_reasoning",
    "counting": "multi_session_reasoning",
    "duration_comparison": "temporal_reasoning",
    "entity": "information_extraction",
    "entity_resolution": "multi_session_reasoning",
    "knowledge_update": "knowledge_update",
    "order_ranking": "temporal_reasoning",
    "previnfo": "information_extraction",
    "temporal_info_extraction": "temporal_reasoning",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    routes = {row["question_id"]: row["predicted_subtype"] for row in manifest["routes"]}
    ids = {str(item["question_id"]) for item in items}
    if ids != set(routes):
        raise RuntimeError("Route manifest does not exactly cover dataset")
    for item in items:
        predicted = routes[str(item["question_id"])]
        item["question_subtype"] = predicted
        item["question_type"] = TYPE_BY_SUBTYPE[predicted]
    Path(args.output).write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
