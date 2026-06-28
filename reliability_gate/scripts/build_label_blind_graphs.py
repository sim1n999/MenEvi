"""Build holdout graphs without exposing answer labels to graph construction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "memlens_repro" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mmkg_memory as builder  # noqa: arithmetic repair02
from experiment_utils import load_caption_cache, load_items, save_json  # noqa: arithmetic repair02


FORBIDDEN = {"answer", "answer_session_ids", "reference_answer"}


def contains_forbidden(value) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN or contains_forbidden(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(contains_forbidden(child) for child in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--caption-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cache = load_caption_cache(args.caption_cache)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = []
    for item in load_items(args.dataset):
        safe = {
            key: value for key, value in item.items()
            if key not in FORBIDDEN
        }
        graph = builder.build_graph(safe, cache, "qwen_vl")
        # The legacy builder emits this key even for label-blind input.
        # Its value is empty here; remove it before validation and persistence.
        graph.get("metadata", {}).pop("answer_session_ids", None)
        if contains_forbidden(graph):
            raise RuntimeError(f"Forbidden label field entered graph {safe['question_id']}")
        qid = str(safe["question_id"])
        save_json(output / f"{qid}.json", graph)
        counts = Counter(node.get("type") for node in graph.get("nodes", []))
        stats.append({
            "question_id": qid,
            "question_type": safe.get("question_type"),
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            **counts,
        })
    save_json(output.parent / "graph_stats.json", stats)
    save_json(output.parent / "graph_manifest.json", {
        "count": len(stats),
        "caption_cache": args.caption_cache,
        "label_fields_removed_before_build": sorted(FORBIDDEN),
        "label_blind": True,
    })
    print(f"Wrote {len(stats)} label-blind holdout graphs to {output}")


if __name__ == "__main__":
    main()

