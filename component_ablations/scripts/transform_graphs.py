from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VISUAL_TYPES = {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"}
TEMPORAL_EDGES = {"before", "after", "supersedes", "updates", "temporal"}


def transform(graph: dict, variant: str) -> dict:
    value = json.loads(json.dumps(graph))
    nodes = value.get("nodes", [])
    edges = value.get("edges", [])
    remove_ids = set()
    if variant == "no_state":
        remove_ids = {str(node.get("id")) for node in nodes if node.get("type") == "StateVersion"}
    elif variant == "no_visual":
        remove_ids = {str(node.get("id")) for node in nodes if node.get("type") in VISUAL_TYPES}
    if remove_ids:
        nodes = [node for node in nodes if str(node.get("id")) not in remove_ids]
        edges = [edge for edge in edges if str(edge.get("source")) not in remove_ids and str(edge.get("target")) not in remove_ids]
    if variant == "no_edges":
        edges = []
    elif variant == "no_temporal":
        edges = [edge for edge in edges if str(edge.get("type", "")).lower() not in TEMPORAL_EDGES]
    value["nodes"], value["edges"] = nodes, edges
    value.setdefault("metadata", {})["component_graph_ablation"] = variant
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["no_edges", "no_state", "no_visual", "no_temporal"], required=True)
    parser.add_argument("--expected-count", type=int, default=789)
    args = parser.parse_args()
    source, output = Path(args.input_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = []
    paths = sorted(source.glob("*.json"))
    if len(paths) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} graphs, found {len(paths)}")
    for path in paths:
        graph = transform(json.loads(path.read_text(encoding="utf-8")), args.variant)
        (output / path.name).write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
        counts = Counter(node.get("type") for node in graph.get("nodes", []))
        stats.append({"question_id": path.stem, "nodes": len(graph.get("nodes", [])),
                      "edges": len(graph.get("edges", [])), "node_types": dict(counts)})
    (output.parent / "transform_manifest.json").write_text(
        json.dumps({"variant": args.variant, "count": len(paths), "stats": stats}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

