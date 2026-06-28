"""Retrieve question-relevant subgraphs from MM-KG memory graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from experiment_utils import BM25Index, load_items, save_json, write_jsonl


def incident_edges(graph, node_ids):
    node_ids = set(node_ids)
    return [e for e in graph.get("edges", []) if e.get("source") in node_ids or e.get("target") in node_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-budget", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []

    for item in tqdm(items, desc="Retrieve MMKG"):
        qid = item.get("question_id")
        graph = __import__("json").loads((graph_dir / f"{qid}.json").read_text(encoding="utf-8"))
        docs = [
            {
                "node_id": n["id"],
                "text": f"{n.get('type', '')}: {n.get('text', '')} {n.get('date', '')}",
                "node": n,
            }
            for n in graph.get("nodes", [])
        ]
        ranked = BM25Index(docs).score(item.get("question", ""))
        selected_ids = []
        for _, doc in ranked:
            if doc["node_id"] not in selected_ids:
                selected_ids.append(doc["node_id"])
            if len(selected_ids) >= args.node_budget:
                break

        # Add session nodes for selected evidence and neighbors through incident edges.
        selected = set(selected_ids)
        for e in graph.get("edges", []):
            if e.get("source") in selected or e.get("target") in selected:
                selected.add(e.get("source"))
                selected.add(e.get("target"))
            if len(selected) >= args.node_budget:
                break

        nodes = [n for n in graph.get("nodes", []) if n["id"] in selected]
        edges = [e for e in graph.get("edges", []) if e.get("source") in selected and e.get("target") in selected]
        subgraph = {"question_id": qid, "nodes": nodes, "edges": edges, "metadata": graph.get("metadata", {})}
        save_json(out_dir / f"{qid}.json", subgraph)

        retrieved_sessions = {
            n.get("session_id") for n in nodes if n.get("session_id")
        }
        answer_ids = set(item.get("answer_session_ids") or [])
        logs.append(
            {
                "question_id": qid,
                "question_type": item.get("question_type"),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "answer_session_ids": list(answer_ids),
                "retrieved_session_ids": sorted(retrieved_sessions),
                "session_hit": bool(answer_ids & retrieved_sessions) if answer_ids else None,
                "session_all_hit": answer_ids <= retrieved_sessions if answer_ids else None,
                "top_node_ids": selected_ids[:10],
            }
        )

    write_jsonl(out_dir.parent / "retrieval_logs.jsonl", logs)
    print(f"Wrote subgraphs to {out_dir}")


if __name__ == "__main__":
    main()
