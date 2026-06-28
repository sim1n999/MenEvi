"""Build answer-focused evidence packets from retrieved KG subgraphs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from eval_v2 import infer_contract, load_items, save_json, write_csv


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
VISUAL_TYPES = {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"}
STATE_TYPES = {"StateVersion"}


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def node_text(node: Dict[str, Any]) -> str:
    parts = [
        node.get("type", ""),
        node.get("date", ""),
        node.get("session_id", ""),
        node.get("text", ""),
    ]
    return " ".join(str(x) for x in parts if x)


def compact_text(text: Any, max_chars: int = 650) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def compact_node(node: Dict[str, Any]) -> str:
    fields = [
        f"[{node.get('type', 'Node')}]",
        f"date={node.get('date')}" if node.get("date") else "",
        f"session={node.get('session_id')}" if node.get("session_id") else "",
        compact_text(node.get("text", "")),
    ]
    return " ".join(x for x in fields if x).strip()


def relevance_score(question_terms: Counter, node: Dict[str, Any]) -> float:
    text_terms = Counter(tokenize(node_text(node)))
    overlap = sum(min(question_terms[t], text_terms[t]) for t in question_terms)
    score = float(overlap)
    node_type = node.get("type")
    if node_type in VISUAL_TYPES:
        score += 1.5
    if node_type in STATE_TYPES:
        score += 1.0
    if node_type == "Session":
        score -= 0.5
    if node_type == "Turn":
        score -= 1.0
        score -= min(len(str(node.get("text", ""))) / 5000.0, 1.0)
    return score


def build_packet(item: Dict[str, Any], graph: Dict[str, Any], packet_budget: int) -> Dict[str, Any]:
    question_terms = Counter(tokenize(item.get("question", "")))
    nodes = [node for node in graph.get("nodes", []) if node.get("type") != "Question"]
    ranked = sorted(nodes, key=lambda n: relevance_score(question_terms, n), reverse=True)
    top_nodes = ranked[:packet_budget]
    top_ids = {n.get("id") for n in top_nodes}
    session_nodes = {
        n.get("id"): n for n in nodes if n.get("type") == "Session" and n.get("id")
    }

    visual = [n for n in top_nodes if n.get("type") in VISUAL_TYPES]
    states = [n for n in top_nodes if n.get("type") in STATE_TYPES]
    candidates = [n for n in top_nodes if n.get("type") not in VISUAL_TYPES | STATE_TYPES and n.get("type") != "Session"]
    selected_session_ids: List[str] = []
    for node in top_nodes:
        session_id = node.get("session_id")
        if session_id and session_id not in selected_session_ids:
            selected_session_ids.append(session_id)

    session_lines: List[str] = []
    for session_id in selected_session_ids[:12]:
        session_node = session_nodes.get(session_id)
        if session_node:
            session_lines.append(compact_node(session_node))
        else:
            session_lines.append(f"[Session] session={session_id}")

    edges = []
    for edge in graph.get("edges", []):
        if edge.get("source") in top_ids and edge.get("target") in top_ids:
            edges.append(f"{edge.get('source')} --{edge.get('type')}--> {edge.get('target')}")

    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "question_subtype": item.get("question_subtype"),
        "contract": infer_contract(item),
        "reference_answer": item.get("answer"),
        "packet_budget": packet_budget,
        "top_candidates": [compact_node(n) for n in candidates[:24]],
        "supporting_sessions": session_lines,
        "temporal_update_evidence": [compact_node(n) for n in states[:16]] + edges[:24],
        "visual_evidence": [compact_node(n) for n in visual[:24]],
        "stats": {
            "source_node_count": len(nodes),
            "packet_node_count": len(top_nodes),
            "candidate_count": len(candidates[:24]),
            "visual_count": len(visual[:24]),
            "state_count": len(states[:16]),
            "session_count": len(session_lines),
            "edge_count": len(edges[:24]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subgraph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-budget", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.dataset)
    if args.max_samples:
        items = items[: args.max_samples]

    out_dir = Path(args.output_dir)
    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for item in items:
        qid = item.get("question_id")
        graph_path = Path(args.subgraph_dir) / f"{qid}.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        packet = build_packet(item, graph, args.packet_budget)
        save_json(packet_dir / f"{qid}.json", packet)
        stats.append({"question_id": qid, "question_type": item.get("question_type"), **packet["stats"]})

    write_csv(out_dir / "packet_stats.csv", stats)
    save_json(
        out_dir / "packet_manifest.json",
        {
            "dataset": args.dataset,
            "subgraph_dir": args.subgraph_dir,
            "packet_budget": args.packet_budget,
            "count": len(items),
            "packet_dir": str(packet_dir),
        },
    )
    print(f"Wrote {len(items)} evidence packets to {packet_dir}")


if __name__ == "__main__":
    main()
