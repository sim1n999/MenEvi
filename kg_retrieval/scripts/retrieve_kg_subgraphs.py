"""Retrieve KG subgraphs for KG retrieval baseline into this experiment folder."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from local_memlens_utils import load_items, save_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, docs: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.get("text", "")) for doc in self.docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = len(self.docs)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        query_terms = tokenize(query)
        scored = []
        for doc, term_freq, doc_len in zip(self.docs, self.term_freqs, self.doc_lens):
            score = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue
                freq = term_freq[term]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            scored.append((score, doc))
        return sorted(scored, key=lambda item: item[0], reverse=True)


def node_text(node: Dict[str, Any]) -> str:
    parts = [
        node.get("type", ""),
        node.get("text", ""),
        node.get("date", ""),
        node.get("session_id", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def retrieve_one(item: Dict[str, Any], graph: Dict[str, Any], node_budget: int) -> Dict[str, Any]:
    docs = [
        {
            "node_id": node["id"],
            "text": node_text(node),
            "node": node,
        }
        for node in graph.get("nodes", [])
        if node.get("id")
    ]
    ranked = BM25Index(docs).score(item.get("question", ""))
    selected_ids: List[str] = []
    for _, doc in ranked:
        node_id = doc["node_id"]
        if node_id not in selected_ids:
            selected_ids.append(node_id)
        if len(selected_ids) >= node_budget:
            break

    selected = set(selected_ids)
    for edge in graph.get("edges", []):
        if edge.get("source") in selected or edge.get("target") in selected:
            selected.add(edge.get("source"))
            selected.add(edge.get("target"))
        if len(selected) >= node_budget:
            break

    nodes = [node for node in graph.get("nodes", []) if node.get("id") in selected]
    edges = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("source") in selected and edge.get("target") in selected
    ]
    return {
        "question_id": item.get("question_id"),
        "nodes": nodes,
        "edges": edges,
        "metadata": graph.get("metadata", {}),
        "top_node_ids": selected_ids[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-budget", type=int, default=120)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.output_dir)
    subgraph_dir = out_dir / "retrieved_subgraphs"
    subgraph_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    for item in items:
        qid = item.get("question_id")
        graph_path = graph_dir / f"{qid}.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"Full graph missing for {qid}: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        subgraph = retrieve_one(item, graph, args.node_budget)
        save_json(subgraph_dir / f"{qid}.json", subgraph)

        retrieved_sessions = {
            node.get("session_id")
            for node in subgraph.get("nodes", [])
            if node.get("session_id")
        }
        answer_ids = set(item.get("answer_session_ids") or [])
        logs.append(
            {
                "question_id": qid,
                "question_type": item.get("question_type"),
                "node_count": len(subgraph.get("nodes", [])),
                "edge_count": len(subgraph.get("edges", [])),
                "answer_session_ids": sorted(answer_ids),
                "retrieved_session_ids": sorted(retrieved_sessions),
                "session_hit": bool(answer_ids & retrieved_sessions) if answer_ids else None,
                "session_all_hit": answer_ids <= retrieved_sessions if answer_ids else None,
                "top_node_ids": subgraph.get("top_node_ids", [])[:10],
            }
        )

    write_jsonl(out_dir / "retrieval_logs.jsonl", logs)
    save_json(
        out_dir / "retrieval_manifest.json",
        {
            "input": args.input,
            "graph_dir": args.graph_dir,
            "node_budget": args.node_budget,
            "count": len(items),
            "subgraph_dir": str(subgraph_dir),
        },
    )
    print(f"Wrote {len(items)} retrieved subgraphs to {subgraph_dir}")


if __name__ == "__main__":
    main()
