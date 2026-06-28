"""Build a deterministic multimodal memory graph for MemLens.

This is a runnable first-pass graph builder. It creates session, turn, image,
caption/fact, temporal, and lightweight state nodes. The JSON schema is stable
so the extraction step can later be upgraded to LLM-based extraction without
changing retrieval or answering scripts.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from experiment_utils import (
    caption_for_image,
    image_key,
    iter_sessions,
    load_caption_cache,
    load_items,
    save_json,
    turn_text,
)


SENT_RE = re.compile(r"(?<=[.!?])\s+")
STATE_PAT = re.compile(
    r"\b(now|current|currently|favorite|favourite|go-to|prefer|choose|picked|switched|instead|no longer)\b",
    re.IGNORECASE,
)


def add_node(nodes, node_id, node_type, text, **attrs):
    nodes.append({"id": node_id, "type": node_type, "text": text, **attrs})


def add_edge(edges, src, dst, edge_type, **attrs):
    edges.append({"source": src, "target": dst, "type": edge_type, **attrs})


def extract_fact_sentences(text: str, max_sentences: int = 4):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 20]
    if not sents and text:
        sents = [text[:500]]
    return sents[:max_sentences]


def build_graph(item, caption_cache, caption_source):
    nodes = []
    edges = []
    qid = item.get("question_id")
    add_node(nodes, "question", "Question", item.get("question", ""), date=item.get("question_date"))

    prev_session_id = None
    state_versions = []
    for s_idx, sid, date, turns in iter_sessions(item):
        s_node = f"session:{sid}"
        add_node(nodes, s_node, "Session", f"Session {sid} at {date}", session_id=sid, date=date, index=s_idx)
        add_edge(edges, s_node, "question", "candidate_context")
        if prev_session_id:
            add_edge(edges, f"session:{prev_session_id}", s_node, "before")
        prev_session_id = sid

        for t_idx, turn in enumerate(turns):
            t_node = f"turn:{sid}:{t_idx}"
            text = turn_text(turn)
            add_node(nodes, t_node, "Turn", text, session_id=sid, date=date, role=turn.get("role"), turn_index=t_idx)
            add_edge(edges, s_node, t_node, "contains")

            for f_idx, sent in enumerate(extract_fact_sentences(turn.get("content", ""))):
                f_node = f"fact:{sid}:{t_idx}:{f_idx}"
                add_node(nodes, f_node, "Fact", sent, session_id=sid, date=date)
                add_edge(edges, t_node, f_node, "supports")
                if STATE_PAT.search(sent):
                    st_node = f"state:{sid}:{t_idx}:{f_idx}"
                    add_node(nodes, st_node, "StateVersion", sent, session_id=sid, date=date)
                    add_edge(edges, f_node, st_node, "states")
                    if state_versions:
                        add_edge(edges, state_versions[-1], st_node, "supersedes")
                    state_versions.append(st_node)

            for i_idx, img in enumerate(turn.get("images", []) or []):
                key = image_key(img)
                img_node = f"image:{key}"
                add_node(nodes, img_node, "Image", key, session_id=sid, date=date)
                add_edge(edges, t_node, img_node, "has_image")
                cap = caption_for_image(img, caption_source, caption_cache)
                if cap:
                    cap_node = f"visual:{sid}:{t_idx}:{i_idx}"
                    add_node(nodes, cap_node, "VisualFact", cap, session_id=sid, date=date, image_id=key)
                    add_edge(edges, img_node, cap_node, "depicts")
                    add_edge(edges, cap_node, t_node, "visual_grounding")

    stats = Counter(n["type"] for n in nodes)
    return {
        "question_id": qid,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "question_date": item.get("question_date"),
            "answer_session_ids": item.get("answer_session_ids", []),
            "node_type_counts": dict(stats),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caption-source", choices=["dataset", "qwen_vl"], default="qwen_vl")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    caption_cache = load_caption_cache(args.caption_cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_stats = []
    for item in tqdm(items, desc="Build MMKG"):
        graph = build_graph(item, caption_cache, args.caption_source)
        save_json(out_dir / f"{item.get('question_id')}.json", graph)
        graph_stats.append(
            {
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                **graph["metadata"]["node_type_counts"],
            }
        )
    save_json(out_dir.parent / "graph_stats.json", graph_stats)
    print(f"Wrote {len(graph_stats)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
