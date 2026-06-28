"""Build visual/OCR enhanced type-aware evidence packets.

This Visual-evidence evaluation builder keeps the compact-packet idea from Answer-evidence evaluation but
allocates more evidence budget to visual text, objects, colors, positions, and
attributes for information_extraction questions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from eval_v2 import infer_contract, load_items, save_json, write_csv  # noqa: arithmetic repair02


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
VISUAL_TYPES = {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"}
STATE_TYPES = {"StateVersion"}
LOW_VALUE_TYPES = {"Question"}

VISUAL_DETAIL_TERMS = {
    "visible_text",
    "visible text",
    "short_caption",
    "caption",
    "visible_objects",
    "visible objects",
    "attributes",
    "color",
    "colour",
    "left",
    "right",
    "top",
    "bottom",
    "background",
    "foreground",
    "label",
    "sign",
    "title",
    "cover",
    "brand",
    "word",
    "text",
    "name",
}

INFO_EXTRACTION_CUES = {
    "shown",
    "image",
    "photo",
    "picture",
    "visible",
    "text",
    "word",
    "name",
    "title",
    "cover",
    "sign",
    "label",
    "brand",
    "color",
    "colour",
    "left",
    "right",
    "top",
    "bottom",
    "background",
    "object",
    "wearing",
    "advertisement",
    "poster",
}

GENERIC_VISUAL_QUERY_TERMS = {
    "what",
    "which",
    "is",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "for",
    "my",
    "i",
    "shown",
    "image",
    "photo",
    "picture",
    "visible",
    "answer",
    "short",
    "phrase",
    "only",
    "no",
    "explanation",
}


def tokenize(text: Any) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(str(text or ""))]


def node_text(node: Dict[str, Any]) -> str:
    parts = [
        node.get("type", ""),
        node.get("id", ""),
        node.get("date", ""),
        node.get("session_id", ""),
        node.get("text", ""),
    ]
    return " ".join(str(x) for x in parts if x)


def compact_text(text: Any, max_chars: int = 760) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def compact_node(node: Dict[str, Any], max_chars: int = 760) -> str:
    fields = [
        f"[{node.get('type', 'Node')}]",
        f"date={node.get('date')}" if node.get("date") else "",
        f"session={node.get('session_id')}" if node.get("session_id") else "",
        compact_text(node.get("text", ""), max_chars=max_chars),
    ]
    return " ".join(x for x in fields if x).strip()


def is_visual_node(node: Dict[str, Any]) -> bool:
    text = node_text(node).lower()
    return node.get("type") in VISUAL_TYPES or node.get("id", "").startswith("visual:") or "visible_text" in text


def has_visual_detail(node: Dict[str, Any]) -> bool:
    text = node_text(node).lower()
    return any(term in text for term in VISUAL_DETAIL_TERMS)


def overlap_score(question_terms: Counter, node: Dict[str, Any]) -> float:
    terms = Counter(tokenize(node_text(node)))
    return float(sum(min(question_terms[t], terms[t]) for t in question_terms))


def visual_detail_score(question_terms: Counter, node: Dict[str, Any], is_info_extraction: bool) -> float:
    text = node_text(node).lower()
    score = 0.0
    if is_visual_node(node):
        score += 4.0 if is_info_extraction else 1.5
    if has_visual_detail(node):
        score += 3.0 if is_info_extraction else 1.0
    if is_info_extraction:
        content_terms = [t for t in question_terms if t not in GENERIC_VISUAL_QUERY_TERMS]
        content_overlap = sum(1 for t in content_terms if t in text)
        score += 4.0 * content_overlap
        for cue in INFO_EXTRACTION_CUES:
            if cue in text:
                score += 0.25
        for cue in question_terms:
            if cue in INFO_EXTRACTION_CUES and cue in text:
                score += 2.0
    return score


def relevance_score(question_terms: Counter, node: Dict[str, Any], question_type: str) -> float:
    is_info_extraction = question_type == "information_extraction"
    score = overlap_score(question_terms, node)
    score += visual_detail_score(question_terms, node, is_info_extraction)

    node_type = node.get("type")
    if node_type in STATE_TYPES:
        score += 1.2
    if node_type == "Session":
        score -= 0.4
    if node_type == "Turn":
        score -= 0.8
        score -= min(len(str(node.get("text", ""))) / 4500.0, 1.2)
    return score


def ranked(nodes: Iterable[Dict[str, Any]], question_terms: Counter, question_type: str) -> List[Dict[str, Any]]:
    return sorted(nodes, key=lambda n: relevance_score(question_terms, n, question_type), reverse=True)


def unique_extend(target: List[Dict[str, Any]], candidates: Iterable[Dict[str, Any]], seen: set[str], limit: int) -> None:
    for node in candidates:
        node_id = node.get("id")
        if not node_id or node_id in seen:
            continue
        target.append(node)
        seen.add(node_id)
        if len(target) >= limit:
            return


def profile_for(question_type: str) -> Dict[str, int]:
    if question_type == "information_extraction":
        return {
            "visual_limit": 32,
            "ocr_limit": 18,
            "candidate_limit": 18,
            "state_limit": 8,
            "session_limit": 8,
            "edge_limit": 18,
        }
    if question_type == "answer_refusal":
        return {
            "visual_limit": 18,
            "ocr_limit": 10,
            "candidate_limit": 20,
            "state_limit": 10,
            "session_limit": 10,
            "edge_limit": 20,
        }
    return {
        "visual_limit": 24,
        "ocr_limit": 12,
        "candidate_limit": 24,
        "state_limit": 16,
        "session_limit": 12,
        "edge_limit": 24,
    }


def select_nodes(nodes: List[Dict[str, Any]], question_terms: Counter, question_type: str, packet_budget: int) -> List[Dict[str, Any]]:
    visual_nodes = ranked([n for n in nodes if is_visual_node(n)], question_terms, question_type)
    state_nodes = ranked([n for n in nodes if n.get("type") in STATE_TYPES], question_terms, question_type)
    candidate_nodes = ranked(
        [n for n in nodes if n.get("type") not in VISUAL_TYPES | STATE_TYPES | {"Session"} and not is_visual_node(n)],
        question_terms,
        question_type,
    )
    all_ranked = ranked(nodes, question_terms, question_type)

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    if question_type == "information_extraction":
        unique_extend(selected, visual_nodes, seen, min(32, packet_budget))
        unique_extend(selected, candidate_nodes, seen, min(50, packet_budget))
        unique_extend(selected, state_nodes, seen, min(58, packet_budget))
        unique_extend(selected, all_ranked, seen, packet_budget)
    else:
        unique_extend(selected, all_ranked, seen, packet_budget)

    return selected[:packet_budget]


def build_packet(item: Dict[str, Any], graph: Dict[str, Any], packet_budget: int) -> Dict[str, Any]:
    question_type = item.get("question_type") or "unknown"
    question_terms = Counter(tokenize(item.get("question", "")))
    nodes = [node for node in graph.get("nodes", []) if node.get("type") not in LOW_VALUE_TYPES]
    selected = select_nodes(nodes, question_terms, question_type, packet_budget)
    selected_ids = {n.get("id") for n in selected}
    profile = profile_for(question_type)

    visual = ranked([n for n in selected if is_visual_node(n)], question_terms, question_type)
    ocr_attr = [n for n in visual if has_visual_detail(n)]
    states = ranked([n for n in selected if n.get("type") in STATE_TYPES], question_terms, question_type)
    candidates = ranked(
        [n for n in selected if n.get("type") not in VISUAL_TYPES | STATE_TYPES | {"Session"} and not is_visual_node(n)],
        question_terms,
        question_type,
    )

    session_nodes = {n.get("id"): n for n in nodes if n.get("type") == "Session" and n.get("id")}
    selected_session_ids: List[str] = []
    for node in selected:
        session_id = node.get("session_id")
        if session_id and session_id not in selected_session_ids:
            selected_session_ids.append(session_id)

    session_lines: List[str] = []
    for session_id in selected_session_ids[: profile["session_limit"]]:
        session_node = session_nodes.get(session_id)
        session_lines.append(compact_node(session_node) if session_node else f"[Session] session={session_id}")

    edges = []
    for edge in graph.get("edges", []):
        if edge.get("source") in selected_ids and edge.get("target") in selected_ids:
            edges.append(f"{edge.get('source')} --{edge.get('type')}--> {edge.get('target')}")

    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": question_type,
        "question_subtype": item.get("question_subtype"),
        "contract": infer_contract(item),
        "reference_answer": item.get("answer"),
        "packet_budget": packet_budget,
        "strategy": "visual_ocr_enhanced_type_aware",
        "visual_ocr_focus_evidence": [compact_node(n, max_chars=900) for n in visual[: profile["visual_limit"]]],
        "ocr_attribute_evidence": [compact_node(n, max_chars=900) for n in ocr_attr[: profile["ocr_limit"]]],
        "top_answer_candidates": [compact_node(n) for n in candidates[: profile["candidate_limit"]]],
        "temporal_update_evidence": [compact_node(n) for n in states[: profile["state_limit"]]],
        "supporting_sessions": session_lines,
        "related_edges": edges[: profile["edge_limit"]],
        "stats": {
            "source_node_count": len(nodes),
            "packet_node_count": len(selected),
            "visual_count": len(visual[: profile["visual_limit"]]),
            "ocr_attribute_count": len(ocr_attr[: profile["ocr_limit"]]),
            "candidate_count": len(candidates[: profile["candidate_limit"]]),
            "state_count": len(states[: profile["state_limit"]]),
            "session_count": len(session_lines),
            "edge_count": len(edges[: profile["edge_limit"]]),
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
        stats.append(
            {
                "question_id": qid,
                "question_type": item.get("question_type"),
                "question_subtype": item.get("question_subtype"),
                **packet["stats"],
            }
        )

    write_csv(out_dir / "packet_stats.csv", stats)
    save_json(
        out_dir / "packet_manifest.json",
        {
            "dataset": args.dataset,
            "subgraph_dir": args.subgraph_dir,
            "packet_budget": args.packet_budget,
            "count": len(items),
            "packet_dir": str(packet_dir),
            "method": "visual_ocr_enhanced_type_aware",
        },
    )
    print(f"Wrote {len(items)} visual/OCR evidence packets to {packet_dir}")


if __name__ == "__main__":
    main()

