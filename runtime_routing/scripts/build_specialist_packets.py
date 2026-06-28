"""Build duration and arithmetic specialist packets for Runtime-routing evaluation.

The builder only uses the question and the retrieved subgraph. It never reads
answer_session_ids or the reference answer when selecting evidence or computing
a candidate answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from eval_v2 import load_items, save_json, write_csv  # noqa: arithmetic repair02


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
CURRENCY_RE = re.compile(r"(?P<currency>[$\u00a3\u20ac])\s*(?P<amount>\d+(?:\.\d{1,2})?)")
DURATION_RE = re.compile(
    r"Duration\s*1:\s*(?P<one>.*?)(?=\s*Duration\s*2:)\s*Duration\s*2:\s*(?P<two>.*?)(?=\s*A\.|\s*Answer\s+with)",
    flags=re.IGNORECASE | re.DOTALL,
)

STOPWORDS = {
    "a", "an", "and", "at", "doing", "duration", "for", "i", "in", "my", "of", "on",
    "period", "spent", "the", "time", "to", "with", "worked", "working",
}
PURCHASE_CUES = {
    "bought", "cost", "paid", "picked", "price", "purchase", "purchased", "spent",
}
NEGATIVE_CUES = {
    "budget", "could", "example", "limit", "maybe", "retail", "save", "worth",
}


def tokenize(text: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def content_terms(text: Any) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def compact(text: Any, limit: int = 700) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def node_line(node: Dict[str, Any]) -> str:
    fields = [
        f"[{node.get('type', 'Node')}]",
        f"date={node.get('date')}" if node.get("date") else "",
        f"session={node.get('session_id')}" if node.get("session_id") else "",
        compact(node.get("text", "")),
    ]
    return " ".join(field for field in fields if field)


def parse_date(value: Any) -> Optional[datetime]:
    match = DATE_RE.search(str(value or ""))
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def lexical_overlap(terms: Iterable[str], text: Any) -> int:
    available = Counter(tokenize(text))
    return sum(1 for term in set(terms) if available.get(term, 0) > 0)


def parse_duration_labels(question: str) -> Tuple[str, str]:
    normalized = question.replace("*", " ")
    match = DURATION_RE.search(normalized)
    if not match:
        return "Duration 1", "Duration 2"
    return compact(match.group("one"), 240).strip(" ;."), compact(match.group("two"), 240).strip(" ;.")


def ranked_duration_evidence(nodes: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    terms = content_terms(label)
    required_overlap = min(2, len(set(terms)))
    rows = []
    for node in nodes:
        if node.get("type") in {"Question", "Session", "Image"}:
            continue
        overlap = lexical_overlap(terms, node.get("text", ""))
        if overlap < required_overlap:
            continue
        score = 3.0 * overlap
        if node.get("type") in {"Fact", "StateVersion"}:
            score += 1.0
        if parse_date(node.get("date")):
            score += 0.5
        rows.append({"node": node, "score": score, "overlap": overlap})
    rows.sort(key=lambda row: (row["score"], str(row["node"].get("date", ""))), reverse=True)
    return rows


def best_per_session(rows: List[Dict[str, Any]], limit: int = 16) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        node = row["node"]
        key = node.get("session_id") or node.get("id")
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def duration_bounds(rows: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[datetime]]:
    dates = sorted({parse_date(row["node"].get("date")) for row in rows} - {None})
    if len(dates) < 2:
        return None, None
    return dates[0], dates[-1]


def build_duration_packet(item: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    label_one, label_two = parse_duration_labels(str(item.get("question", "")))
    nodes = list(graph.get("nodes", []))
    rows_one = best_per_session(ranked_duration_evidence(nodes, label_one))
    rows_two = best_per_session(ranked_duration_evidence(nodes, label_two))
    start_one, end_one = duration_bounds(rows_one)
    start_two, end_two = duration_bounds(rows_two)

    answer = None
    days_one = None
    days_two = None
    confidence = "low"
    if start_one and end_one:
        days_one = (end_one - start_one).days
    if start_two and end_two:
        days_two = (end_two - start_two).days
    if days_one is not None and days_two is not None:
        if days_one != days_two:
            answer = "A" if days_one > days_two else "B"
            disjoint_sessions = {
                row["node"].get("session_id") for row in rows_one
            }.isdisjoint({row["node"].get("session_id") for row in rows_two})
            if len(rows_one) >= 2 and len(rows_two) >= 2 and disjoint_sessions:
                confidence = "high"
            else:
                confidence = "medium"

    calculation = {
        "duration_1_start": start_one.strftime("%Y/%m/%d") if start_one else None,
        "duration_1_end": end_one.strftime("%Y/%m/%d") if end_one else None,
        "duration_1_days": days_one,
        "duration_2_start": start_two.strftime("%Y/%m/%d") if start_two else None,
        "duration_2_end": end_two.strftime("%Y/%m/%d") if end_two else None,
        "duration_2_days": days_two,
        "programmatic_answer": answer,
        "confidence": confidence,
    }
    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "question_subtype": item.get("question_subtype"),
        "contract": "ab",
        "strategy": "duration_specialist_v1",
        "duration_1_label": label_one,
        "duration_2_label": label_two,
        "duration_1_evidence": [node_line(row["node"]) for row in rows_one],
        "duration_2_evidence": [node_line(row["node"]) for row in rows_two],
        "calculation": calculation,
        "stats": {
            "duration_1_evidence_count": len(rows_one),
            "duration_2_evidence_count": len(rows_two),
            "programmatic_confidence": confidence,
        },
    }


def parse_spending_target(question: str) -> str:
    match = re.search(r"spent\s+on\s+(.+?)(?:\?|\n|$)", question, flags=re.IGNORECASE)
    return compact(match.group(1), 240).strip(" .") if match else question


def question_month(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if "this month" not in str(item.get("question", "")).lower():
        return None
    date = parse_date(item.get("question_date"))
    return (date.year, date.month) if date else None


def amount_candidates(item: Dict[str, Any], graph: Dict[str, Any], target: str) -> List[Dict[str, Any]]:
    terms = content_terms(target)
    month_filter = question_month(item)
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if node.get("type") in {"Question", "Session", "Image"}:
            continue
        text = str(node.get("text", ""))
        matches = list(CURRENCY_RE.finditer(text))
        if not matches:
            continue
        node_date = parse_date(node.get("date"))
        if month_filter and (not node_date or (node_date.year, node_date.month) != month_filter):
            continue
        overlap = lexical_overlap(terms, text)
        lower = text.lower()
        cue_count = sum(1 for cue in PURCHASE_CUES if cue in lower)
        negative_count = sum(1 for cue in NEGATIVE_CUES if cue in lower)
        score = 2.0 * overlap + 1.5 * min(cue_count, 2) - 1.0 * negative_count
        if node.get("type") in {"Fact", "StateVersion"}:
            score += 0.5
        for match in matches:
            currency = match.group("currency")
            amount = Decimal(match.group("amount"))
            session_id = str(node.get("session_id") or node.get("id"))
            key = (session_id, currency, str(amount.normalize()))
            candidate = {
                "session_id": node.get("session_id"),
                "date": node.get("date"),
                "currency": currency,
                "amount": str(amount),
                "score": score,
                "target_overlap": overlap,
                "purchase_cues": cue_count,
                "evidence": node_line(node),
            }
            if key not in dedup or score > dedup[key]["score"]:
                dedup[key] = candidate
    return sorted(dedup.values(), key=lambda row: (row["score"], row["date"] or ""), reverse=True)


def build_arithmetic_packet(item: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    target = parse_spending_target(str(item.get("question", "")))
    candidates = amount_candidates(item, graph, target)
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"a{index:02d}"
    selected = [row for row in candidates if row["target_overlap"] > 0 and row["purchase_cues"] > 0 and row["score"] >= 3.5]
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_session[str(row.get("session_id"))].append(row)
    ambiguous_sessions = [sid for sid, rows in by_session.items() if len(rows) > 1]
    currencies = {row["currency"] for row in selected}

    answer = None
    confidence = "low"
    total = None
    if selected and len(currencies) == 1:
        total = sum((Decimal(row["amount"]) for row in selected), Decimal("0"))
        currency = next(iter(currencies))
        answer = f"{currency}{total:.2f}"
        # Completeness cannot be proven from a retrieved subgraph alone. arithmetic repair
        # therefore uses model selection plus deterministic summation.
        confidence = "medium" if not ambiguous_sessions else "low"

    calculation = {
        "target": target,
        "selected_amounts": [f"{row['currency']}{Decimal(row['amount']):.2f}" for row in selected],
        "candidate_sum": f"{next(iter(currencies))}{total:.2f}" if total is not None and len(currencies) == 1 else None,
        "programmatic_answer": answer,
        "confidence": confidence,
        "ambiguous_sessions": ambiguous_sessions,
    }
    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "question_subtype": item.get("question_subtype"),
        "contract": "number",
        "strategy": "arithmetic_specialist_v1",
        "target": target,
        "amount_candidates": candidates[:24],
        "selected_purchase_evidence": [row["evidence"] for row in selected[:16]],
        "calculation": calculation,
        "stats": {
            "amount_candidate_count": len(candidates),
            "selected_amount_count": len(selected),
            "programmatic_confidence": confidence,
        },
    }


def build_packet(item: Dict[str, Any], graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    subtype = item.get("question_subtype")
    if subtype == "duration_comparison":
        return build_duration_packet(item, graph)
    if subtype == "arithmetic":
        return build_arithmetic_packet(item, graph)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subgraph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    items = load_items(args.dataset)
    if args.max_samples:
        items = items[: args.max_samples]
    items = [item for item in items if item.get("question_subtype") in {"duration_comparison", "arithmetic"}]

    out_dir = Path(args.output_dir)
    packet_dir = out_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for item in items:
        qid = item.get("question_id")
        graph_path = Path(args.subgraph_dir) / f"{qid}.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        packet = build_packet(item, graph)
        if packet is None:
            continue
        save_json(packet_dir / f"{qid}.json", packet)
        stats.append(
            {
                "question_id": qid,
                "question_subtype": item.get("question_subtype"),
                **packet.get("stats", {}),
                "programmatic_answer": packet.get("calculation", {}).get("programmatic_answer"),
            }
        )

    write_csv(out_dir / "specialist_stats.csv", stats)
    save_json(
        out_dir / "specialist_manifest.json",
        {
            "dataset": args.dataset,
            "subgraph_dir": args.subgraph_dir,
            "count": len(stats),
            "packet_dir": str(packet_dir),
            "label_leakage": False,
        },
    )
    print(f"Wrote {len(stats)} specialist packets to {packet_dir}")


if __name__ == "__main__":
    main()

