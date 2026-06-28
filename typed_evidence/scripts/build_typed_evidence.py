"""Compile full memory graphs into typed evidence records for Typed-evidence evaluation.

Selection uses question text and graph content only. Reference answers and
answer_session_ids are deliberately absent from every compiler function.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TARGET_SUBTYPES = {"arithmetic", "duration_comparison", "entity", "previnfo"}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
CURRENCY_RE = re.compile(r"(?P<currency>[$£€])\s*(?P<amount>\d+(?:\.\d{1,2})?)")
DURATION_RE = re.compile(
    r"Duration\s*1:\s*(?P<one>.*?)(?=\s*Duration\s*2:)\s*Duration\s*2:\s*"
    r"(?P<two>.*?)(?=\s*A\.|\s*Answer\s+with)",
    flags=re.I | re.S,
)
PURCHASE_PATTERNS = (
    r"\b(?:bought|purchased|ordered|paid|pay|spent|picked up|grabbed|got|acquired|checked out)\b",
    r"\b(?:cost me|total came to|charged)\b",
)
NEGATIVE_PATTERNS = (
    r"\b(?:budget|hypothetical|could buy|might buy|plan to|planning to|"
    r"retail price|list price|worth|save up|considering)\b",
)
START_PATTERNS = (r"\bstarted\b", r"\bbegan\b", r"\bfrom\b", r"\bsince\b", r"\bmoved in\b", r"\bjoined\b")
END_PATTERNS = (r"\bended\b", r"\bfinished\b", r"\buntil\b", r"\bleft\b", r"\bcompleted\b", r"\bmoved out\b")
GENERIC = {
    "a", "an", "and", "answer", "at", "duration", "exact", "following", "for",
    "from", "have", "how", "i", "image", "in", "is", "it", "longer", "me",
    "my", "of", "on", "only", "period", "phrase", "question", "short", "side",
    "spent", "the", "this", "time", "to", "total", "two", "what", "which", "with",
}
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_items(path: str | Path) -> List[Dict[str, Any]]:
    raw = load_json(path)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


def tokens(value: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value or ""))]


def terms(value: Any) -> List[str]:
    return [token for token in tokens(value) if token not in GENERIC and len(token) > 1]


def compact(value: Any, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def overlap(query_terms: Iterable[str], value: Any) -> int:
    available = Counter(tokens(value))
    return sum(1 for token in set(query_terms) if available[token])


def parse_date(value: Any) -> Optional[datetime]:
    match = DATE_RE.search(str(value or ""))
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def date_strings(value: Any) -> List[str]:
    output = []
    for year, month, day in DATE_RE.findall(str(value or "")):
        try:
            output.append(datetime(int(year), int(month), int(day)).strftime("%Y/%m/%d"))
        except ValueError:
            continue
    return output


def requested_month(item: Dict[str, Any]) -> Optional[int]:
    question = str(item.get("question", "")).lower()
    for name, number in MONTHS.items():
        if re.search(rf"\b{name}\b", question):
            return number
    if "this month" in question:
        date = parse_date(item.get("question_date"))
        return date.month if date else None
    return None


def spending_target(question: str) -> str:
    match = re.search(r"spent\s+on\s+(.+?)(?:\s+this\s+month|\s+for\s+this|\?|\n|$)", question, flags=re.I)
    return compact(match.group(1), 180).strip(" .") if match else compact(question, 180)


def node_record(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": node.get("id"),
        "node_type": node.get("type"),
        "session_id": node.get("session_id"),
        "session_date": node.get("date"),
        "text": compact(node.get("text")),
    }


def session_context(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        sid = node.get("session_id")
        if sid and node.get("type") in {"Fact", "StateVersion", "VisualFact"}:
            grouped[str(sid)].append(str(node.get("text", "")))
    return {sid: compact(" ".join(parts), 5000) for sid, parts in grouped.items()}


def build_purchase_events(item: Dict[str, Any], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    target = spending_target(str(item.get("question", "")))
    target_terms = terms(target)
    month = requested_month(item)
    contexts = session_context(nodes)
    candidates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for node in nodes:
        if node.get("type") not in {"Fact", "StateVersion", "Turn"}:
            continue
        text = str(node.get("text", ""))
        amounts = list(CURRENCY_RE.finditer(text))
        if not amounts:
            continue
        sid = str(node.get("session_id") or node.get("id"))
        context = contexts.get(sid, text)
        node_overlap = overlap(target_terms, text)
        context_overlap = overlap(target_terms, context)
        purchase_cues = sum(bool(re.search(pattern, text, flags=re.I)) for pattern in PURCHASE_PATTERNS)
        negative_cues = sum(bool(re.search(pattern, text, flags=re.I)) for pattern in NEGATIVE_PATTERNS)
        node_date = parse_date(node.get("date"))
        month_match = month is None or bool(node_date and node_date.month == month)

        for amount_match in amounts:
            amount = Decimal(amount_match.group("amount"))
            currency = amount_match.group("currency")
            key = (sid, currency, str(amount.normalize()))
            score = 3.0 * node_overlap + 1.25 * context_overlap + 2.5 * purchase_cues
            if node.get("type") in {"Fact", "StateVersion"}:
                score += 1.0
            score -= 3.0 * negative_cues
            if not month_match:
                score -= 5.0
            status = "completed" if purchase_cues and not negative_cues else "uncertain"
            candidate = {
                "record_type": "PurchaseEvent",
                "event_id": "",
                "target": target,
                "currency": currency,
                "amount": str(amount),
                "status": status,
                "session_id": node.get("session_id"),
                "event_date": node.get("date"),
                "requested_month": month,
                "month_match": month_match,
                "target_overlap": max(node_overlap, context_overlap),
                "purchase_cue_count": purchase_cues,
                "negative_cue_count": negative_cues,
                "score": round(score, 3),
                "evidence": compact(text, 1000),
                "session_context": compact(context, 1400),
                "source_node_id": node.get("id"),
            }
            if key not in candidates or score > float(candidates[key]["score"]):
                candidates[key] = candidate

    ranked = sorted(candidates.values(), key=lambda row: (row["score"], row["event_date"] or ""), reverse=True)
    for index, row in enumerate(ranked, 1):
        row["event_id"] = f"purchase_{index:02d}"
    return {
        "record_type": "PurchaseEvidenceSet",
        "target": target,
        "requested_month": month,
        "events": ranked[:40],
        "compiler_note": "Events are candidates; the selector must retain only completed, target-matching purchases.",
    }


def duration_labels(question: str) -> Tuple[str, str]:
    match = DURATION_RE.search(question.replace("*", " "))
    if not match:
        return "Duration 1", "Duration 2"
    return compact(match.group("one"), 220).strip(" .;"), compact(match.group("two"), 220).strip(" .;")


def boundary_kind(text: str) -> str:
    has_start = any(re.search(pattern, text, flags=re.I) for pattern in START_PATTERNS)
    has_end = any(re.search(pattern, text, flags=re.I) for pattern in END_PATTERNS)
    if has_start and not has_end:
        return "start"
    if has_end and not has_start:
        return "end"
    if has_start and has_end:
        return "range"
    return "observation"


def build_duration_boundaries(item: Dict[str, Any], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = duration_labels(str(item.get("question", "")))
    sets = []
    contexts = session_context(nodes)
    for index, label in enumerate(labels, 1):
        label_terms = terms(label)
        rows = []
        for node in nodes:
            if node.get("type") not in {"Fact", "StateVersion", "Turn", "VisualFact"}:
                continue
            text = str(node.get("text", ""))
            sid = str(node.get("session_id") or "")
            context_score = overlap(label_terms, contexts.get(sid, ""))
            local_score = overlap(label_terms, text)
            if not local_score and not context_score:
                continue
            kind = boundary_kind(text)
            explicit_dates = date_strings(text)
            score = 3.0 * local_score + context_score
            if kind in {"start", "end", "range"}:
                score += 2.0
            if explicit_dates:
                score += 2.0
            if node.get("type") == "StateVersion":
                score += 0.5
            rows.append({
                "record_type": "DurationBoundary",
                "boundary_id": "",
                "duration_index": index,
                "label": label,
                "boundary_kind": kind,
                "explicit_dates": explicit_dates,
                "observation_date": node.get("date"),
                "session_id": node.get("session_id"),
                "score": round(score, 3),
                "evidence": compact(text, 1100),
                "source_node_id": node.get("id"),
            })
        rows.sort(key=lambda row: (row["score"], row["observation_date"] or ""), reverse=True)
        for row_index, row in enumerate(rows[:36], 1):
            row["boundary_id"] = f"duration_{index}_{row_index:02d}"
        sets.append({"duration_index": index, "label": label, "boundaries": rows[:36]})
    return {
        "record_type": "DurationEvidenceSet",
        "durations": sets,
        "compiler_note": "The model must choose genuine start/end boundaries; no min/max answer is precomputed.",
    }


def visual_query_terms(question: str) -> List[str]:
    return terms(re.sub(r"\b(left|right|top|bottom|image|photo|shown|visible)\b", " ", question, flags=re.I))


def resolve_image(image_id: str, image_dir: Path) -> Optional[str]:
    candidate = image_dir / image_id
    if candidate.is_file():
        return str(candidate.resolve())
    by_name = list(image_dir.rglob(Path(image_id).name))
    return str(by_name[0].resolve()) if by_name else None


def build_visual_targets(item: Dict[str, Any], nodes: List[Dict[str, Any]], image_dir: Path) -> Dict[str, Any]:
    query = visual_query_terms(str(item.get("question", "")))
    contexts = session_context(nodes)
    images: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if node.get("type") == "Image":
            image_id = str(node.get("text") or node.get("id", "").removeprefix("image:"))
            images[image_id] = {
                "record_type": "VisualTarget",
                "target_id": "",
                "image_id": image_id,
                "image_path": resolve_image(image_id, image_dir),
                "session_id": node.get("session_id"),
                "session_date": node.get("date"),
                "visual_fact": "",
                "score": 0.0,
            }
    for node in nodes:
        if node.get("type") != "VisualFact":
            continue
        image_id = str(node.get("image_id") or "")
        if image_id not in images:
            images[image_id] = {
                "record_type": "VisualTarget", "target_id": "", "image_id": image_id,
                "image_path": resolve_image(image_id, image_dir), "session_id": node.get("session_id"),
                "session_date": node.get("date"), "visual_fact": "", "score": 0.0,
            }
        images[image_id]["visual_fact"] = compact(node.get("text"), 1300)

    for row in images.values():
        context = contexts.get(str(row.get("session_id") or ""), "")
        visual_overlap = overlap(query, row.get("visual_fact", ""))
        context_overlap = overlap(query, context)
        row["score"] = round(4.0 * visual_overlap + 1.5 * context_overlap, 3)
        row["session_context"] = compact(context, 1600)
    ranked = sorted(images.values(), key=lambda row: (row["score"], row["session_date"] or ""), reverse=True)
    for index, row in enumerate(ranked[:8], 1):
        row["target_id"] = f"visual_{index:02d}"
    return {
        "record_type": "VisualEvidenceSet",
        "question_focus": str(item.get("question", "")).splitlines()[0],
        "targets": ranked[:8],
        "compiler_note": "Inspect these images with the question, then answer only from grounded observations.",
    }


def build_packet(item: Dict[str, Any], graph: Dict[str, Any], image_dir: Path) -> Dict[str, Any]:
    subtype = str(item.get("question_subtype"))
    nodes = list(graph.get("nodes", []))
    packet = {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "question_subtype": subtype,
        "strategy": "typed_evidence_compiler_v1",
        "label_leakage": False,
    }
    if subtype == "arithmetic":
        packet["typed_evidence"] = build_purchase_events(item, nodes)
    elif subtype == "duration_comparison":
        packet["typed_evidence"] = build_duration_boundaries(item, nodes)
    elif subtype in {"entity", "previnfo"}:
        packet["typed_evidence"] = build_visual_targets(item, nodes, image_dir)
    else:
        raise ValueError(f"Unsupported subtype: {subtype}")
    return packet


def write_stats(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    items = [item for item in load_items(args.dataset) if item.get("question_subtype") in TARGET_SUBTYPES]
    if args.max_samples:
        items = items[:args.max_samples]
    output_dir = Path(args.output_dir)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for item in items:
        qid = str(item["question_id"])
        graph = load_json(Path(args.graph_dir) / f"{qid}.json")
        packet = build_packet(item, graph, Path(args.image_dir))
        save_json(packet_dir / f"{qid}.json", packet)
        evidence = packet["typed_evidence"]
        count = len(evidence.get("events", evidence.get("durations", evidence.get("targets", []))))
        stats.append({"question_id": qid, "question_subtype": item.get("question_subtype"),
                      "record_type": evidence["record_type"], "top_level_record_count": count})

    write_stats(output_dir / "compiler_stats.csv", stats)
    save_json(output_dir / "manifest.json", {
        "strategy": "typed_evidence_compiler_v1",
        "dataset": args.dataset,
        "graph_dir": args.graph_dir,
        "image_dir": args.image_dir,
        "count": len(stats),
        "target_subtypes": sorted(TARGET_SUBTYPES),
        "label_leakage": False,
        "forbidden_selection_fields": ["answer", "answer_session_ids"],
    })
    print(f"Wrote {len(stats)} typed packets to {packet_dir}")


if __name__ == "__main__":
    main()
