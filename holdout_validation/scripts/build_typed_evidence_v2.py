"""Compile leakage-safe typed evidence records for Holdout validation.

The compiler uses question text and memory-graph content only. Reference
answers and answer-session IDs are never used for evidence selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPED_SCRIPTS = (
    PROJECT_ROOT
    / "typed_evidence"
    / "scripts"
)
if str(TYPED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TYPED_SCRIPTS))

import build_typed_evidence as gbuild  # noqa: arithmetic repair02


TARGET_SUBTYPES = {"arithmetic", "duration_comparison", "entity", "previnfo"}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compact(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build_transaction_groups(
    item: Dict[str, Any],
    nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base = gbuild.build_purchase_events(item, nodes)
    rows = []
    for event in base.get("events", []):
        rows.append(
            {
                "record_type": "TransactionGroup",
                "group_id": "",
                "currency": event.get("currency"),
                "total_amount": event.get("amount"),
                "status": event.get("status"),
                "month_match": event.get("month_match"),
                "session_id": event.get("session_id"),
                "transaction_date": event.get("event_date"),
                "target": event.get("target"),
                "target_overlap": event.get("target_overlap", 0),
                "purchase_cue_count": event.get("purchase_cue_count", 0),
                "negative_cue_count": event.get("negative_cue_count", 0),
                "score": event.get("score", 0),
                "evidence": compact(event.get("evidence")),
                "session_context": compact(
                    event.get("session_context"),
                    1600,
                ),
                "member_event_ids": [event.get("event_id")],
                "source_node_ids": [event.get("source_node_id")],
            }
        )

    rows.sort(
        key=lambda row: (
            float(row.get("score") or 0),
            str(row.get("transaction_date") or ""),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows[:40], 1):
        row["group_id"] = f"txn_{index:02d}"

    return {
        "record_type": "TransactionGroupSet",
        "target": base.get("target"),
        "requested_month": base.get("requested_month"),
        "groups": rows[:40],
        "compiler_note": (
            "Select only completed target-matching transaction groups. "
            "The deterministic tool sums selected group totals."
        ),
    }


def parse_any_date(value: Any) -> Optional[datetime]:
    return gbuild.parse_date(value)


def endpoint_date(boundary: Dict[str, Any], role: str) -> Optional[datetime]:
    explicit = list(boundary.get("explicit_dates") or [])
    value = None
    if explicit:
        value = explicit[0] if role == "start" else explicit[-1]
    if value is None:
        value = boundary.get("observation_date")
    return parse_any_date(value)


def candidate_interval(
    duration_index: int,
    label: str,
    start: Dict[str, Any],
    end: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    start_date = endpoint_date(start, "start")
    end_date = endpoint_date(end, "end")
    if start_date is None or end_date is None or end_date < start_date:
        return None

    start_kind = str(start.get("boundary_kind") or "")
    end_kind = str(end.get("boundary_kind") or "")
    if start_kind == "end" or end_kind == "start":
        return None

    days = (end_date - start_date).days
    score = float(start.get("score") or 0) + float(end.get("score") or 0)
    if start_kind in {"start", "range"}:
        score += 2.0
    if end_kind in {"end", "range"}:
        score += 2.0
    if start.get("explicit_dates"):
        score += 1.0
    if end.get("explicit_dates"):
        score += 1.0
    if (
        start.get("session_id")
        and start.get("session_id") == end.get("session_id")
    ):
        score += 0.5
    if start_kind == "observation" and end_kind == "observation":
        score -= 0.5

    return {
        "record_type": "DurationIntervalCandidate",
        "interval_id": "",
        "duration_index": duration_index,
        "label": label,
        "start_date": start_date.strftime("%Y/%m/%d"),
        "end_date": end_date.strftime("%Y/%m/%d"),
        "duration_days": days,
        "start_kind": start_kind,
        "end_kind": end_kind,
        "start_session_id": start.get("session_id"),
        "end_session_id": end.get("session_id"),
        "score": round(score, 3),
        "start_evidence": compact(start.get("evidence"), 700),
        "end_evidence": compact(end.get("evidence"), 700),
        "source_boundary_ids": [
            start.get("boundary_id"),
            end.get("boundary_id"),
        ],
    }


def build_interval_set(
    item: Dict[str, Any],
    nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base = gbuild.build_duration_boundaries(item, nodes)
    output_durations = []

    for duration in base.get("durations", []):
        index = int(duration["duration_index"])
        label = str(duration.get("label") or f"Duration {index}")
        boundaries = list(duration.get("boundaries") or [])
        best_by_span: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for start in boundaries:
            for end in boundaries:
                row = candidate_interval(index, label, start, end)
                if row is None:
                    continue
                key = (row["start_date"], row["end_date"])
                previous = best_by_span.get(key)
                if previous is None or row["score"] > previous["score"]:
                    best_by_span[key] = row

        ranked = sorted(
            best_by_span.values(),
            key=lambda row: (
                float(row["score"]),
                int(row["duration_days"]),
            ),
            reverse=True,
        )[:24]
        for rank, row in enumerate(ranked, 1):
            row["interval_id"] = f"duration_{index}_interval_{rank:02d}"

        output_durations.append(
            {
                "duration_index": index,
                "label": label,
                "intervals": ranked,
                "source_boundary_count": len(boundaries),
            }
        )

    return {
        "record_type": "DurationIntervalSet",
        "durations": output_durations,
        "compiler_note": (
            "Choose one complete interval for each duration. "
            "If either duration has no valid interval, abstain."
        ),
    }


def anonymize_visual_targets(evidence: Dict[str, Any]) -> Dict[str, Any]:
    targets = []
    for index, target in enumerate(evidence.get("targets", []), 1):
        internal = dict(target)
        internal["target_id"] = f"visual_{index:02d}"
        # Paths remain internal for pixel loading. H prompts must expose only
        # target_id and never image_id or image_path.
        targets.append(internal)
    return {
        **evidence,
        "targets": targets,
        "model_visible_identifier_policy": "target_id_only",
    }


def build_packet(
    item: Dict[str, Any],
    graph: Dict[str, Any],
    image_dir: Path,
) -> Dict[str, Any]:
    subtype = str(item.get("question_subtype"))
    nodes = list(graph.get("nodes") or [])
    packet = {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "question_subtype": subtype,
        "strategy": "h_typed_evidence_v2",
        "label_leakage": False,
        "forbidden_selection_fields": [
            "answer",
            "answer_session_ids",
        ],
    }

    if subtype == "arithmetic":
        packet["typed_evidence"] = build_transaction_groups(item, nodes)
    elif subtype == "duration_comparison":
        packet["typed_evidence"] = build_interval_set(item, nodes)
    elif subtype in {"entity", "previnfo"}:
        visual = gbuild.build_visual_targets(item, nodes, image_dir)
        packet["typed_evidence"] = anonymize_visual_targets(visual)
    else:
        raise ValueError(f"Unsupported subtype: {subtype}")
    return packet


def write_stats(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
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

    items = [
        item
        for item in gbuild.load_items(args.dataset)
        if item.get("question_subtype") in TARGET_SUBTYPES
    ]
    if args.max_samples:
        items = items[: args.max_samples]

    output_dir = Path(args.output_dir)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    stats = []

    for item in items:
        qid = str(item["question_id"])
        graph = gbuild.load_json(Path(args.graph_dir) / f"{qid}.json")
        packet = build_packet(item, graph, Path(args.image_dir))
        save_json(packet_dir / f"{qid}.json", packet)
        evidence = packet["typed_evidence"]
        if evidence["record_type"] == "TransactionGroupSet":
            count = len(evidence.get("groups", []))
        elif evidence["record_type"] == "DurationIntervalSet":
            count = sum(
                len(duration.get("intervals", []))
                for duration in evidence.get("durations", [])
            )
        else:
            count = len(evidence.get("targets", []))
        stats.append(
            {
                "question_id": qid,
                "question_subtype": item.get("question_subtype"),
                "record_type": evidence["record_type"],
                "candidate_count": count,
            }
        )

    write_stats(output_dir / "compiler_stats.csv", stats)
    save_json(
        output_dir / "manifest.json",
        {
            "strategy": "h_typed_evidence_v2",
            "dataset": args.dataset,
            "graph_dir": args.graph_dir,
            "image_dir": args.image_dir,
            "count": len(stats),
            "target_subtypes": sorted(TARGET_SUBTYPES),
            "label_leakage": False,
            "model_visible_image_identifiers": "anonymous_target_ids_only",
            "forbidden_selection_fields": [
                "answer",
                "answer_session_ids",
            ],
        },
    )
    print(f"Wrote {len(stats)} H packets to {packet_dir}")


if __name__ == "__main__":
    main()

