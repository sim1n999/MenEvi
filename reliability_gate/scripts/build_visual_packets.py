"""Build only leakage-safe entity/previnfo packets for formal Reliability-gate validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_SCRIPTS = (
    PROJECT_ROOT
    / "holdout_validation"
    / "scripts"
)
if str(HOLDOUT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HOLDOUT_SCRIPTS))

import build_typed_evidence_v2 as hbuild  # noqa: arithmetic repair02


VISUAL_SUBTYPES = {"entity", "previnfo"}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
        for item in hbuild.gbuild.load_items(args.dataset)
        if item.get("question_subtype") in VISUAL_SUBTYPES
    ]
    if args.max_samples:
        items = items[: args.max_samples]

    output_dir = Path(args.output_dir)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    stats = []

    for item in items:
        qid = str(item["question_id"])
        graph = hbuild.gbuild.load_json(
            Path(args.graph_dir) / f"{qid}.json"
        )
        packet = hbuild.build_packet(item, graph, Path(args.image_dir))
        save_json(packet_dir / f"{qid}.json", packet)
        targets = packet["typed_evidence"].get("targets", [])
        stats.append(
            {
                "question_id": qid,
                "question_subtype": item.get("question_subtype"),
                "target_count": len(targets),
                "model_visible_identifier_policy": "target_id_only",
            }
        )

    stats_path = output_dir / "compiler_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "question_subtype",
                "target_count",
                "model_visible_identifier_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(stats)

    save_json(
        output_dir / "manifest.json",
        {
            "strategy": "i_visual_packets_v1",
            "dataset": args.dataset,
            "graph_dir": args.graph_dir,
            "image_dir": args.image_dir,
            "count": len(stats),
            "target_subtypes": sorted(VISUAL_SUBTYPES),
            "label_leakage": False,
            "model_visible_image_identifiers": "anonymous_target_ids_only",
        },
    )
    print(f"Wrote {len(stats)} I visual packets to {packet_dir}")


if __name__ == "__main__":
    main()

