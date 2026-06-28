#!/usr/bin/env python3
"""Create the canonical 195-question MEMLENS agent subset JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("32k", "64k", "128k", "256k")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--subset-file",
        type=Path,
        default=None,
        help="Defaults to <data-root>/agent_subset_195.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="Context-length datasets to filter.",
    )
    args = parser.parse_args()

    subset_file = args.subset_file or args.data_root / "agent_subset_195.json"
    subset = load_json(subset_file)
    qids = subset["question_ids"]
    qid_set = set(qids)

    args.out_root.mkdir(parents=True, exist_ok=True)
    dump_json(subset, args.out_root / "agent_subset_195.json")

    print(f"Loaded {len(qids)} canonical question ids from {subset_file}")
    for name in args.datasets:
        src = args.data_root / f"dataset_{name}.json"
        dst = args.out_root / f"dataset_{name}.json"
        data = load_json(src)
        filtered = [item for item in data if item.get("question_id") in qid_set]

        missing = [qid for qid in qids if qid not in {x.get("question_id") for x in filtered}]
        if missing:
            raise SystemExit(
                f"{src} is missing {len(missing)} subset ids; first missing id: {missing[0]}"
            )

        # Preserve the canonical subset order for reproducible runs.
        by_id = {item["question_id"]: item for item in filtered}
        ordered = [by_id[qid] for qid in qids]
        dump_json(ordered, dst)
        print(f"Wrote {len(ordered)} items: {dst}")


if __name__ == "__main__":
    main()
