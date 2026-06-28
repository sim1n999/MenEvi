"""Create a label-independent MemLens development/holdout protocol.

Membership is the set difference between full-dataset question IDs and the
already-touched agent-subset IDs. Labels never participate in selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise TypeError(f"Expected a JSON list or data list: {path}")
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_unique(items: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for item in items:
        qid = str(item.get("question_id", ""))
        if not qid:
            raise ValueError(f"Missing question_id in {label}")
        if qid in result:
            raise ValueError(f"Duplicate question_id in {label}: {qid}")
        result[qid] = item
    return result


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_protocol(
    full_path: Path,
    touched_path: Path,
    output_dir: Path,
    write_datasets: bool = False,
) -> Dict[str, Any]:
    full_items = load_items(full_path)
    touched_items = load_items(touched_path)
    full = index_unique(full_items, "full dataset")
    touched = index_unique(touched_items, "touched dataset")
    missing = sorted(set(touched) - set(full))
    if missing:
        raise ValueError(f"Touched IDs absent from full dataset: {missing[:10]}")

    # These version checks do not affect membership.
    for qid, item in touched.items():
        source = full[qid]
        for field in ("question", "question_type", "question_subtype"):
            if item.get(field) != source.get(field):
                raise ValueError(f"Dataset mismatch for {qid}, field={field}")

    touched_ids = sorted(touched)
    holdout_ids = sorted(set(full) - set(touched))
    holdout_set = set(holdout_ids)
    holdout_items = [
        item for item in full_items if str(item["question_id"]) in holdout_set
    ]

    save_json(output_dir / "touched_dev_ids.json", touched_ids)
    save_json(output_dir / "holdout_ids.json", holdout_ids)
    if write_datasets:
        save_json(output_dir / "touched_dev_32k.json", touched_items)
        save_json(output_dir / "holdout_32k.json", holdout_items)

    ids_digest = hashlib.sha256(
        ("\n".join(holdout_ids) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "protocol": "clean_holdout_v1",
        "selection_rule": "full_question_ids_minus_touched_question_ids",
        "selection_uses_labels": False,
        "full_dataset": str(full_path),
        "touched_dataset": str(touched_path),
        "full_dataset_sha256": sha256_file(full_path),
        "touched_dataset_sha256": sha256_file(touched_path),
        "holdout_ids_sha256": ids_digest,
        "full_count": len(full_items),
        "touched_dev_count": len(touched_items),
        "holdout_count": len(holdout_items),
        "write_datasets": write_datasets,
    }
    save_json(output_dir / "protocol_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dataset", required=True, type=Path)
    parser.add_argument("--touched-dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write-datasets", action="store_true")
    args = parser.parse_args()
    manifest = build_protocol(
        args.full_dataset,
        args.touched_dataset,
        args.output_dir,
        args.write_datasets,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
