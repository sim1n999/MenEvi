"""Validate complete, label-blind holdout assets before Reliability-gate validation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "memlens_repro" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from experiment_utils import image_key, load_items  # noqa: arithmetic repair02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.json")):
        digest.update(file.name.encode("utf-8"))
        digest.update(sha256(file).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--caption-cache", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    expected = {str(item["question_id"]) for item in items}
    if len(expected) != 594:
        raise RuntimeError(f"Expected 594 holdout IDs, found {len(expected)}")
    images = set()
    for item in items:
        for session in item.get("haystack_sessions", []):
            turns = session.get("session", []) if isinstance(session, dict) else session
            for turn in turns:
                images.update(image_key(image) for image in turn.get("images", []) or [])

    caption_ids = set()
    caption_path = Path(args.caption_cache)
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            caption_ids.add(str(json.loads(line)["image_id"]))
    if caption_ids != images:
        raise RuntimeError(
            f"Caption coverage mismatch: expected={len(images)} actual={len(caption_ids)}"
        )

    root = Path(args.asset_root)
    directories = {
        "graphs": root / "kg_memory" / "graphs",
        "subgraphs": root / "retrieval_budget120" / "retrieved_subgraphs",
        "c_packets": root / "c_packets" / "packets",
        "d_packets": root / "d_packets" / "packets",
        "specialist_packets": root / "specialist_packets" / "packets",
    }
    specialist_expected = {
        str(item["question_id"])
        for item in items
        if item.get("question_subtype") in {"arithmetic", "duration_comparison"}
    }
    expected_by_asset = {
        name: (specialist_expected if name == "specialist_packets" else expected)
        for name in directories
    }
    counts = {}
    for name, directory in directories.items():
        ids = {path.stem for path in directory.glob("*.json")}
        required = expected_by_asset[name]
        if ids != required:
            raise RuntimeError(
                f"{name} ID mismatch: missing={len(required - ids)} extra={len(ids - required)}"
            )
        counts[name] = len(ids)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("hit_count") != 0:
        raise RuntimeError("Baseline prompt leakage audit failed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "ready_for_freeze",
        "holdout_count": len(expected),
        "unique_image_count": len(images),
        "caption_count": len(caption_ids),
        "caption_cache_sha256": sha256(caption_path),
        "asset_counts": counts,
        "asset_directory_sha256": {
            name: directory_sha256(directory)
            for name, directory in directories.items()
        },
        "baseline_prompt_leakage_audit_passed": True,
        "selection_uses_labels": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated I holdout assets: {output}")


if __name__ == "__main__":
    main()



