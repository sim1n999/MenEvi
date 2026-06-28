from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--length-results", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-output", required=True)
    args = parser.parse_args()
    roots = [Path(args.profile_root), Path(args.length_results), Path(args.baseline_results)]
    profiles = []
    seen = set()
    for root in roots:
        for path in root.rglob("*.profile.json"):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            row = json.loads(path.read_text(encoding="utf-8"))
            profiles.append({"stage": row["label"], "wall_seconds": row["wall_seconds"],
                             "peak_observed_gpu_memory_mib": row.get("peak_observed_gpu_memory_mib"),
                             "returncode": row["returncode"], "profile_file": str(path)})
    write_csv(Path(args.output), sorted(profiles, key=lambda row: row["stage"]))
    storage = []
    shared = Path(args.profile_root) / "fresh_shared_assets"
    for length in ("32k", "64k", "128k", "256k"):
        path = shared / f"assets_{length}_full"
        if path.exists():
            storage.append({"length": length, "artifact_bytes": directory_size(path),
                            "artifact_mib": directory_size(path) / 2**20})
    write_csv(Path(args.storage_output), storage)


if __name__ == "__main__":
    main()
