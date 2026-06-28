from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=789)
    args = parser.parse_args()
    candidates = []
    for path in Path(args.input_dir).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if len(payload.get("data", [])) == args.expected_count:
            candidates.append((path.stat().st_mtime, path, payload))
    if not candidates:
        raise RuntimeError("No complete MemLens prediction payload found")
    _, source, payload = max(candidates)
    payload["normalized_source_file"] = str(source)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
