from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-count", type=int, default=789)
    args = parser.parse_args()
    raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = raw.get("data", raw) if isinstance(raw, dict) else raw
    ids = [str(row["question_id"]) for row in rows]
    if len(rows) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} rows, found {len(rows)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate question IDs detected")
    print(f"Validated full dataset: {args.dataset} ({len(rows)} unique questions)")


if __name__ == "__main__":
    main()

