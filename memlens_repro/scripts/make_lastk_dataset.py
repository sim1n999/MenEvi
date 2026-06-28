"""Create a MemLens derived dataset keeping only the last k sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import load_items, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    items = load_items(args.input)
    out = []
    for item in items:
        new_item = dict(item)
        for key in ("haystack_sessions", "haystack_session_ids", "haystack_dates"):
            vals = list(item.get(key, []))
            new_item[key] = vals[-args.k :] if args.k > 0 else []
        out.append(new_item)

    save_json(args.output, out)
    print(f"Wrote {len(out)} items to {Path(args.output)}")


if __name__ == "__main__":
    main()
