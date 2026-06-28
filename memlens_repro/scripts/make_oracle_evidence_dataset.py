"""Create an oracle-evidence MemLens dataset using answer_session_ids."""

from __future__ import annotations

import argparse

from experiment_utils import load_items, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = []
    for item in load_items(args.input):
        answer_ids = set(item.get("answer_session_ids") or [])
        new_item = dict(item)
        sessions = item.get("haystack_sessions", [])
        ids = item.get("haystack_session_ids", [])
        dates = item.get("haystack_dates", [])
        keep = [i for i, sid in enumerate(ids) if sid in answer_ids]
        new_item["haystack_sessions"] = [sessions[i] for i in keep]
        new_item["haystack_session_ids"] = [ids[i] for i in keep]
        new_item["haystack_dates"] = [dates[i] for i in keep]
        out.append(new_item)

    save_json(args.output, out)
    print(f"Wrote {len(out)} oracle-evidence items to {args.output}")


if __name__ == "__main__":
    main()
