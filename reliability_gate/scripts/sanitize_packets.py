"""Remove evaluation-only labels from generated model packet artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", action="append", required=True)
    args = parser.parse_args()
    changed = 0
    for directory in args.packet_dir:
        for path in sorted(Path(directory).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            removed = False
            for key in ("reference_answer", "answer_session_ids"):
                if key in payload:
                    del payload[key]
                    removed = True
            if removed:
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                changed += 1
    print(f"Sanitized evaluation-only labels from {changed} packet files")


if __name__ == "__main__":
    main()

