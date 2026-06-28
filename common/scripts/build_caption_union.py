from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "memlens_repro" / "scripts"))
from experiment_utils import image_key  # noqa: arithmetic repair02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    images = {}
    for dataset in args.dataset:
        rows = json.loads(Path(dataset).read_text(encoding="utf-8"))
        for item in rows:
            for session in item.get("haystack_sessions", []):
                turns = session.get("session", []) if isinstance(session, dict) else session
                for turn in turns:
                    context = str(turn.get("content", "")).replace("<image>", "").strip()[:800]
                    for image in turn.get("images", []) or []:
                        images.setdefault(image_key(image), (image, context))
    union = []
    for index, (_, (image, context)) in enumerate(sorted(images.items())):
        union.append({"question_id": f"caption_union_{index:05d}", "haystack_sessions": [[{
            "role": "user", "content": context, "images": [image]
        }]]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(union, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote caption union with {len(union)} unique images")


if __name__ == "__main__":
    main()

