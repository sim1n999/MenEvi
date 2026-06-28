"""Build a Qwen2.5-VL image caption cache for MemLens images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

REPRO_ROOT = Path(__file__).resolve().parents[1]
MEMLENS_DIR = REPRO_ROOT / "MEMLENS"
if str(MEMLENS_DIR) not in sys.path:
    sys.path.insert(0, str(MEMLENS_DIR))

from experiment_utils import image_key, load_items, write_jsonl  # noqa: arithmetic repair02
from utils import resolve_image_path  # noqa: arithmetic repair02
from vlm_models import load_LLM  # noqa: arithmetic repair02


CAPTION_SYSTEM = (
    "You are a careful visual memory extractor. Extract only visually grounded facts. "
    "Return compact JSON with keys short_caption, visible_objects, visible_text, "
    "attributes, possible_memory_facts, uncertain_observations."
)


def safe_parse_json(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    return {"short_caption": text.strip(), "raw_output": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    seen = {}
    for item in items:
        for session in item.get("haystack_sessions", []):
            turns = session.get("session", []) if isinstance(session, dict) else session
            for turn in turns:
                text = (turn.get("content") or "").replace("<image>", "").strip()
                for img in turn.get("images", []) or []:
                    key = image_key(img)
                    if key not in seen:
                        seen[key] = {"img": img, "context": text[:800]}

    rows = []
    if args.dry_run:
        for key, meta in seen.items():
            rows.append(
                {
                    "image_id": key,
                    "image_path": resolve_image_path(meta["img"], args.image_dir) or key,
                    "short_caption": meta["img"].get("blip_caption", "") if isinstance(meta["img"], dict) else "",
                    "visible_objects": [],
                    "visible_text": [],
                    "attributes": [],
                    "possible_memory_facts": [],
                }
            )
        write_jsonl(args.output, rows)
        print(f"Wrote dry-run caption cache for {len(rows)} images to {args.output}")
        return

    model_args = SimpleNamespace(
        model_name_or_path=args.model,
        temperature=0.0,
        top_p=1.0,
        input_max_length=4096,
        generation_max_length=args.generation_max_length,
        generation_min_length=0,
        do_sample=False,
        stop_newline=False,
        use_chat_template=True,
        dtype=args.dtype,
        device_map="auto",
        max_memory=None,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=False,
        offload_folder=None,
        use_yarn=False,
        do_prefill=False,
        use_gradient_checkpointing=False,
        repetition_penalty=None,
        vision_chunk_size=1,
        disable_vision_chunking=False,
        max_image_size=args.max_image_size,
    )
    model = load_LLM(model_args)

    for key, meta in tqdm(seen.items(), desc="Caption images"):
        path = resolve_image_path(meta["img"], args.image_dir)
        if not path:
            continue
        prompt = (
            f"{CAPTION_SYSTEM}\n\n"
            f"Surrounding dialogue context:\n{meta['context']}\n\n"
            "Return JSON only."
        )
        item = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": CAPTION_SYSTEM}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": path}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        }
        inputs = model.prepare_inputs(item, {})
        out = model.generate(inputs=inputs)
        parsed = safe_parse_json(out.get("output", ""))
        parsed.update({"image_id": key, "image_path": path})
        rows.append(parsed)

    write_jsonl(args.output, rows)
    print(f"Wrote caption cache for {len(rows)} images to {args.output}")


if __name__ == "__main__":
    main()
