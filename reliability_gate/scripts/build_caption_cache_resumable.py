"""Build a resumable, label-blind visual caption cache for I holdout assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPRO = Path(__file__).resolve().parents[2] / "memlens_repro"
MEMLENS = REPRO / "MEMLENS"
for path in (REPRO / "scripts", MEMLENS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_utils import image_key, load_items  # noqa: arithmetic repair02
from utils import resolve_image_path  # noqa: arithmetic repair02
from vlm_models import load_LLM  # noqa: arithmetic repair02

SYSTEM = (
    "You are a careful visual memory extractor. Extract only visually grounded "
    "facts. Return compact JSON with keys short_caption, visible_objects, "
    "visible_text, attributes, possible_memory_facts, uncertain_observations."
)


def parse_output(text: str) -> dict:
    try:
        start, end = text.find("{"), text.rfind("}")
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    except (ValueError, json.JSONDecodeError):
        pass
    return {"short_caption": text.strip(), "raw_output": text}


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()

    ids = set()
    good_lines = []
    malformed = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            image_id = row.get("image_id")
            if image_id is None:
                raise KeyError("missing image_id")
        except (json.JSONDecodeError, KeyError) as error:
            malformed.append({
                "line": number,
                "error": str(error),
                "content": line,
            })
            continue
        ids.add(str(image_id))
        good_lines.append(json.dumps(row, ensure_ascii=False))

    if malformed:
        quarantine = path.with_name(path.name + ".malformed_lines.json")
        repair = path.with_name(path.name + ".repair")
        quarantine.write_text(
            json.dumps(
                {"source": str(path), "malformed_lines": malformed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        repair.write_text(
            "\n".join(good_lines) + ("\n" if good_lines else ""),
            encoding="utf-8",
        )
        repair.replace(path)
        print(
            "Repaired malformed caption checkpoint: "
            f"kept {len(good_lines)} rows, quarantined {len(malformed)} bad lines at {quarantine}",
            flush=True,
        )

    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-dir")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not args.prompt_only and not args.model:
        parser.error("--model is required unless --prompt-only is set")

    images = {}
    for item in load_items(args.input):
        safe_item = {
            key: value for key, value in item.items()
            if key not in {"answer", "answer_session_ids"}
        }
        for session in safe_item.get("haystack_sessions", []):
            turns = session.get("session", []) if isinstance(session, dict) else session
            for turn in turns:
                context = str(turn.get("content") or "").replace("<image>", "").strip()[:800]
                for image in turn.get("images", []) or []:
                    images.setdefault(image_key(image), (image, context))

    requests = Path(args.request_dir) if args.request_dir else None
    if requests:
        requests.mkdir(parents=True, exist_ok=True)
    if args.prompt_only:
        for index, (key, (_, context)) in enumerate(images.items()):
            prompt = f"{SYSTEM}\n\nSurrounding dialogue context:\n{context}\n\nReturn JSON only."
            if requests:
                (requests / f"image_{index:05d}.json").write_text(
                    json.dumps({"target_id": f"image_{index:05d}", "prompt": prompt},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        print(f"Wrote {len(images)} label-blind caption prompts")
        return

    done = existing_ids(Path(args.output))
    unexpected = done - set(images)
    if unexpected:
        raise RuntimeError(
            f"Caption checkpoint contains {len(unexpected)} unexpected image IDs"
        )
    if len(done) == len(images):
        print(f"Caption cache already complete: {len(done)}/{len(images)}")
        return
    model_args = SimpleNamespace(
        model_name_or_path=args.model, temperature=0.0, top_p=1.0,
        input_max_length=4096, generation_max_length=192,
        generation_min_length=0, do_sample=False, stop_newline=False,
        use_chat_template=True, dtype=args.dtype, device_map="auto",
        max_memory=None, attn_implementation="sdpa",
        load_in_4bit=not args.no_4bit, load_in_8bit=False,
        offload_folder=None, use_yarn=False, do_prefill=False,
        use_gradient_checkpointing=False, repetition_penalty=None,
        vision_chunk_size=1, disable_vision_chunking=False,
        max_image_size=args.max_image_size,
    )
    model = load_LLM(model_args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = len(done)
    with output.open("a", encoding="utf-8") as handle:
        for key, (image, context) in images.items():
            if key in done:
                continue
            path = resolve_image_path(image, args.image_dir)
            if not path:
                raise FileNotFoundError(f"Cannot resolve holdout image: {key}")
            prompt = f"{SYSTEM}\n\nSurrounding dialogue context:\n{context}\n\nReturn JSON only."
            request = {"messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": path}},
                    {"type": "text", "text": prompt},
                ]},
            ]}
            generated = model.generate(inputs=model.prepare_inputs(request, {}))
            row = parse_output(str(generated.get("output", "")))
            row.update({"image_id": key, "image_path": path})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if completed % 25 == 0:
                print(f"Caption checkpoint: {completed}/{len(images)}")
    print(f"Caption cache complete: {completed}/{len(images)}")


if __name__ == "__main__":
    main()

