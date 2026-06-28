"""Run question-conditioned visual inspection for G visual targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = PROJECT_ROOT / "memlens_repro"
MEMLENS_DIR = REPRO_ROOT / "MEMLENS"
for path in (REPRO_ROOT / "scripts", MEMLENS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_utils import write_jsonl  # noqa: arithmetic repair02
from vlm_models import load_LLM  # noqa: arithmetic repair02


def safe_json(text: str) -> Dict[str, Any]:
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(text[start:end + 1])
            if isinstance(value, dict):
                return value
    except (ValueError, json.JSONDecodeError):
        pass
    return {"raw_observation": text.strip()}


def prompt_for(question: str) -> str:
    return (
        "Inspect this image specifically for the memory question below. "
        "Do not answer from world knowledge or surrounding dialogue. Return JSON only with keys "
        "relevant (true/false), grounded_observation, visible_text, spatial_relation, count, "
        "colors, uncertainty. Record only details that are actually visible.\n\n"
        f"Memory question: {question}"
    )


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=args.model, temperature=0.0, top_p=1.0,
        input_max_length=4096, generation_max_length=args.generation_max_length,
        generation_min_length=0, do_sample=False, stop_newline=False,
        use_chat_template=True, dtype=args.dtype, device_map="auto", max_memory=None,
        attn_implementation=args.attn_implementation, load_in_4bit=not args.no_4bit,
        load_in_8bit=False, offload_folder=None, use_yarn=False, do_prefill=False,
        use_gradient_checkpointing=False, repetition_penalty=None, vision_chunk_size=1,
        disable_vision_chunking=False, max_image_size=args.max_image_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generation-max-length", type=int, default=192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    if not args.prompt_only and not args.model:
        parser.error("--model is required unless --prompt-only is set")

    packet_paths = []
    for path in sorted(Path(args.packet_dir).glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("question_subtype") in {"entity", "previnfo"}:
            packet_paths.append(path)
    if args.max_samples:
        packet_paths = packet_paths[:args.max_samples]

    output_dir = Path(args.output_dir)
    request_dir = output_dir / "requests"
    observation_dir = output_dir / "observations"
    request_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)
    model = None if args.prompt_only else load_LLM(model_args(args))
    all_rows = []

    for packet_path in packet_paths:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        qid, question = str(packet["question_id"]), str(packet["question"])
        rows = []
        for target in packet["typed_evidence"].get("targets", [])[:args.top_k]:
            prompt = prompt_for(question)
            request = {
                "question_id": qid,
                "target_id": target.get("target_id"),
                "image_id": target.get("image_id"),
                "image_path": target.get("image_path"),
                "prompt": prompt,
            }
            (request_dir / f"{qid}_{target.get('target_id')}.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if args.prompt_only:
                row = {**request, "execution_mode": "prompt_only"}
            elif not target.get("image_path"):
                row = {**request, "execution_mode": "missing_image", "observation": {}}
            else:
                item = {"messages": [
                    {"role": "system", "content": [{"type": "text", "text": "You are a precise visual inspector."}]},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": target["image_path"]}},
                        {"type": "text", "text": prompt},
                    ]},
                ]}
                inputs = model.prepare_inputs(item, {})
                generated = model.generate(inputs=inputs)
                raw = generated.get("output", "")
                row = {**request, "execution_mode": "model", "raw_observation": raw,
                       "observation": safe_json(raw)}
            rows.append(row)
            all_rows.append(row)
        (observation_dir / f"{qid}.json").write_text(
            json.dumps({"question_id": qid, "question": question, "data": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_jsonl(output_dir / "observations.jsonl", all_rows)
    manifest = {"packet_dir": args.packet_dir, "question_count": len(packet_paths),
                "observation_count": len(all_rows), "top_k": args.top_k,
                "execution_mode": "prompt_only" if args.prompt_only else "model"}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(all_rows)} visual inspection records to {output_dir}")


if __name__ == "__main__":
    main()
