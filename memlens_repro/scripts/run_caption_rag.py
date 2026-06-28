"""Caption-augmented BM25 RAG baseline for MemLens."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from experiment_utils import (
    TextGenerator,
    build_answer_prompt,
    finalize_run,
    load_caption_cache,
    load_items,
    retrieve_sessions,
    score_prediction,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--caption-source", choices=["dataset", "qwen_vl"], default="dataset")
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    cache = load_caption_cache(args.caption_cache)
    source = "dataset" if args.caption_source == "dataset" else "qwen_vl"
    generator = None if args.dry_run else TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)

    results = []
    logs = []
    start = time.time()
    for item in tqdm(items, desc=f"{args.caption_source} caption RAG"):
        selected, log = retrieve_sessions(item, args.top_k, caption_source=source, caption_cache=cache)
        logs.append(log)
        context = "\n\n".join(doc["text"] for doc in selected)
        prompt = build_answer_prompt(item, context, "You are answering from text plus image-caption memory.")
        if args.dry_run:
            raw = "Insufficient information"
            gen = {"input_len": len(prompt.split()), "output_len": len(raw.split())}
        else:
            gen = generator.generate(prompt, args.generation_max_length)
            raw = gen["output"]
        scored = score_prediction(raw, item.get("answer", ""))
        results.append(
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "question_type": item.get("question_type"),
                "question_subtype": item.get("question_subtype"),
                "reference_answer": item.get("answer", ""),
                "raw_prediction": raw,
                **scored,
                "input_len": gen["input_len"],
                "output_len": gen["output_len"],
            }
        )

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "retrieval_logs.jsonl", logs)
    finalize_run(args, results, start, out_dir)
    print(f"Wrote predictions and metrics to {out_dir}")


if __name__ == "__main__":
    main()
