"""BM25 text RAG baseline for MemLens."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from experiment_utils import (
    TextGenerator,
    build_answer_prompt,
    finalize_run,
    load_items,
    retrieve_sessions,
    score_prediction,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
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
    generator = None if args.dry_run else TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)

    results = []
    retrieval_logs = []
    start = time.time()
    for item in tqdm(items, desc="BM25 text RAG"):
        selected, log = retrieve_sessions(item, args.top_k)
        retrieval_logs.append(log)
        context = "\n\n".join(doc["text"] for doc in selected)
        prompt = build_answer_prompt(item, context, "You are answering from BM25-retrieved text sessions.")
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
    write_jsonl(out_dir / "retrieval_logs.jsonl", retrieval_logs)
    payload = finalize_run(args, results, start, out_dir)
    payload["retrieval_log_count"] = len(retrieval_logs)
    print(f"Wrote predictions and metrics to {out_dir}")


if __name__ == "__main__":
    main()
