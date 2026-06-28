"""Session-summary memory baseline for MemLens."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from experiment_utils import (
    BM25Index,
    TextGenerator,
    build_answer_prompt,
    finalize_run,
    iter_sessions,
    load_caption_cache,
    load_items,
    score_prediction,
    session_to_text,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--caption-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--summary-max-length", type=int, default=192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    cache = load_caption_cache(args.caption_cache)
    generator = None if args.dry_run else TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)
    out_dir = Path(args.output_dir)

    summaries = []
    results = []
    logs = []
    start = time.time()
    for item in tqdm(items, desc="Summary memory"):
        docs = []
        for _, sid, date, turns in iter_sessions(item):
            raw_session = session_to_text(sid, date, turns, caption_source="qwen_vl", caption_cache=cache)
            if args.dry_run:
                summary = raw_session[:1200]
                gen_len = len(summary.split())
            else:
                prompt = (
                    "Summarize this dialogue session as long-term memory. Preserve names, numbers, "
                    "dates, preferences, state updates, and visual facts. Return concise JSON/text.\n\n"
                    f"{raw_session}"
                )
                gen = generator.generate(prompt, args.summary_max_length)
                summary = gen["output"]
                gen_len = gen["output_len"]
            row = {
                "question_id": item.get("question_id"),
                "session_id": sid,
                "date": date,
                "session_summary": summary,
                "output_len": gen_len,
            }
            summaries.append(row)
            docs.append({"session_id": sid, "date": date, "text": f"Session ID: {sid}\nDate: {date}\n{summary}"})

        ranked = BM25Index(docs).score(item.get("question", ""))[: args.top_k]
        selected = [{**doc, "score": score} for score, doc in ranked]
        retrieved_ids = [doc["session_id"] for doc in selected]
        answer_ids = set(item.get("answer_session_ids") or [])
        logs.append(
            {
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
                "answer_session_ids": list(answer_ids),
                "retrieved_session_ids": retrieved_ids,
                "session_hit": bool(answer_ids & set(retrieved_ids)) if answer_ids else None,
                "session_all_hit": answer_ids <= set(retrieved_ids) if answer_ids else None,
            }
        )
        context = "\n\n".join(doc["text"] for doc in selected)
        prompt = build_answer_prompt(item, context, "You are answering from retrieved session summaries.")
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

    write_jsonl(out_dir / "session_summaries.jsonl", summaries)
    write_jsonl(out_dir / "retrieval_logs.jsonl", logs)
    finalize_run(args, results, start, out_dir)
    print(f"Wrote summaries, predictions, and metrics to {out_dir}")


if __name__ == "__main__":
    main()
