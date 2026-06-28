"""Run soft-refusal KG answering for MemLens.

All outputs are written to the requested output directory. Use --prompt-only to
generate prompts and prompt statistics without loading a model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from local_memlens_utils import TextGenerator, finalize_run, load_items, save_json, score_prediction, write_csv


VISUAL_TYPES = {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"}
STATE_TYPES = {"StateVersion"}


def compact_node_line(node: Dict[str, Any]) -> str:
    fields = [
        f"[{node.get('type', 'Node')}]",
        f"id={node.get('id', '')}",
        f"date={node.get('date', '')}" if node.get("date") else "",
        f"session={node.get('session_id', '')}" if node.get("session_id") else "",
        str(node.get("text", "")).strip(),
    ]
    return " ".join(x for x in fields if x).strip()


def split_graph(graph: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    current: List[str] = []
    visual: List[str] = []
    states: List[str] = []
    sessions: List[str] = []
    for node in graph.get("nodes", []):
        line = compact_node_line(node)
        node_type = node.get("type")
        if node_type in VISUAL_TYPES:
            visual.append(line)
        elif node_type in STATE_TYPES:
            states.append(line)
        elif node_type == "Session":
            sessions.append(line)
        else:
            current.append(line)

    edge_lines = []
    for edge in graph.get("edges", [])[:180]:
        edge_lines.append(f"{edge.get('source')} --{edge.get('type')}--> {edge.get('target')}")
    return sessions, current, states + edge_lines, visual


def soft_refusal_prompt(item: Dict[str, Any], graph: Dict[str, Any], policy_text: str) -> str:
    sessions, current, temporal, visual = split_graph(graph)
    return (
        "You are answering a long-term conversational memory question from a retrieved graph memory.\n"
        "Use only the graph memory below. Follow the soft-refusal policy exactly.\n\n"
        f"Soft-refusal policy:\n{policy_text.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {item.get('question_type', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n\n"
        f"Relevant sessions:\n{chr(10).join(sessions[:80]) or '- none'}\n\n"
        f"Candidate memories:\n{chr(10).join(current[:140]) or '- none'}\n\n"
        f"Temporal/update evidence:\n{chr(10).join(temporal[:160]) or '- none'}\n\n"
        f"Visual evidence:\n{chr(10).join(visual[:120]) or '- none'}\n\n"
        "Final answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--subgraph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    subgraph_dir = Path(args.subgraph_dir)
    out_dir = Path(args.output_dir)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    policy_text = Path(args.policy).read_text(encoding="utf-8")

    generator = None
    if not args.prompt_only:
        if not args.model:
            raise ValueError("--model is required unless --prompt-only is set")
        generator = TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)

    results = []
    prompt_stats = []
    start = time.time()
    for item in items:
        qid = item.get("question_id")
        graph_path = subgraph_dir / f"{qid}.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        prompt = soft_refusal_prompt(item, graph, policy_text)
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")
        input_len = len(prompt.split())
        prompt_stats.append(
            {
                "question_id": qid,
                "question_type": item.get("question_type"),
                "input_len_words": input_len,
                "node_count": len(graph.get("nodes", [])),
                "edge_count": len(graph.get("edges", [])),
            }
        )

        if args.prompt_only:
            continue

        assert generator is not None
        gen = generator.generate(prompt, args.generation_max_length)
        raw = gen["output"]
        scored = score_prediction(raw, item.get("answer", ""))
        results.append(
            {
                "question_id": qid,
                "question": item.get("question"),
                "question_type": item.get("question_type"),
                "question_subtype": item.get("question_subtype"),
                "reference_answer": item.get("answer", ""),
                "raw_prediction": raw,
                "input_len": gen["input_len"],
                "output_len": gen["output_len"],
                **scored,
            }
        )

    write_csv(out_dir / "prompt_stats.csv", prompt_stats)
    save_json(out_dir / "prompt_stats.json", prompt_stats)
    if args.prompt_only:
        save_json(
            out_dir / "prompt_only_manifest.json",
            {
                "args": vars(args),
                "count": len(prompt_stats),
                "avg_input_len_words": (
                    sum(row["input_len_words"] for row in prompt_stats) / len(prompt_stats)
                    if prompt_stats
                    else 0.0
                ),
            },
        )
        print(f"Wrote prompt-only outputs to {out_dir}")
        return

    finalize_run(args, results, start, out_dir, extra={"prompt_stats": prompt_stats})
    print(f"Wrote soft-refusal KG predictions to {out_dir}")


if __name__ == "__main__":
    main()
