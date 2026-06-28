"""Answer MemLens questions from retrieved MM-KG subgraphs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from experiment_utils import TextGenerator, finalize_run, load_items, save_json, score_prediction


ABLATION_NODE_TYPES = {
    "no_visual_node": {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"},
    "no_state_chain": {"StateVersion"},
    "no_inference_node": {"Inference"},
}
ABLATION_EDGE_TYPES = {
    "no_state_chain": {"supersedes", "updates"},
    "no_temporal_edge": {"before", "after"},
    "no_evidence_path": {"supports", "candidate_context", "visual_grounding"},
    "flat_memory": {"before", "after", "supports", "contains", "has_image", "depicts", "states", "supersedes"},
}


def apply_ablation(graph, ablation):
    if not ablation:
        return graph
    remove_types = ABLATION_NODE_TYPES.get(ablation, set())
    nodes = [n for n in graph.get("nodes", []) if n.get("type") not in remove_types]
    keep_ids = {n["id"] for n in nodes}
    remove_edges = ABLATION_EDGE_TYPES.get(ablation, set())
    edges = [
        e
        for e in graph.get("edges", [])
        if e.get("source") in keep_ids and e.get("target") in keep_ids and e.get("type") not in remove_edges
    ]
    if ablation == "flat_memory":
        edges = []
    return {**graph, "nodes": nodes, "edges": edges}


def graph_prompt(item, graph):
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    current = []
    visual = []
    states = []
    evidence = []
    for n in nodes:
        line = f"- [{n.get('type')}] {n.get('date', '')} {n.get('text', '')}"
        if n.get("type") in {"Image", "VisualFact", "VisualObject", "VisualText", "VisualAttribute"}:
            visual.append(line)
        elif n.get("type") == "StateVersion":
            states.append(line)
        else:
            current.append(line)
    for e in edges[:120]:
        evidence.append(f"- {e.get('source')} --{e.get('type')}--> {e.get('target')}")
    return (
        "You answer a long-term conversational memory question using only the provided graph memory. "
        "Prefer current valid states over outdated states. If unsupported, answer \"Insufficient information\".\n\n"
        f"Question Date:\n{item.get('question_date', 'unknown')}\n\n"
        f"Question:\n{item.get('question', '')}\n\n"
        f"Question Type:\n{item.get('question_type', '')}\n\n"
        f"Current Valid Memories:\n{chr(10).join(current[:120])}\n\n"
        f"Temporal / Update Chains:\n{chr(10).join(states[:80])}\n\n"
        f"Visual Evidence:\n{chr(10).join(visual[:80])}\n\n"
        f"Evidence Paths:\n{chr(10).join(evidence)}\n\n"
        "Instruction:\nAnswer with a short phrase only. Do not explain unless the question requires it. "
        "If unsupported, answer \"Insufficient information\"."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--subgraph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = load_items(args.input, args.max_samples)
    generator = None if args.dry_run else TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)
    subgraph_dir = Path(args.subgraph_dir)
    out_dir = Path(args.output_dir)
    prompt_dir = out_dir / "graph_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start = time.time()
    for item in tqdm(items, desc="MMKG answering"):
        qid = item.get("question_id")
        graph = __import__("json").loads((subgraph_dir / f"{qid}.json").read_text(encoding="utf-8"))
        graph = apply_ablation(graph, args.ablation)
        prompt = graph_prompt(item, graph)
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")
        if args.dry_run:
            raw = "Insufficient information"
            gen = {"input_len": len(prompt.split()), "output_len": len(raw.split())}
        else:
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
                **scored,
                "input_len": gen["input_len"],
                "output_len": gen["output_len"],
            }
        )

    finalize_run(args, results, start, out_dir)
    print(f"Wrote KG predictions to {out_dir}")


if __name__ == "__main__":
    main()
