"""Run type-aware KG answering from Visual-evidence evaluation visual/OCR evidence packets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from eval_v2 import compute_metrics_v2, load_items, save_json, score_row_v2, write_csv  # noqa: arithmetic repair02


def section(title: str, lines: List[str], limit: int) -> str:
    body = "\n".join(lines[:limit]) if lines else "- none"
    return f"{title}:\n{body}"


def type_hint(question_type: str, question_subtype: str) -> str:
    if question_type == "information_extraction":
        return (
            "This is an extraction question. First inspect Visual/OCR focus evidence and OCR/attribute evidence. "
            "If a plausible visible text/object/color/position/label answer appears there, answer directly."
        )
    if question_type == "answer_refusal":
        return "This is a refusal-detection question. Refuse only if no packet section directly supports an answer."
    if question_type == "temporal_reasoning":
        return "This is a temporal question. Prefer dated state/update evidence and compare dates or durations carefully."
    if question_type == "multi_session_reasoning":
        return "This is a multi-session reasoning question. Aggregate across supporting sessions and output only the contract answer."
    if question_type == "knowledge_update":
        return "This is a knowledge-update question. Prefer the latest state/version evidence over older evidence."
    return f"Question subtype: {question_subtype or 'unknown'}."


def make_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    question_type = item.get("question_type", "unknown")
    question_subtype = item.get("question_subtype", "unknown")
    return (
        "You answer long-term memory questions from a compact evidence packet.\n"
        "Use only the evidence packet. Follow the output contract exactly.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {question_type}\n"
        f"Question subtype: {question_subtype}\n"
        f"Output contract: {packet.get('contract')}\n"
        f"Type-specific hint: {type_hint(question_type, question_subtype)}\n"
        f"Question: {item.get('question', '')}\n\n"
        f"{section('Visual/OCR focus evidence', packet.get('visual_ocr_focus_evidence', []), 32)}\n\n"
        f"{section('OCR and visual attribute evidence', packet.get('ocr_attribute_evidence', []), 18)}\n\n"
        f"{section('Top answer candidates', packet.get('top_answer_candidates', []), 24)}\n\n"
        f"{section('Temporal/update evidence', packet.get('temporal_update_evidence', []), 20)}\n\n"
        f"{section('Related KG edges', packet.get('related_edges', []), 24)}\n\n"
        f"{section('Supporting sessions', packet.get('supporting_sessions', []), 12)}\n\n"
        "Final answer:"
    )


class TextGenerator:
    def __init__(self, model_path: str, load_in_4bit: bool = True, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path_obj = Path(model_path).expanduser()
        looks_like_local_path = (
            model_path_obj.is_absolute()
            or model_path.startswith(".")
            or "/" in model_path
            or "\\" in model_path
        )
        if looks_like_local_path:
            model_path_obj = model_path_obj.resolve()
            if not model_path_obj.is_dir():
                raise FileNotFoundError(f"Local model directory not found: {model_path_obj}")
            model_path = str(model_path_obj)

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=quant,
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_new_tokens: int = 64) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_len = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, input_len:]
        output = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"output": output, "input_len": input_len, "output_len": int(new_ids.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-max-length", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    items = load_items(args.dataset)
    if args.max_samples:
        items = items[: args.max_samples]

    out_dir = Path(args.output_dir)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    policy = Path(args.policy).read_text(encoding="utf-8")

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
        packet_path = Path(args.packet_dir) / f"{qid}.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        prompt = make_prompt(item, packet, policy)
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")
        prompt_stats.append(
            {
                "question_id": qid,
                "question_type": item.get("question_type"),
                "question_subtype": item.get("question_subtype"),
                "input_len_words": len(prompt.split()),
                **packet.get("stats", {}),
            }
        )
        if args.prompt_only:
            continue

        assert generator is not None
        gen = generator.generate(prompt, args.generation_max_length)
        base_row = {
            "question_id": qid,
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "question_subtype": item.get("question_subtype"),
            "reference_answer": item.get("answer", ""),
            "raw_prediction": gen["output"],
            "input_len": gen["input_len"],
            "output_len": gen["output_len"],
        }
        results.append(score_row_v2(base_row, item))

    write_csv(out_dir / "prompt_stats.csv", prompt_stats)
    save_json(out_dir / "prompt_stats.json", prompt_stats)
    if args.prompt_only:
        avg_words = sum(row["input_len_words"] for row in prompt_stats) / len(prompt_stats) if prompt_stats else 0.0
        save_json(out_dir / "prompt_only_manifest.json", {"args": vars(args), "count": len(prompt_stats), "avg_input_len_words": avg_words})
        print(f"Wrote prompt-only outputs to {out_dir}")
        return

    payload = {
        "args": vars(args),
        "data": results,
        "metrics_v2": compute_metrics_v2(results),
        "averaged_metrics": {
            "input_len": sum(float(x.get("input_len", 0)) for x in results) / len(results) if results else 0.0,
            "output_len": sum(float(x.get("output_len", 0)) for x in results) / len(results) if results else 0.0,
        },
        "throughput": len(results) / max(time.time() - start, 1e-6),
        "prompt_stats": prompt_stats,
    }
    save_json(out_dir / "predictions.json", payload)
    save_json(out_dir / "metrics_v2.json", payload["metrics_v2"])
    save_json(out_dir / "run_config.json", vars(args))
    print(f"Wrote visual/OCR type-aware KG predictions to {out_dir}")


if __name__ == "__main__":
    main()

