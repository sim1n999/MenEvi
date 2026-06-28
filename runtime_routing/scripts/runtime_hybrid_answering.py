"""Run Runtime-routing evaluation with real per-question packet and prompt routing."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
sys.path.insert(0, str(ANSWER_SCRIPTS))

from answer_focused_kg_answering import TextGenerator  # noqa: arithmetic repair02
from eval_v2 import compute_metrics_v2, load_items, save_json, score_row_v2, write_csv  # noqa: arithmetic repair02


USE_D_SUBTYPES = {
    "entity",
    "previnfo",
    "knowledge_update",
    "temporal_info_extraction",
}
SPECIALIST_SUBTYPES = {"duration_comparison", "arithmetic"}


def section(title: str, lines: List[str], limit: int) -> str:
    body = "\n".join(str(line) for line in lines[:limit]) if lines else "- none"
    return f"{title}:\n{body}"


def choose_route(subtype: str, variant: str) -> Tuple[str, str]:
    if variant in {"runtime_specialists", "runtime_specialists_override"} and subtype in SPECIALIST_SUBTYPES:
        return f"{subtype}_specialist", "E specialist selected for this subtype"
    if subtype in USE_D_SUBTYPES:
        return "D", "D packet selected by the subtype routing policy"
    return "C", "C packet selected by the subtype routing policy"


def make_c_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    return (
        "You answer long-term memory questions from a compact evidence packet.\n"
        "Use only the evidence packet. Follow the output contract exactly.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {item.get('question_type', 'unknown')}\n"
        f"Question subtype: {item.get('question_subtype', 'unknown')}\n"
        f"Output contract: {packet.get('contract')}\n"
        f"Question: {item.get('question', '')}\n\n"
        f"{section('Top answer candidates', packet.get('top_candidates', []), 24)}\n\n"
        f"{section('Visual evidence', packet.get('visual_evidence', []), 24)}\n\n"
        f"{section('Temporal/update evidence', packet.get('temporal_update_evidence', []), 32)}\n\n"
        f"{section('Supporting sessions', packet.get('supporting_sessions', []), 12)}\n\n"
        "Final answer:"
    )


def make_d_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    return (
        "You answer long-term memory questions from a type-aware evidence packet.\n"
        "Use only the evidence packet. Follow the output contract exactly.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question type: {item.get('question_type', 'unknown')}\n"
        f"Question subtype: {item.get('question_subtype', 'unknown')}\n"
        f"Output contract: {packet.get('contract')}\n"
        f"Question: {item.get('question', '')}\n\n"
        f"{section('Visual/OCR focus evidence', packet.get('visual_ocr_focus_evidence', []), 32)}\n\n"
        f"{section('OCR and visual attribute evidence', packet.get('ocr_attribute_evidence', []), 18)}\n\n"
        f"{section('Top answer candidates', packet.get('top_answer_candidates', []), 24)}\n\n"
        f"{section('Temporal/update evidence', packet.get('temporal_update_evidence', []), 20)}\n\n"
        f"{section('Related KG edges', packet.get('related_edges', []), 24)}\n\n"
        f"{section('Supporting sessions', packet.get('supporting_sessions', []), 12)}\n\n"
        "Final answer:"
    )


def make_duration_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    calculation = packet.get("calculation", {})
    return (
        "You compare two durations from a specialist evidence packet.\n"
        "Use only the packet. Return exactly A or B.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question: {item.get('question', '')}\n"
        f"Duration 1 label: {packet.get('duration_1_label')}\n"
        f"Duration 2 label: {packet.get('duration_2_label')}\n\n"
        f"{section('Duration 1 dated evidence', packet.get('duration_1_evidence', []), 16)}\n\n"
        f"{section('Duration 2 dated evidence', packet.get('duration_2_evidence', []), 16)}\n\n"
        f"Deterministic boundary calculation: {json.dumps(calculation, ensure_ascii=False)}\n\n"
        "Final answer:"
    )


def make_arithmetic_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    candidates = [
        f"{row.get('candidate_id')} {row.get('currency')}{row.get('amount')} score={row.get('score')} "
        f"session={row.get('session_id')} date={row.get('date')} evidence={row.get('evidence')}"
        for row in packet.get("amount_candidates", [])
    ]
    return (
        "You solve a spending-total question from a specialist evidence packet.\n"
        "Use only actual purchases relevant to the target. Ignore budgets, examples, retail prices, and hypothetical amounts.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n"
        f"Target: {packet.get('target')}\n"
        f"Output contract: {packet.get('contract')}\n\n"
        f"{section('Currency amount candidates', candidates, 24)}\n\n"
        f"{section('High-confidence purchase evidence', packet.get('selected_purchase_evidence', []), 16)}\n\n"
        f"Deterministic candidate calculation: {json.dumps(packet.get('calculation', {}), ensure_ascii=False)}\n\n"
        "Return the exact total only. Final answer:"
    )


def make_arithmetic_selection_prompt(item: Dict[str, Any], packet: Dict[str, Any], policy: str) -> str:
    candidates = [
        f"{row.get('candidate_id')} | {row.get('currency')}{row.get('amount')} | "
        f"session={row.get('session_id')} | date={row.get('date')} | {row.get('evidence')}"
        for row in packet.get("amount_candidates", [])
    ]
    return (
        "Select completed purchases relevant to the spending target.\n"
        "Do not calculate. Return JSON only: {\"selected_ids\":[\"a01\",\"a02\"]}.\n\n"
        f"Policy:\n{policy.strip()}\n\n"
        f"Question date: {item.get('question_date', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n"
        f"Target: {packet.get('target')}\n\n"
        f"{section('Candidate purchases', candidates, 24)}\n"
    )


def calculate_selected_amounts(selection: str, packet: Dict[str, Any]) -> Tuple[str, List[str]]:
    selected_ids = sorted(set(re.findall(r"\ba\d{2}\b", selection.lower())))
    by_id = {str(row.get("candidate_id")): row for row in packet.get("amount_candidates", [])}
    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    if not selected:
        return "", []
    currencies = {str(row.get("currency")) for row in selected}
    if len(currencies) != 1:
        return "", selected_ids
    total = sum((Decimal(str(row.get("amount"))) for row in selected), Decimal("0"))
    return f"{next(iter(currencies))}{total:.2f}", selected_ids


def make_prompt(route: str, item: Dict[str, Any], packet: Dict[str, Any], policies: Dict[str, str]) -> str:
    if route == "C":
        return make_c_prompt(item, packet, policies["C"])
    if route == "D":
        return make_d_prompt(item, packet, policies["D"])
    if route == "duration_comparison_specialist":
        return make_duration_prompt(item, packet, policies["duration"])
    if route == "arithmetic_specialist":
        return make_arithmetic_prompt(item, packet, policies["arithmetic"])
    raise ValueError(f"Unknown route: {route}")


def packet_path_for(route: str, qid: str, args: argparse.Namespace) -> Path:
    if route == "C":
        return Path(args.c_packet_dir) / f"{qid}.json"
    if route == "D":
        return Path(args.d_packet_dir) / f"{qid}.json"
    if not args.specialist_packet_dir:
        raise ValueError("--specialist-packet-dir is required for specialist variants")
    return Path(args.specialist_packet_dir) / f"{qid}.json"


def programmatic_override(packet: Dict[str, Any], variant: str) -> Tuple[bool, str]:
    if variant != "runtime_specialists_override":
        return False, ""
    calculation = packet.get("calculation", {})
    if calculation.get("confidence") != "high" or not calculation.get("programmatic_answer"):
        return False, ""
    return True, str(calculation["programmatic_answer"])


def route_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("route_source"))].append(row)
    output = []
    for route, group in sorted(grouped.items()):
        output.append(
            {
                "route_source": route,
                "count": len(group),
                "programmatic_count": sum(int(row.get("execution_mode") == "programmatic_override") for row in group),
                "calculator_count": sum(int(row.get("execution_mode") == "model_selector_plus_calculator") for row in group),
                "calculator_fallback_count": sum(int(row.get("execution_mode") == "model_selector_fallback") for row in group),
                "avg_prompt_words": sum(float(row.get("prompt_len_words", 0)) for row in group) / len(group),
                "sub_em_v2": sum(int(row.get("sub_em_v2", 0)) for row in group) / len(group) if group and "sub_em_v2" in group[0] else None,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--c-packet-dir", required=True)
    parser.add_argument("--d-packet-dir", required=True)
    parser.add_argument("--specialist-packet-dir", default=None)
    parser.add_argument("--policy-c", required=True)
    parser.add_argument("--policy-d", required=True)
    parser.add_argument("--policy-duration", required=True)
    parser.add_argument("--policy-arithmetic", required=True)
    parser.add_argument("--variant", choices=["runtime_cd", "runtime_specialists", "runtime_specialists_override"], required=True)
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
    policies = {
        "C": Path(args.policy_c).read_text(encoding="utf-8"),
        "D": Path(args.policy_d).read_text(encoding="utf-8"),
        "duration": Path(args.policy_duration).read_text(encoding="utf-8"),
        "arithmetic": Path(args.policy_arithmetic).read_text(encoding="utf-8"),
    }

    generator = None
    if not args.prompt_only:
        if not args.model:
            raise ValueError("--model is required unless --prompt-only is set")
        generator = TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)

    out_dir = Path(args.output_dir)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    results = []
    start = time.time()

    for item in items:
        qid = str(item.get("question_id"))
        subtype = str(item.get("question_subtype") or "unknown")
        route, reason = choose_route(subtype, args.variant)
        packet_path = packet_path_for(route, qid, args)
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        prompt = make_prompt(route, item, packet, policies)
        if args.variant == "runtime_specialists_override" and route == "arithmetic_specialist":
            prompt = make_arithmetic_selection_prompt(item, packet, policies["arithmetic"])
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")

        use_override, override_answer = programmatic_override(packet, args.variant)
        trace = {
            "question_id": qid,
            "question_type": item.get("question_type"),
            "question_subtype": subtype,
            "route_source": route,
            "route_reason": reason,
            "packet_strategy": packet.get("strategy"),
            "packet_path": str(packet_path),
            "prompt_len_words": len(prompt.split()),
            "programmatic_eligible": bool(packet.get("calculation", {}).get("confidence") == "high"),
            "programmatic_answer": packet.get("calculation", {}).get("programmatic_answer"),
        }
        traces.append(trace)
        if args.prompt_only:
            continue

        tool_trace = None
        if args.variant == "runtime_specialists_override" and route == "arithmetic_specialist":
            assert generator is not None
            selection = generator.generate(prompt, args.generation_max_length)
            calculated, selected_ids = calculate_selected_amounts(selection["output"], packet)
            tool_trace = {
                "selector_output": selection["output"],
                "selected_ids": selected_ids,
                "calculator_output": calculated or None,
            }
            if calculated:
                raw = calculated
                input_len = selection["input_len"]
                output_len = selection["output_len"] + len(raw.split())
                execution_mode = "model_selector_plus_calculator"
            else:
                fallback_prompt = make_arithmetic_prompt(item, packet, policies["arithmetic"])
                fallback = generator.generate(fallback_prompt, args.generation_max_length)
                raw = fallback["output"]
                input_len = selection["input_len"] + fallback["input_len"]
                output_len = selection["output_len"] + fallback["output_len"]
                execution_mode = "model_selector_fallback"
        elif use_override:
            raw = override_answer
            input_len = 0
            output_len = len(raw.split())
            execution_mode = "programmatic_override"
        else:
            assert generator is not None
            generated = generator.generate(prompt, args.generation_max_length)
            raw = generated["output"]
            input_len = generated["input_len"]
            output_len = generated["output_len"]
            execution_mode = "model"

        base_row = {
            "question_id": qid,
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "question_subtype": subtype,
            "reference_answer": item.get("answer", ""),
            "raw_prediction": raw,
            "input_len": input_len,
            "output_len": output_len,
            "prompt_len_words": len(prompt.split()),
            "route_source": route,
            "route_reason": reason,
            "route_policy": args.variant,
            "execution_mode": execution_mode,
            "programmatic_confidence": packet.get("calculation", {}).get("confidence"),
            "tool_trace": tool_trace,
        }
        results.append(score_row_v2(base_row, item))

    write_csv(out_dir / "route_trace.csv", traces)
    save_json(out_dir / "route_trace.json", traces)
    if args.prompt_only:
        write_csv(out_dir / "route_summary.csv", route_summary(traces))
        save_json(
            out_dir / "prompt_only_manifest.json",
            {"args": vars(args), "count": len(traces), "variant": args.variant},
        )
        print(f"Wrote prompt-only runtime routing outputs to {out_dir}")
        return

    payload = {
        "args": vars(args),
        "data": results,
        "metrics_v2": compute_metrics_v2(results),
        "averaged_metrics": {
            "input_len": sum(float(row.get("input_len", 0)) for row in results) / len(results) if results else 0.0,
            "output_len": sum(float(row.get("output_len", 0)) for row in results) / len(results) if results else 0.0,
        },
        "throughput": len(results) / max(time.time() - start, 1e-6),
        "source_file": str(out_dir / "predictions.json"),
    }
    save_json(out_dir / "predictions.json", payload)
    save_json(out_dir / "metrics_v2.json", payload["metrics_v2"])
    save_json(out_dir / "run_config.json", vars(args))
    write_csv(out_dir / "route_summary.csv", route_summary(results))
    print(f"Wrote runtime hybrid predictions to {out_dir}")


if __name__ == "__main__":
    main()

