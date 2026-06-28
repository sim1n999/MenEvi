"""Run typed G specialists with deterministic arithmetic and duration tools."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = PROJECT_ROOT / "answer_evidence" / "scripts"
if str(ANSWER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ANSWER_SCRIPTS))

from answer_focused_kg_answering import TextGenerator  # noqa: arithmetic repair02
from eval_v21 import compute_metrics, load_items, save_json, score_row  # noqa: arithmetic repair02


def compact(value: Any, limit: int = 620) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def purchase_prompt(packet: Dict[str, Any], policy: str) -> str:
    evidence = packet["typed_evidence"]
    rows = []
    for event in evidence.get("events", [])[:24]:
        rows.append(
            f"{event['event_id']} | {event['currency']}{event['amount']} | "
            f"status={event['status']} month_match={event['month_match']} "
            f"date={event.get('event_date')} | {compact(event.get('evidence'))}"
        )
    return (
        "Select completed purchases that match the target. Do not calculate. "
        "Ignore budgets, examples, list prices, recommendations, and hypothetical amounts. "
        "Return JSON only: {\"selected_ids\":[\"purchase_01\"]}.\n\n"
        f"Policy:\n{policy}\n\nQuestion: {packet['question']}\n"
        f"Target: {evidence['target']}\nRequested month: {evidence.get('requested_month')}\n\n"
        "PurchaseEvent candidates:\n" + "\n".join(rows)
    )


def duration_prompt(packet: Dict[str, Any], policy: str) -> str:
    lines = []
    for duration in packet["typed_evidence"].get("durations", []):
        lines.append(f"Duration {duration['duration_index']}: {duration['label']}")
        for boundary in duration.get("boundaries", [])[:18]:
            lines.append(
                f"{boundary['boundary_id']} | kind={boundary['boundary_kind']} | "
                f"explicit_dates={boundary['explicit_dates']} observation_date={boundary['observation_date']} | "
                f"{compact(boundary['evidence'])}"
            )
    return (
        "Choose one genuine start and one genuine end boundary for each duration. "
        "Observation dates are acceptable only when the text describes the activity at that time. "
        "Return JSON only with keys duration_1_start, duration_1_end, duration_2_start, "
        "duration_2_end, each containing one boundary_id. Do not return A or B.\n\n"
        f"Policy:\n{policy}\n\nQuestion: {packet['question']}\n\n" + "\n".join(lines)
    )


def visual_prompt(packet: Dict[str, Any], observations: Dict[str, Any], policy: str) -> str:
    lines = []
    for row in observations.get("data", []):
        if row.get("execution_mode") != "model":
            continue
        lines.append(
            f"{row.get('target_id')} image={row.get('image_id')} "
            f"observation={json.dumps(row.get('observation', {}), ensure_ascii=False)}"
        )
    return (
        "Answer the memory question using only relevant question-conditioned visual observations. "
        "Return a short phrase only, with no explanation. If none of the observations visibly "
        "supports an answer, return exactly Insufficient information.\n\n"
        f"Policy:\n{policy}\n\nQuestion: {packet['question']}\n\n"
        "Visual observations:\n" + ("\n".join(lines) if lines else "- none")
    )


def selected_purchase_total(output: str, packet: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    ids = sorted(set(re.findall(r"purchase_\d{2}", output.lower())))
    events = {event["event_id"]: event for event in packet["typed_evidence"].get("events", [])}
    selected = [events[event_id] for event_id in ids if event_id in events]
    trace = {"selector_output": output, "selected_ids": ids}
    if not selected:
        return None, trace
    unknown = sorted(set(ids) - set(events))
    if unknown:
        trace["error"] = "unknown purchase id"
        trace["unknown_ids"] = unknown
        return None, trace
    if any(event.get("status") != "completed" or event.get("month_match") is False for event in selected):
        trace["error"] = "ineligible purchase event"
        return None, trace
    currencies = {event["currency"] for event in selected}
    if len(currencies) != 1:
        trace["error"] = "mixed currencies"
        return None, trace
    total = sum((Decimal(event["amount"]) for event in selected), Decimal("0"))
    answer = f"{next(iter(currencies))}{total:.2f}"
    trace["calculator_output"] = answer
    trace["selected_events"] = selected
    return answer, trace


def boundary_date(boundary: Dict[str, Any], role: str) -> Optional[datetime]:
    explicit = list(boundary.get("explicit_dates", []))
    value = None
    if explicit:
        value = explicit[0] if role == "start" else explicit[-1]
    if not value:
        value = boundary.get("observation_date")
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", str(value or ""))
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def selected_duration_answer(output: str, packet: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    try:
        start, end = output.find("{"), output.rfind("}")
        selection = json.loads(output[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None, {"selector_output": output, "error": "invalid JSON"}
    boundaries = {}
    for duration in packet["typed_evidence"].get("durations", []):
        for boundary in duration.get("boundaries", []):
            boundaries[boundary["boundary_id"]] = boundary
    keys = ("duration_1_start", "duration_1_end", "duration_2_start", "duration_2_end")
    if any(selection.get(key) not in boundaries for key in keys):
        return None, {"selector_output": output, "selection": selection, "error": "unknown boundary id"}
    for key in keys:
        boundary = boundaries[selection[key]]
        expected_duration = int(key.split("_")[1])
        role = "start" if key.endswith("start") else "end"
        if int(boundary.get("duration_index", -1)) != expected_duration:
            return None, {
                "selector_output": output, "selection": selection,
                "error": f"cross-duration boundary for {key}",
            }
        if boundary.get("boundary_kind") not in {role, "range", "observation"}:
            return None, {
                "selector_output": output, "selection": selection,
                "error": f"incompatible boundary role for {key}",
            }
    dates = {
        key: boundary_date(boundaries[selection[key]], "start" if key.endswith("start") else "end")
        for key in keys
    }
    if any(value is None for value in dates.values()):
        return None, {"selector_output": output, "selection": selection, "error": "unparseable date"}
    days_one = (dates["duration_1_end"] - dates["duration_1_start"]).days
    days_two = (dates["duration_2_end"] - dates["duration_2_start"]).days
    trace = {
        "selector_output": output,
        "selection": selection,
        "duration_1_days": days_one,
        "duration_2_days": days_two,
    }
    if days_one < 0 or days_two < 0 or days_one == days_two:
        trace["error"] = "invalid or tied duration"
        return None, trace
    answer = "A" if days_one > days_two else "B"
    trace["calculator_output"] = answer
    return answer, trace


def load_observations(directory: Optional[str], qid: str) -> Dict[str, Any]:
    if not directory:
        return {"data": []}
    path = Path(directory) / f"{qid}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"data": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-observation-dir")
    parser.add_argument("--model")
    parser.add_argument("--generation-max-length", type=int, default=160)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if not args.prompt_only and not args.model:
        parser.error("--model is required unless --prompt-only is set")

    items = {str(item["question_id"]): item for item in load_items(args.dataset)}
    packet_paths = sorted(Path(args.packet_dir).glob("*.json"))
    if args.max_samples:
        packet_paths = packet_paths[:args.max_samples]
    policy = Path(args.policy).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    generator = None if args.prompt_only else TextGenerator(
        args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype
    )
    rows, traces = [], []
    started = time.time()

    for packet_path in packet_paths:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        qid, subtype = str(packet["question_id"]), str(packet["question_subtype"])
        observations = load_observations(args.visual_observation_dir, qid)
        if subtype == "arithmetic":
            prompt = purchase_prompt(packet, policy)
        elif subtype == "duration_comparison":
            prompt = duration_prompt(packet, policy)
        else:
            prompt = visual_prompt(packet, observations, policy)
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")
        if args.prompt_only:
            traces.append({"question_id": qid, "question_subtype": subtype,
                           "prompt_words": len(prompt.split()), "execution_mode": "prompt_only"})
            continue

        assert generator is not None
        generated = generator.generate(prompt, args.generation_max_length)
        selector_output = generated["output"]
        if subtype == "arithmetic":
            answer, tool_trace = selected_purchase_total(selector_output, packet)
            mode = "event_selector_plus_decimal"
        elif subtype == "duration_comparison":
            answer, tool_trace = selected_duration_answer(selector_output, packet)
            mode = "boundary_selector_plus_date_math"
        else:
            answer, tool_trace = selector_output.strip(), {
                "visual_observation_count": sum(
                    row.get("execution_mode") == "model" for row in observations.get("data", [])
                )
            }
            mode = "question_conditioned_visual_answer"
        eligible = bool(answer and answer.strip() and answer.strip().lower() != "insufficient information")
        raw = answer or ""
        base = {
            "question_id": qid,
            "question": packet["question"],
            "question_type": packet["question_type"],
            "question_subtype": subtype,
            "reference_answer": items[qid].get("answer", ""),
            "raw_prediction": raw,
            "input_len": generated.get("input_len", 0),
            "output_len": generated.get("output_len", 0),
            "prompt_len_words": len(prompt.split()),
            "execution_mode": mode,
            "eligible_for_merge": eligible,
            "tool_trace": tool_trace,
        }
        rows.append(score_row(base, items[qid]))
        traces.append({"question_id": qid, "question_subtype": subtype,
                       "eligible_for_merge": eligible, "execution_mode": mode,
                       "tool_trace": tool_trace})

    save_json(output_dir / "run_manifest.json", {
        "args": vars(args), "packet_count": len(packet_paths), "prediction_count": len(rows),
        "elapsed_seconds": time.time() - started, "label_leakage": False,
    })
    save_json(output_dir / "traces.json", traces)
    if rows:
        save_json(output_dir / "predictions.json", {
            "args": vars(args), "data": rows, "metrics_v21": compute_metrics(rows)
        })
    print(f"Processed {len(packet_paths)} packets; produced {len(rows)} predictions")


if __name__ == "__main__":
    main()

