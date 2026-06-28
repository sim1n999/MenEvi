"""Run Holdout validation typed specialists with deterministic tools."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_SCRIPTS = (
    PROJECT_ROOT
    / "answer_evidence"
    / "scripts"
)
TYPED_SCRIPTS = (
    PROJECT_ROOT
    / "typed_evidence"
    / "scripts"
)
for scripts in (ANSWER_SCRIPTS, TYPED_SCRIPTS):
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

from answer_focused_kg_answering import TextGenerator  # noqa: arithmetic repair02
from eval_v21 import compute_metrics, load_items, save_json, score_row  # noqa: arithmetic repair02


def compact(value: Any, limit: int = 620) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def transaction_prompt(packet: Dict[str, Any], policy: str) -> str:
    evidence = packet["typed_evidence"]
    rows = []
    for group in evidence.get("groups", [])[:32]:
        rows.append(
            f"{group['group_id']} | "
            f"{group.get('currency')}{group.get('total_amount')} | "
            f"status={group.get('status')} "
            f"month_match={group.get('month_match')} "
            f"target_overlap={group.get('target_overlap')} "
            f"date={group.get('transaction_date')} | "
            f"{compact(group.get('evidence'))}"
        )
    return (
        "Select completed transaction groups that match the target and time window. "
        "Do not calculate. Ignore budgets, examples, list prices, recommendations, "
        "plans, and hypothetical amounts. Return JSON only in the form "
        "{\"selected_group_ids\":[\"txn_01\",\"txn_02\"]}.\n\n"
        f"Policy:\n{policy}\n\n"
        f"Question: {packet['question']}\n"
        f"Target: {evidence.get('target')}\n"
        f"Requested month: {evidence.get('requested_month')}\n\n"
        "TransactionGroup candidates:\n"
        + ("\n".join(rows) if rows else "- none")
    )


def interval_prompt(packet: Dict[str, Any], policy: str) -> str:
    lines: List[str] = []
    durations = packet["typed_evidence"].get("durations", [])
    for duration in durations:
        index = duration["duration_index"]
        lines.append(f"Duration {index}: {duration.get('label')}")
        for interval in duration.get("intervals", [])[:16]:
            lines.append(
                f"{interval['interval_id']} | "
                f"{interval['start_date']} to {interval['end_date']} | "
                f"days={interval['duration_days']} "
                f"start_kind={interval['start_kind']} "
                f"end_kind={interval['end_kind']} | "
                f"start_evidence={compact(interval.get('start_evidence'), 420)} | "
                f"end_evidence={compact(interval.get('end_evidence'), 420)}"
            )
    return (
        "Choose one complete interval for each duration. Do not calculate and do "
        "not return A or B. Use only listed interval IDs. If either duration lacks "
        "a supported interval, return exactly Insufficient information. Otherwise "
        "return JSON only in the form "
        "{\"duration_1_interval\":\"duration_1_interval_01\","
        "\"duration_2_interval\":\"duration_2_interval_01\"}.\n\n"
        f"Policy:\n{policy}\n\n"
        f"Question: {packet['question']}\n\n"
        + ("\n".join(lines) if lines else "- no valid intervals")
    )


def visual_prompt(
    packet: Dict[str, Any],
    observations: Dict[str, Any],
    policy: str,
) -> str:
    lines = []
    for row in observations.get("data", []):
        if row.get("execution_mode") != "model":
            continue
        # Deliberately omit image_id, image_path, basename, and directory.
        lines.append(
            f"{row.get('target_id')} | rank={len(lines) + 1} | "
            f"observation="
            f"{json.dumps(row.get('observation', {}), ensure_ascii=False)}"
        )
    return (
        "Answer the memory question using only the anonymous, rank-ordered visual "
        "observations below. Earlier ranks are more relevant retrieval candidates, "
        "but visible evidence is decisive. If observations conflict and no answer "
        "is clearly supported, return exactly Insufficient information. Return a "
        "short phrase only with no explanation.\n\n"
        f"Policy:\n{policy}\n\n"
        f"Question: {packet['question']}\n\n"
        "Anonymous visual observations:\n"
        + ("\n".join(lines) if lines else "- none")
    )


def parse_json_object(output: str) -> Optional[Dict[str, Any]]:
    start, end = output.find("{"), output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def selected_transaction_total(
    output: str,
    packet: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    parsed = parse_json_object(output)
    raw_ids = parsed.get("selected_group_ids", []) if parsed else []
    if not isinstance(raw_ids, list):
        raw_ids = []
    ids = sorted(
        {
            value.lower()
            for value in raw_ids
            if isinstance(value, str)
            and re.fullmatch(r"txn_\d{2}", value.lower())
        }
    )
    groups = {
        str(group["group_id"]): group
        for group in packet["typed_evidence"].get("groups", [])
    }
    trace: Dict[str, Any] = {
        "selector_output": output,
        "selected_group_ids": ids,
    }
    if parsed is None:
        trace["error"] = "invalid JSON"
        return None, trace
    if not ids:
        trace["error"] = "no selected transaction group"
        return None, trace
    unknown = sorted(set(ids) - set(groups))
    if unknown:
        trace["error"] = "unknown transaction group"
        trace["unknown_ids"] = unknown
        return None, trace

    selected = [groups[group_id] for group_id in ids]
    if any(
        group.get("status") != "completed"
        or group.get("month_match") is False
        for group in selected
    ):
        trace["error"] = "ineligible transaction group"
        return None, trace
    currencies = {str(group.get("currency")) for group in selected}
    if len(currencies) != 1:
        trace["error"] = "mixed currencies"
        return None, trace

    total = sum(
        (Decimal(str(group["total_amount"])) for group in selected),
        Decimal("0"),
    )
    answer = f"{next(iter(currencies))}{total:.2f}"
    trace["calculator_output"] = answer
    trace["selected_groups"] = selected
    return answer, trace


def selected_interval_answer(
    output: str,
    packet: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    parsed = parse_json_object(output)
    trace: Dict[str, Any] = {"selector_output": output}
    if parsed is None:
        trace["error"] = "invalid JSON"
        return None, trace

    keys = ("duration_1_interval", "duration_2_interval")
    selection = {key: parsed.get(key) for key in keys}
    trace["selection"] = selection
    intervals: Dict[str, Dict[str, Any]] = {}
    for duration in packet["typed_evidence"].get("durations", []):
        for interval in duration.get("intervals", []):
            intervals[str(interval["interval_id"])] = interval

    if any(selection[key] not in intervals for key in keys):
        trace["error"] = "unknown or missing interval id"
        return None, trace

    one = intervals[str(selection["duration_1_interval"])]
    two = intervals[str(selection["duration_2_interval"])]
    if int(one.get("duration_index", -1)) != 1:
        trace["error"] = "cross-duration interval for duration 1"
        return None, trace
    if int(two.get("duration_index", -1)) != 2:
        trace["error"] = "cross-duration interval for duration 2"
        return None, trace

    days_one = int(one["duration_days"])
    days_two = int(two["duration_days"])
    trace["duration_1_days"] = days_one
    trace["duration_2_days"] = days_two
    trace["selected_intervals"] = [one, two]
    if days_one == days_two:
        trace["error"] = "tied duration"
        return None, trace

    answer = "A" if days_one > days_two else "B"
    trace["calculator_output"] = answer
    return answer, trace


def load_observations(directory: Optional[str], qid: str) -> Dict[str, Any]:
    if not directory:
        return {"data": []}
    path = Path(directory) / f"{qid}.json"
    if not path.is_file():
        return {"data": []}
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(
    packet: Dict[str, Any],
    observations: Dict[str, Any],
    policy: str,
) -> str:
    subtype = str(packet["question_subtype"])
    if subtype == "arithmetic":
        return transaction_prompt(packet, policy)
    if subtype == "duration_comparison":
        return interval_prompt(packet, policy)
    return visual_prompt(packet, observations, policy)


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

    items = {
        str(item["question_id"]): item
        for item in load_items(args.dataset)
    }
    packet_paths = sorted(Path(args.packet_dir).glob("*.json"))
    if args.max_samples:
        packet_paths = packet_paths[: args.max_samples]

    policy = Path(args.policy).read_text(encoding="utf-8")
    output_dir = Path(args.output_dir)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    generator = (
        None
        if args.prompt_only
        else TextGenerator(
            args.model,
            load_in_4bit=not args.no_4bit,
            dtype=args.dtype,
        )
    )
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    started = time.time()

    for packet_path in packet_paths:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        qid = str(packet["question_id"])
        subtype = str(packet["question_subtype"])
        observations = load_observations(args.visual_observation_dir, qid)
        prompt = build_prompt(packet, observations, policy)
        (prompt_dir / f"{qid}.txt").write_text(prompt, encoding="utf-8")

        if args.prompt_only:
            traces.append(
                {
                    "question_id": qid,
                    "question_subtype": subtype,
                    "prompt_words": len(prompt.split()),
                    "execution_mode": "prompt_only",
                }
            )
            continue

        assert generator is not None
        generated = generator.generate(prompt, args.generation_max_length)
        selector_output = str(generated["output"])

        if subtype == "arithmetic":
            answer, tool_trace = selected_transaction_total(
                selector_output,
                packet,
            )
            mode = "transaction_selector_plus_decimal"
        elif subtype == "duration_comparison":
            answer, tool_trace = selected_interval_answer(
                selector_output,
                packet,
            )
            mode = "interval_selector_plus_date_math"
        else:
            answer = selector_output.strip()
            tool_trace = {
                "anonymous_visual_observation_count": sum(
                    row.get("execution_mode") == "model"
                    for row in observations.get("data", [])
                )
            }
            mode = "anonymous_question_conditioned_visual_answer"

        eligible = bool(
            answer
            and answer.strip()
            and answer.strip().lower() != "insufficient information"
            and not tool_trace.get("error")
        )
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
        traces.append(
            {
                "question_id": qid,
                "question_subtype": subtype,
                "eligible_for_merge": eligible,
                "execution_mode": mode,
                "tool_trace": tool_trace,
            }
        )

    save_json(
        output_dir / "run_manifest.json",
        {
            "args": vars(args),
            "packet_count": len(packet_paths),
            "prediction_count": len(rows),
            "elapsed_seconds": time.time() - started,
            "label_leakage": False,
            "model_visible_image_identifiers": "anonymous_target_ids_only",
        },
    )
    save_json(output_dir / "traces.json", traces)
    if rows:
        save_json(
            output_dir / "predictions.json",
            {
                "args": vars(args),
                "data": rows,
                "metrics_v21": compute_metrics(rows),
            },
        )
    print(
        f"Processed {len(packet_paths)} H packets; "
        f"produced {len(rows)} predictions"
    )


if __name__ == "__main__":
    main()

