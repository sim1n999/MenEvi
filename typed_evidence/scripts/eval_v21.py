"""Leakage-safe type-aware evaluator for Typed-evidence evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

REFUSAL_TEXT = "insufficient information"
DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])")
FULL_NUMBER_RE = re.compile(r"^\s*[$£€]?\s*[-+]?\d+(?:\.\d+)?\s*$")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_items(path: str | Path) -> List[Dict[str, Any]]:
    raw = load_json(path)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(an|the)\b", " ", text)
    return " ".join(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text).split())


def is_refusal(value: Any) -> bool:
    return REFUSAL_TEXT in normalize_text(value)


def extract_number(value: Any) -> Optional[str]:
    matches = NUMBER_RE.findall(str(value or ""))
    if not matches:
        return None
    try:
        number = Decimal(matches[-1])
    except InvalidOperation:
        return None
    if number == 0:
        number = Decimal(0)
    rendered = format(number.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def extract_date(value: Any) -> Optional[str]:
    match = DATE_RE.search(str(value or ""))
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}/{month:02d}/{day:02d}"


def extract_ab(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    for pattern in (r"^\s*([AB])(?:[\s.):]|$)", r"\banswer\s*(?:is|:)?\s*([AB])\b"):
        match = re.search(pattern, raw, flags=re.I)
        if match:
            return match.group(1).upper()
    return None


def infer_contract(item: Dict[str, Any]) -> str:
    question = str(item.get("question", "")).lower()
    subtype = str(item.get("question_subtype", "")).lower()
    if item.get("question_type") == "answer_refusal" or is_refusal(item.get("answer")):
        return "refusal"
    if subtype == "duration_comparison" or 'answer with "a" or "b"' in question:
        return "ab"
    if (subtype in {"counting", "arithmetic"} or "single number" in question or
            "exact amount" in question or FULL_NUMBER_RE.match(str(item.get("answer", "")))):
        return "number"
    if "yyyy/mm/dd" in question:
        return "date"
    return "phrase"


def first_line(raw: str) -> str:
    return next((line.strip() for line in raw.splitlines() if line.strip()), raw)


def parse_output(value: Any, item: Dict[str, Any]) -> str:
    raw = str(value or "").strip()
    if is_refusal(raw):
        return "Insufficient information"
    contract = infer_contract(item)
    if contract == "number":
        return extract_number(raw) or first_line(raw)
    if contract == "date":
        return extract_date(raw) or first_line(raw)
    if contract == "ab":
        return extract_ab(raw) or first_line(raw)
    return first_line(raw)


def exact_match(prediction: str, reference: str, item: Dict[str, Any]) -> bool:
    if is_refusal(reference):
        return is_refusal(prediction)
    contract = infer_contract(item)
    if contract == "number":
        return extract_number(prediction) == extract_number(reference)
    if contract == "date":
        return extract_date(prediction) == extract_date(reference)
    if contract == "ab":
        return extract_ab(prediction) == extract_ab(reference)
    pred, ref = normalize_text(prediction), normalize_text(reference)
    return pred == ref or bool(pred and ref and (pred in ref or ref in pred))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_row(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("raw_prediction", row.get("prediction", ""))
    parsed = parse_output(raw, item)
    reference = str(item.get("answer", ""))
    return {**row, "contract_v21": infer_contract(item), "prediction_v21": parsed,
            "parsed_output_v21": normalize_text(parsed),
            "sub_em_v21": int(exact_match(parsed, reference, item)),
            "f1_v21": token_f1(parsed, reference),
            "is_refusal_v21": int(is_refusal(raw) or is_refusal(parsed))}


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(group)
        correct = sum(int(row["sub_em_v21"]) for row in group)
        return {"count": count, "sub_em_count_v21": correct,
                "sub_em_v21": correct / count if count else 0.0,
                "f1_v21": sum(float(row["f1_v21"]) for row in group) / count if count else 0.0}

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("question_type") or "unknown")].append(row)
        by_subtype[str(row.get("question_subtype") or "unknown")].append(row)
    return {"overall": summary(rows),
            "by_question_type": {key: summary(value) for key, value in sorted(by_type.items())},
            "by_question_subtype": {key: summary(value) for key, value in sorted(by_subtype.items())}}


def rescore(dataset: str | Path, payload_path: str | Path) -> Dict[str, Any]:
    items = {str(item["question_id"]): item for item in load_items(dataset)}
    payload = load_json(payload_path)
    if isinstance(payload, list):
        payload = {"data": payload}
    rows = [score_row(row, items[str(row["question_id"])]) for row in payload.get("data", [])]
    return {**payload, "data": rows, "metrics_v21": compute_metrics(rows), "evaluator": "eval_v2.1"}


def self_test() -> None:
    item = {"question_subtype": "arithmetic", "question_type": "multi_session_reasoning",
            "question": "How much? Answer with the exact amount", "answer": "$105"}
    assert extract_number("$105.00") == "105"
    assert exact_match("$105.00", "$105", item)
    assert exact_match("GBP 67.500", "67.50", item)
    assert not exact_match("$105.01", "$105", item)
    print("eval_v2.1 self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.dataset, args.predictions, args.output)):
        parser.error("--dataset, --predictions, and --output are required")
    save_json(args.output, rescore(args.dataset, args.predictions))
    print(f"Wrote eval_v2.1 results to {args.output}")


if __name__ == "__main__":
    main()
