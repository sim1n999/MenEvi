"""Type-aware evaluation helpers for Answer-evidence evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REFUSAL_TEXT = "insufficient information"
DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_items(path: str | Path) -> List[Dict[str, Any]]:
    raw = load_json(path)
    return raw.get("data", raw) if isinstance(raw, dict) else raw


def result_file_from_dir(result_dir: str | Path) -> Optional[Path]:
    result_dir = Path(result_dir)
    if (result_dir / "predictions.json").is_file():
        return result_dir / "predictions.json"
    files = [p for p in result_dir.glob("*.json") if not p.name.endswith(".metrics")]
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1] if files else None


def load_prediction_payload(result_dir: str | Path) -> Dict[str, Any]:
    rf = result_file_from_dir(result_dir)
    if not rf:
        raise FileNotFoundError(f"No prediction JSON found in {result_dir}")
    payload = load_json(rf)
    if isinstance(payload, list):
        payload = {"data": payload}
    payload["source_file"] = str(rf)
    return payload


def normalize_loose(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def normalize_keep_ab(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def is_refusal_raw(text: Any) -> bool:
    return REFUSAL_TEXT in normalize_loose(text)


def extract_date(text: Any) -> Optional[str]:
    match = DATE_RE.search(str(text or ""))
    if not match:
        return None
    y, m, d = match.groups()
    return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"


def extract_number(text: Any) -> Optional[str]:
    matches = NUMBER_RE.findall(str(text or ""))
    if not matches:
        return None
    value = matches[-1]
    return value[:-2] if value.endswith(".0") else value


def extract_ab(text: Any) -> Optional[str]:
    raw = str(text or "").strip()
    patterns = [
        r"^\s*([AB])(?:[\.\):\s]|$)",
        r"\banswer\s*(?:is|:)?\s*([AB])\b",
        r"\btherefore,?\s*(?:the answer is)?\s*([AB])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def first_nonempty_line(text: Any) -> str:
    raw = str(text or "").strip()
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw


def infer_contract(item: Dict[str, Any]) -> str:
    question = str(item.get("question", "")).lower()
    subtype = str(item.get("question_subtype", "")).lower()
    answer = str(item.get("answer", "")).strip()
    if item.get("question_type") == "answer_refusal" or is_refusal_raw(answer):
        return "refusal"
    if '"a" or "b"' in question or "answer with a or b" in question or subtype == "duration_comparison":
        return "ab"
    if "single number" in question or subtype in {"counting", "arithmetic"}:
        return "number"
    if "yyyy/mm/dd" in question or "date" in question and DATE_RE.search(answer):
        return "date"
    return "phrase"


def parse_for_contract(raw_output: Any, item: Dict[str, Any]) -> str:
    contract = infer_contract(item)
    raw = str(raw_output or "").strip()
    if is_refusal_raw(raw):
        return "Insufficient information"
    if contract == "ab":
        return extract_ab(raw) or first_nonempty_line(raw)
    if contract == "number":
        return extract_number(raw) or first_nonempty_line(raw)
    if contract == "date":
        return extract_date(raw) or first_nonempty_line(raw)
    return first_nonempty_line(raw)


def sub_em_v2(prediction: str, reference: str, item: Dict[str, Any]) -> bool:
    contract = infer_contract(item)
    if is_refusal_raw(reference):
        return is_refusal_raw(prediction)
    if contract == "ab":
        return (extract_ab(prediction) or prediction.strip().upper()) == reference.strip().upper()
    if contract == "number":
        return extract_number(prediction) == extract_number(reference)
    if contract == "date":
        return extract_date(prediction) == extract_date(reference)
    pred = normalize_keep_ab(prediction)
    ref = normalize_keep_ab(reference)
    if not pred or not ref:
        return pred == ref
    return pred in ref or ref in pred


def f1_score_v2(prediction: str, reference: str) -> float:
    pred_tokens = normalize_keep_ab(prediction).split()
    ref_tokens = normalize_keep_ab(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_row_v2(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("raw_prediction", row.get("prediction", ""))
    parsed = parse_for_contract(raw, item)
    reference = str(row.get("reference_answer", item.get("answer", "")))
    return {
        **row,
        "contract_v2": infer_contract(item),
        "prediction_v2": parsed,
        "parsed_output_v2": normalize_keep_ab(parsed),
        "sub_em_v2": int(sub_em_v2(parsed, reference, item)),
        "f1_v2": f1_score_v2(parsed, reference),
        "is_refusal_v2": int(is_refusal_raw(raw) or is_refusal_raw(parsed)),
    }


def compute_metrics_v2(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        if not n:
            return {"count": 0, "sub_em_v2": 0.0, "f1_v2": 0.0, "refusal_rate_v2": 0.0, "sub_em_count_v2": 0}
        correct = sum(int(x.get("sub_em_v2", 0)) for x in subset)
        return {
            "count": n,
            "sub_em_v2": correct / n,
            "f1_v2": sum(float(x.get("f1_v2", 0.0)) for x in subset) / n,
            "refusal_rate_v2": sum(int(x.get("is_refusal_v2", 0)) for x in subset) / n,
            "sub_em_count_v2": correct,
        }

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_subtype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    answerable = []
    refusal = []
    for row in rows:
        by_type[row.get("question_type") or "unknown"].append(row)
        by_subtype[row.get("question_subtype") or "unknown"].append(row)
        if row.get("question_type") == "answer_refusal":
            refusal.append(row)
        else:
            answerable.append(row)
    return {
        "overall": summarize(rows),
        "answerable": summarize(answerable),
        "abstention": {
            "count": len(refusal),
            "accuracy_v2": (sum(int(x.get("sub_em_v2", 0)) for x in refusal) / len(refusal)) if refusal else 0.0,
        },
        "by_question_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "by_question_subtype": {k: summarize(v) for k, v in sorted(by_subtype.items())},
    }


def rescore_payload(dataset_path: str | Path, result_dir: str | Path) -> Dict[str, Any]:
    items = {item.get("question_id"): item for item in load_items(dataset_path)}
    payload = load_prediction_payload(result_dir)
    rescored = []
    for row in payload.get("data", []):
        qid = row.get("question_id")
        if qid not in items:
            raise KeyError(f"Question not found in dataset: {qid}")
        rescored.append(score_row_v2(row, items[qid]))
    payload["data"] = rescored
    payload["metrics_v2"] = compute_metrics_v2(rescored)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = rescore_payload(args.dataset, args.result_dir)
    save_json(args.output, payload)
    print(f"Wrote rescored payload to {args.output}")


if __name__ == "__main__":
    main()

