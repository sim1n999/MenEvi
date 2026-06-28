"""Local helpers for KG retrieval baseline.

This file intentionally avoids importing code from memlens_repro/scripts so the
new experiment folder contains all custom runner/evaluation code used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REFUSAL_TEXT = "insufficient information"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_items(path: str | Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    raw = load_json(path)
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    return items[:max_samples] if max_samples else items


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
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


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def parse_model_output(raw_output: str) -> str:
    text = str(raw_output or "").strip()
    if not text:
        return ""
    # Keep the first non-empty line; most runs request a short phrase only.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def is_refusal(text: Any) -> bool:
    return REFUSAL_TEXT in normalize_answer(text)


def sub_em(prediction: str, reference: str) -> bool:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return pred == ref
    if is_refusal(reference):
        return is_refusal(prediction)
    return pred in ref or ref in pred


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
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


def score_prediction(raw_output: str, reference: str) -> Dict[str, Any]:
    prediction = parse_model_output(raw_output)
    return {
        "prediction": prediction,
        "parsed_output": normalize_answer(prediction),
        "sub_em": int(sub_em(prediction, reference)),
        "f1": f1_score(prediction, reference),
        "is_refusal": int(is_refusal(prediction)),
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        if not n:
            return {"count": 0, "sub_em": 0.0, "f1": 0.0, "refusal_rate": 0.0, "sub_em_count": 0}
        sub_count = sum(int(x.get("sub_em", 0)) for x in rows)
        return {
            "count": n,
            "sub_em": sub_count / n,
            "f1": sum(float(x.get("f1", 0.0)) for x in rows) / n,
            "refusal_rate": sum(int(x.get("is_refusal", is_refusal(x.get("prediction", "")))) for x in rows) / n,
            "sub_em_count": sub_count,
        }

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    answerable: List[Dict[str, Any]] = []
    refusal_items: List[Dict[str, Any]] = []
    for row in results:
        qtype = row.get("question_type") or "unknown"
        by_type[qtype].append(row)
        if qtype == "answer_refusal":
            refusal_items.append(row)
        else:
            answerable.append(row)

    return {
        "overall": summarize(results),
        "by_question_type": {qtype: summarize(rows) for qtype, rows in sorted(by_type.items())},
        "answerable": summarize(answerable),
        "abstention": {
            "count": len(refusal_items),
            "accuracy": (
                sum(int(x.get("sub_em", 0)) for x in refusal_items) / len(refusal_items)
                if refusal_items
                else 0.0
            ),
        },
    }


def finalize_run(
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    start_time: float,
    output_dir: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = compute_metrics(results)
    averaged = {
        "input_len": sum(float(x.get("input_len", 0)) for x in results) / len(results) if results else 0.0,
        "output_len": sum(float(x.get("output_len", 0)) for x in results) / len(results) if results else 0.0,
    }
    payload: Dict[str, Any] = {
        "args": vars(args),
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged,
        "throughput": len(results) / max(time.time() - start_time, 1e-6),
    }
    if extra:
        payload.update(extra)
    output_dir = Path(output_dir)
    save_json(output_dir / "predictions.json", payload)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "run_config.json", vars(args))
    return payload


def result_file_from_dir(result_dir: str | Path) -> Optional[Path]:
    result_dir = Path(result_dir)
    if (result_dir / "predictions.json").is_file():
        return result_dir / "predictions.json"
    files = [p for p in result_dir.glob("*.json") if not p.name.endswith(".metrics")]
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1] if files else None


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
                raise FileNotFoundError(
                    f"Local model directory not found: {model_path_obj}. "
                    "Set MODEL_PATH to a local text model directory."
                )
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

    def generate(self, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
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


def load_prediction_payload(result_dir: str | Path) -> Dict[str, Any]:
    rf = result_file_from_dir(result_dir)
    if not rf:
        raise FileNotFoundError(f"No prediction JSON found in {result_dir}")
    payload = load_json(rf)
    if isinstance(payload, list):
        return {"data": payload, "metrics": compute_metrics(payload), "source_file": str(rf)}
    payload["source_file"] = str(rf)
    return payload
