"""Shared helpers for MemLens experiment runners.

The runners in this folder intentionally keep dependencies light. They reuse the
official MEMLENS parsing/metric code where possible and provide a small text
generation wrapper for local HuggingFace causal LMs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPRO_ROOT = Path(__file__).resolve().parents[1]
MEMLENS_DIR = REPRO_ROOT / "MEMLENS"
if str(MEMLENS_DIR) not in sys.path:
    sys.path.insert(0, str(MEMLENS_DIR))

from parse_utils import compute_metrics, f1_score, normalize_answer, parse_model_output, sub_em  # noqa: arithmetic repair02
from utils import resolve_image_path  # noqa: arithmetic repair02


def load_items(path: str | Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    if max_samples:
        items = items[:max_samples]
    return items


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
    rows = []
    if not path or not Path(path).is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_sessions(item: Dict[str, Any]) -> Iterable[Tuple[int, str, str, List[Dict[str, Any]]]]:
    sessions = item.get("haystack_sessions", [])
    ids = item.get("haystack_session_ids", [])
    dates = item.get("haystack_dates", [])
    for i, session in enumerate(sessions):
        if isinstance(session, dict):
            sid = session.get("session_id") or (ids[i] if i < len(ids) else f"session_{i+1}")
            date = session.get("date") or (dates[i] if i < len(dates) else "unknown")
            turns = session.get("session", [])
        else:
            sid = ids[i] if i < len(ids) else f"session_{i+1}"
            date = dates[i] if i < len(dates) else "unknown"
            turns = session
        yield i, sid, date, turns


def image_key(img_info: Any) -> str:
    if isinstance(img_info, dict):
        return str(
            img_info.get("file")
            or img_info.get("path")
            or img_info.get("file_path")
            or img_info.get("img_file")
            or img_info.get("image_url")
            or img_info
        )
    return str(img_info)


def turn_text(turn: Dict[str, Any], include_image_placeholders: bool = False) -> str:
    role = turn.get("role", "")
    text = str(turn.get("content", "") or "")
    if not include_image_placeholders:
        text = text.replace("<image>", "").strip()
    return f"{role}: {text}".strip()


def load_caption_cache(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path:
        return cache
    for row in read_jsonl(path):
        for key in (row.get("image_id"), row.get("image_path"), row.get("file")):
            if key:
                cache[str(key)] = row
                cache[Path(str(key)).name] = row
    return cache


def caption_for_image(
    img_info: Any,
    source: str = "dataset",
    caption_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    if source == "dataset" and isinstance(img_info, dict):
        return str(img_info.get("blip_caption") or "").strip()

    key = image_key(img_info)
    caption_cache = caption_cache or {}
    row = caption_cache.get(key) or caption_cache.get(Path(key).name)
    if not row:
        return ""

    parts: List[str] = []
    for field in ("short_caption", "visible_text", "visible_objects", "attributes", "possible_memory_facts"):
        val = row.get(field)
        if isinstance(val, list):
            val = "; ".join(str(x) for x in val if x)
        if val:
            parts.append(f"{field}: {val}")
    return " | ".join(parts)


def session_to_text(
    sid: str,
    date: str,
    turns: List[Dict[str, Any]],
    caption_source: Optional[str] = None,
    caption_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    parts = [f"Session ID: {sid}", f"Date: {date}"]
    for turn in turns:
        txt = turn_text(turn)
        if txt:
            parts.append(txt)
        if caption_source:
            for img in turn.get("images", []) or []:
                cap = caption_for_image(img, caption_source, caption_cache)
                if cap:
                    parts.append(f"Image memory ({image_key(img)}): {cap}")
    return "\n".join(parts)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, docs: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d.get("text", "")) for d in self.docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(toks) for toks in self.doc_tokens]
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        q_terms = tokenize(query)
        scores: List[Tuple[float, Dict[str, Any]]] = []
        for doc, tf, dl in zip(self.docs, self.term_freqs, self.doc_lens):
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            scores.append((score, doc))
        return sorted(scores, key=lambda x: x[0], reverse=True)


def retrieve_sessions(
    item: Dict[str, Any],
    top_k: int,
    caption_source: Optional[str] = None,
    caption_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    docs = []
    for idx, sid, date, turns in iter_sessions(item):
        docs.append(
            {
                "session_index": idx,
                "session_id": sid,
                "date": date,
                "text": session_to_text(sid, date, turns, caption_source, caption_cache),
            }
        )
    ranked = BM25Index(docs).score(item.get("question", ""))[:top_k]
    selected = [{**doc, "score": score} for score, doc in ranked]
    answer_ids = set(item.get("answer_session_ids") or [])
    retrieved_ids = [doc["session_id"] for doc in selected]
    log = {
        "question_id": item.get("question_id"),
        "question_type": item.get("question_type"),
        "answer_session_ids": list(answer_ids),
        "retrieved_session_ids": retrieved_ids,
        "session_hit": bool(answer_ids & set(retrieved_ids)) if answer_ids else None,
        "session_all_hit": answer_ids <= set(retrieved_ids) if answer_ids else None,
        "scores": [doc["score"] for doc in selected],
    }
    return selected, log


def build_answer_prompt(
    item: Dict[str, Any],
    context: str,
    system_note: str = "Answer using only the provided memory context.",
) -> str:
    return (
        f"{system_note}\n"
        "If the context does not support an answer, respond with \"Insufficient information\".\n\n"
        f"Memory Context:\n{context}\n\n"
        f"Question Date: {item.get('question_date', 'unknown')}\n"
        f"Question Type: {item.get('question_type', 'unknown')}\n"
        f"Question: {item.get('question', '')}\n\n"
        "Answer with a short phrase only."
    )


class TextGenerator:
    def __init__(self, model_path: str, load_in_4bit: bool = True, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def score_prediction(raw_output: str, reference: str) -> Dict[str, Any]:
    prediction = parse_model_output(raw_output)
    return {
        "prediction": prediction,
        "parsed_output": normalize_answer(prediction),
        "sub_em": int(sub_em(prediction, reference)),
        "f1": f1_score(prediction, reference)[0],
    }


def finalize_run(
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
    start_time: float,
    output_dir: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, _ = compute_metrics(results)
    averaged = {
        "sub_em": sum(float(x.get("sub_em", 0)) for x in results) / len(results) if results else 0.0,
        "f1": sum(float(x.get("f1", 0.0)) for x in results) / len(results) if results else 0.0,
        "input_len": sum(float(x.get("input_len", 0)) for x in results) / len(results) if results else 0.0,
        "output_len": sum(float(x.get("output_len", 0)) for x in results) / len(results) if results else 0.0,
    }
    payload = {
        "args": vars(args),
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged,
        "throughput": len(results) / max(time.time() - start_time, 1e-6),
    }
    if extra:
        payload.update(extra)
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
