from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def session_text(session, include_captions: bool) -> str:
    turns = session.get("session", []) if isinstance(session, dict) else session
    parts = []
    for turn in turns:
        parts.append(str(turn.get("content", "")))
        if include_captions:
            for image in turn.get("images", []) or []:
                if isinstance(image, dict):
                    parts.append(str(image.get("blip_caption", "")))
    return " ".join(parts)


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    n, avgdl = len(tokenized), sum(map(len, tokenized)) / max(len(tokenized), 1)
    df = Counter(token for doc in tokenized for token in set(doc))
    q = Counter(tokenize(query))
    scores = []
    for doc in tokenized:
        tf, score = Counter(doc), 0.0
        for token, qf in q.items():
            if not tf[token]:
                continue
            idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
            denom = tf[token] + 1.5 * (1 - 0.75 + 0.75 * len(doc) / max(avgdl, 1))
            score += qf * idf * tf[token] * 2.5 / denom
        scores.append(score)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-captions", action="store_true")
    args = parser.parse_args()
    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    traces = []
    for item in items:
        sessions = item.get("haystack_sessions", [])
        scores = bm25_scores(str(item.get("question", "")), [session_text(s, args.include_captions) for s in sessions])
        selected = sorted(range(len(sessions)), key=lambda index: (-scores[index], index))[: args.top_k]
        item["haystack_sessions"] = [sessions[index] for index in selected]
        for key in ("haystack_dates", "haystack_session_ids"):
            values = item.get(key, [])
            item[key] = [values[index] for index in selected]
        traces.append({"question_id": str(item["question_id"]), "selected_indices": selected,
                       "scores": [scores[index] for index in selected],
                       "label_fields_used": False, "include_captions": args.include_captions})
    Path(args.output).write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    Path(args.trace).write_text(json.dumps(traces, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

