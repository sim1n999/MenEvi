"""Paired bootstrap statistics for Reliability-gate validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPED_SCRIPTS = (
    PROJECT_ROOT
    / "typed_evidence"
    / "scripts"
)
if str(TYPED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TYPED_SCRIPTS))

from eval_v21 import load_items, load_json, save_json, score_row  # noqa: arithmetic repair02


def rows(payload):
    if isinstance(payload, dict):
        return payload.get("data", [])
    if isinstance(payload, list):
        return payload
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    items = {
        str(item["question_id"]): item
        for item in load_items(args.dataset)
    }
    baseline = {
        str(row["question_id"]): score_row(row, items[str(row["question_id"])])
        for row in rows(load_json(args.baseline))
    }
    candidate = {
        str(row["question_id"]): score_row(row, items[str(row["question_id"])])
        for row in rows(load_json(args.candidate))
    }
    if set(baseline) != set(items) or set(candidate) != set(items):
        raise ValueError("Prediction IDs must exactly match the dataset")

    ids = sorted(items)
    differences = np.asarray(
        [
            int(candidate[qid]["sub_em_v21"])
            - int(baseline[qid]["sub_em_v21"])
            for qid in ids
        ],
        dtype=np.int8,
    )
    rng = np.random.default_rng(args.seed)
    values = np.empty(args.samples, dtype=np.float64)
    batch = 5000
    for start in range(0, args.samples, batch):
        count = min(batch, args.samples - start)
        indices = rng.integers(
            0,
            len(differences),
            size=(count, len(differences)),
        )
        values[start : start + count] = (
            differences[indices].mean(axis=1) * 100
        )

    result = {
        "count": len(ids),
        "seed": args.seed,
        "bootstrap_samples": args.samples,
        "delta_sub_em_points": float(differences.mean() * 100),
        "paired_bootstrap_95_ci_points": [
            float(value)
            for value in np.percentile(values, [2.5, 97.5])
        ],
        "bootstrap_probability_delta_le_zero": float(
            np.mean(values <= 0)
        ),
        "wins": int(np.sum(differences == 1)),
        "losses": int(np.sum(differences == -1)),
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

