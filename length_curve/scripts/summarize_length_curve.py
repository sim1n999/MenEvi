from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for length in ("32k", "64k", "128k", "256k"):
        path = Path(args.results_root) / length / "p1_vs_p0.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))["overall"]
        rows.append({
            "length": length,
            "n": report["count"],
            "p0_sub_em_pct": 100 * report["baseline_sub_em_v21"],
            "p1_sub_em_pct": 100 * report["candidate_sub_em_v21"],
            "delta_sub_em_pp": report["delta_sub_em_points"],
            "p0_token_f1_pct": 100 * report["baseline_f1_v21"],
            "p1_token_f1_pct": 100 * report["candidate_f1_v21"],
            "wins": report["wins"],
            "losses": report["losses"],
            "mcnemar_exact_p": report["mcnemar_exact_two_sided_p"],
        })
    if not rows:
        raise RuntimeError("No completed length comparisons found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
