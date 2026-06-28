from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.comparison_dir).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))["overall"]
        rows.append({"contrast": path.stem, "n": row["count"],
                     "full_correct": row["baseline_correct"],
                     "ablation_correct": row["candidate_correct"],
                     "delta_vs_full_pp": row["delta_sub_em_points"],
                     "wins": row["wins"], "losses": row["losses"],
                     "mcnemar_exact_p": row["mcnemar_exact_two_sided_p"]})
    if not rows:
        raise RuntimeError("No ablation comparisons found")
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
