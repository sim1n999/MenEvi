from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def metric(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics_v21", {}).get("overall", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    methods = ["bm25_text", "caption_rag", "direct_lvlm", "flat_mm_text", "flat_mm_caption"]
    sources = {name: root / name / "predictions_v21.json" for name in methods}
    sources["mmkg_full"] = Path(args.ours)
    rows = []
    for name, path in sources.items():
        values = metric(path)
        rows.append({"method": name, "n": values.get("count"),
                     "sub_em_v21_pct": 100 * values.get("sub_em_v21", 0),
                     "token_f1_v21_pct": 100 * values.get("f1_v21", 0)})
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
