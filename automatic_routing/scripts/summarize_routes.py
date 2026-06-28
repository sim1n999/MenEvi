from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_accuracy(path: Path) -> tuple[float, float, float]:
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return metrics["accuracy"], metrics["operational_route_accuracy"], metrics["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root)
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "comparisons"):
        manifest = directory / "route_manifest.json"
        predictions = directory / "full" / "full_predictions.json"
        if not manifest.is_file() or not predictions.is_file():
            continue
        subtype_accuracy, route_accuracy, router_macro_f1 = load_accuracy(manifest)
        payload = json.loads(predictions.read_text(encoding="utf-8"))
        metrics = payload.get("metrics_v21", {}).get("overall", {})
        rows.append({
            "method": directory.name,
            "n": payload.get("metrics_v21", {}).get("overall", {}).get("count", len(payload.get("data", []))),
            "subtype_accuracy_pct": 100 * subtype_accuracy,
            "operational_route_accuracy_pct": 100 * route_accuracy,
            "router_macro_f1_pct": 100 * router_macro_f1,
            "sub_em_v21_pct": 100 * metrics.get("sub_em_v21", 0),
            "token_f1_v21_pct": 100 * metrics.get("f1_v21", 0),
        })
    if not rows:
        raise RuntimeError("No completed automatic-routing runs")
    output = Path(args.output)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
