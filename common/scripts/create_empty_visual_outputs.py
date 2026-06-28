from __future__ import annotations

import argparse
import json
from pathlib import Path


def empty_payload(dataset: str, output_dir: str) -> dict:
    return {
        "args": {"dataset": dataset, "output_dir": output_dir},
        "data": [],
        "metrics_v21": {"overall": {"count": 0, "sub_em_count_v21": 0,
                                      "sub_em_v21": 0.0, "f1_v21": 0.0}},
        "empty_visual_target_set": True,
        "uses_reference_labels_for_predictions": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "visual_inspection" / "observations").mkdir(parents=True, exist_ok=True)
    specialist = output / "visual_specialist"
    specialist.mkdir(parents=True, exist_ok=True)
    payload = empty_payload(args.dataset, args.output_dir)
    (specialist / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (specialist / "traces.json").write_text("[]\n", encoding="utf-8")
    print(f"Created empty visual outputs under {output}")


if __name__ == "__main__":
    main()
