from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN = {"reference_answer", "answer_session_ids", "answer"}


def infer_contract(packet: dict) -> str:
    question = str(packet.get("question", "")).lower()
    subtype = str(packet.get("question_subtype", "")).lower()
    qtype = str(packet.get("question_type", "")).lower()
    if subtype == "answer_refusal" or qtype == "answer_refusal":
        return "refusal"
    if subtype == "duration_comparison" or 'answer with "a" or "b"' in question:
        return "ab"
    if subtype in {"arithmetic", "counting"} or re.search(r"single number|exact amount|how many", question):
        return "number"
    if "yyyy/mm/dd" in question or re.search(r"answer with (?:the )?date", question):
        return "date"
    return "phrase"


def strip_forbidden(value):
    if isinstance(value, dict):
        return {key: strip_forbidden(child) for key, child in value.items() if key not in FORBIDDEN}
    if isinstance(value, list):
        return [strip_forbidden(child) for child in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", action="append", required=True)
    args = parser.parse_args()
    count = 0
    for directory in args.packet_dir:
        for path in sorted(Path(directory).glob("*.json")):
            packet = strip_forbidden(json.loads(path.read_text(encoding="utf-8")))
            packet["contract"] = infer_contract(packet)
            packet["contract_source"] = "question_and_model_visible_predicted_type_only"
            serialized = json.dumps(packet, ensure_ascii=False, indent=2)
            if any(f'"{key}"' in serialized for key in FORBIDDEN):
                raise RuntimeError(f"Forbidden label remained in {path}")
            path.write_text(serialized, encoding="utf-8")
            count += 1
    print(f"Sanitized and recomputed contracts for {count} packets")


if __name__ == "__main__":
    main()

