#!/usr/bin/env python3
"""Quick sanity checks for the local MEMLENS reproduction layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--qwen-vl-path", type=Path, default=None)
    parser.add_argument("--qwen-llm-path", type=Path, default=None)
    args = parser.parse_args()

    root = args.repro_root.resolve()
    data_root = root / "data" / "memlens"
    code_root = root / "MEMLENS"
    image_dir = data_root / "release_images"
    agent_root = root / "data" / "memlens_agent_subset"

    qwen_vl = args.qwen_vl_path or Path(os.environ.get("QWEN25_VL_PATH", ""))
    qwen_llm = args.qwen_llm_path or Path(os.environ.get("QWEN25_LLM_PATH", ""))

    checks = [
        ("official code", code_root),
        ("dataset root", data_root),
        ("image dir", image_dir),
        ("Qwen2.5-VL path", (root / qwen_vl).resolve() if not qwen_vl.is_absolute() else qwen_vl),
        ("Qwen2.5 text path", (root / qwen_llm).resolve() if not qwen_llm.is_absolute() else qwen_llm),
    ]

    print(f"Repro root: {root}")
    for label, path in checks:
        print(f"{label:20s}: {'OK' if path.exists() else 'MISSING'}  {path}")

    for name in ("32k", "64k", "128k", "256k"):
        path = data_root / f"dataset_{name}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"dataset_{name}.json     : OK  {len(data)} items")
        else:
            print(f"dataset_{name}.json     : MISSING")

    if image_dir.exists():
        print(f"release_images files : {count_files(image_dir)}")

    if agent_root.exists():
        for name in ("32k", "64k", "128k", "256k"):
            path = agent_root / f"dataset_{name}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"agent subset {name:4s} : OK  {len(data)} items")
            else:
                print(f"agent subset {name:4s} : MISSING")


if __name__ == "__main__":
    main()
