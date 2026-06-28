#!/usr/bin/env python3
"""Check whether the active Python environment can load Qwen2.5-VL."""

from __future__ import annotations

import importlib
import sys


def check_module(name: str) -> None:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "<unknown>")
        path = getattr(module, "__file__", "<unknown>")
        print(f"{name}: OK  version={version}  path={path}")
    except Exception as exc:
        print(f"{name}: FAIL  {exc}")
        raise


def main() -> None:
    print(f"python executable: {sys.executable}")
    print(f"python version   : {sys.version}")

    check_module("transformers")
    check_module("accelerate")
    check_module("qwen_vl_utils")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

        print("Qwen2_5_VLForConditionalGeneration: OK")
    except ImportError as exc:
        print("Qwen2_5_VLForConditionalGeneration: FAIL")
        print(exc)
        print()
        print("Fix with the same python executable:")
        print(
            "  python -m pip install -U "
            "\"transformers>=4.51.0\" accelerate qwen-vl-utils"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
