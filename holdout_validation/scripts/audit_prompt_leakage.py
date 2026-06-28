"""Fail closed when model-visible prompts expose benchmark-side labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PATTERNS = (
    "needle_images",
    "haystack_images",
    "answer_session_ids",
    "reference_answer",
)


def prompt_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key.lower() in {"prompt", "model_prompt", "rendered_prompt"}
                and isinstance(child, str)
            ):
                yield child
            else:
                yield from prompt_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from prompt_values(child)


def visible_texts(path: Path) -> Iterable[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".prompt"}:
        yield str(path), path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        for index, prompt in enumerate(prompt_values(raw)):
            yield f"{path}#prompt[{index}]", prompt
    elif suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            for index, prompt in enumerate(prompt_values(raw)):
                yield f"{path}:{line_number}#prompt[{index}]", prompt


def scan(
    prompt_dir: Path,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
) -> List[Dict[str, Any]]:
    compiled = [
        (pattern, re.compile(re.escape(pattern), flags=re.IGNORECASE))
        for pattern in patterns
    ]
    findings: List[Dict[str, Any]] = []
    for path in sorted(p for p in prompt_dir.rglob("*") if p.is_file()):
        for source, text in visible_texts(path):
            for pattern, regex in compiled:
                for match in regex.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append(
                        {"source": source, "line": line, "pattern": pattern}
                    )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", required=True, type=Path)
    parser.add_argument("--forbid", action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    patterns = tuple(DEFAULT_PATTERNS) + tuple(args.forbid)
    findings = scan(args.prompt_dir, patterns)
    report = {
        "prompt_dir": str(args.prompt_dir),
        "patterns": list(patterns),
        "hit_count": len(findings),
        "findings": findings,
        "passed": not findings,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
