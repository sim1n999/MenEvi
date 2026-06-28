"""Freeze or verify the Holdout validation method configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


HOLDOUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HOLDOUT_ROOT.parent
METHOD_FILES = (
    "config.json",
    "scripts/audit_prompt_leakage.py",
    "scripts/build_typed_evidence_v2.py",
    "scripts/run_typed_specialists_v2.py",
    "scripts/merge_with_base.py",
    "prompts/typed_specialist_policy_v2.md",
    "protocol/holdout_ids.json",
    "protocol/protocol_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        value = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return value.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def method_hashes() -> Dict[str, str]:
    output = {}
    for relative in METHOD_FILES:
        path = HOLDOUT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing freeze input: {path}")
        output[relative] = sha256_file(path)
    return output


def require_passed_audit(path: Path) -> None:
    report = load_json(path)
    if report.get("passed") is not True or int(report.get("hit_count", -1)) != 0:
        raise RuntimeError(f"Leakage audit did not pass: {path}")


def freeze(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Frozen config already exists and will not be overwritten: {output}"
        )
    if not args.acknowledge_dev_reviewed:
        raise RuntimeError("--acknowledge-dev-reviewed is required")

    results = HOLDOUT_ROOT / "results" / "dev_195"
    comparison_path = results / "holdout_merged" / "comparison.json"
    visual_audit = results / "visual_prompt_audit" / "leakage_audit.json"
    specialist_audit = (
        results / "specialist_prompt_audit" / "leakage_audit.json"
    )
    for path in (comparison_path, visual_audit, specialist_audit):
        if not path.is_file():
            raise FileNotFoundError(
                f"Complete the full development run before freezing: {path}"
            )
    require_passed_audit(visual_audit)
    require_passed_audit(specialist_audit)

    comparison = load_json(comparison_path)
    protocol = load_json(HOLDOUT_ROOT / "protocol" / "protocol_manifest.json")
    payload = {
        "status": "frozen_for_single_holdout_run",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "acknowledge_dev_reviewed": True,
        "git_commit": git_commit(),
        "text_model": args.text_model,
        "vision_model": args.vision_model,
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "load_in_4bit": not args.no_4bit,
        },
        "method": {
            "strategy": "h_typed_evidence_v2",
            "visual_top_k": 3,
            "target_subtypes": [
                "arithmetic",
                "duration_comparison",
                "entity",
                "previnfo",
            ],
        },
        "development_result": comparison.get("overall"),
        "holdout_protocol": {
            "holdout_count": protocol.get("holdout_count"),
            "holdout_ids_sha256": protocol.get("holdout_ids_sha256"),
        },
        "file_sha256": method_hashes(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen H configuration written once to {output}")


def verify(args: argparse.Namespace) -> None:
    path = Path(args.frozen_config)
    frozen = load_json(path)
    expected = frozen.get("file_sha256", {})
    actual = method_hashes()
    if expected != actual:
        changed = sorted(
            key
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        )
        raise RuntimeError(
            f"Frozen method files changed after freeze: {changed}"
        )
    protocol = load_json(HOLDOUT_ROOT / "protocol" / "protocol_manifest.json")
    if (
        frozen.get("holdout_protocol", {}).get("holdout_ids_sha256")
        != protocol.get("holdout_ids_sha256")
    ):
        raise RuntimeError("Holdout protocol hash changed after freeze")
    print(f"Frozen H configuration verified: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--output",
        default=str(HOLDOUT_ROOT / "frozen_config.json"),
    )
    freeze_parser.add_argument("--text-model", required=True)
    freeze_parser.add_argument("--vision-model", required=True)
    freeze_parser.add_argument("--no-4bit", action="store_true")
    freeze_parser.add_argument(
        "--acknowledge-dev-reviewed",
        action="store_true",
    )
    freeze_parser.set_defaults(function=freeze)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--frozen-config",
        default=str(HOLDOUT_ROOT / "frozen_config.json"),
    )
    verify_parser.set_defaults(function=verify)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

