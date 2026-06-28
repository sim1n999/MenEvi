"""Freeze and verify the preregistered Reliability-gate validation method."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


GATE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GATE_ROOT.parent
HOLDOUT_ROOT = PROJECT_ROOT / "holdout_validation"
METHOD_FILES = (
    "reliability_gate/config.json",
    "reliability_gate/EXPERIMENT_DESIGN.md",
    "reliability_gate/scripts/apply_reliability_gate.py",
    "reliability_gate/scripts/build_visual_packets.py",
    "reliability_gate/scripts/paired_stats.py",
    "reliability_gate/scripts/holdout_guard.py",
    "reliability_gate/scripts/run_holdout.sh",
    "holdout_validation/scripts/audit_prompt_leakage.py",
    "holdout_validation/scripts/build_typed_evidence_v2.py",
    "holdout_validation/scripts/run_typed_specialists_v2.py",
    "holdout_validation/prompts/typed_specialist_policy_v2.md",
    "holdout_validation/protocol/holdout_ids.json",
    "holdout_validation/protocol/protocol_manifest.json",
    "typed_evidence/scripts/run_visual_inspection.py",
    "reliability_gate/scripts/build_caption_cache_resumable.py",
    "reliability_gate/scripts/build_label_blind_graphs.py",
    "reliability_gate/scripts/sanitize_packets.py",
    "reliability_gate/scripts/validate_holdout_assets.py",
    "reliability_gate/scripts/run_prepare_holdout_assets.sh",
    "reliability_gate/scripts/formal_run_guard.py",
    "reliability_gate/scripts/run_formal_holdout.sh",
    "memlens_repro/scripts/build_mmkg_memory.py",
    "memlens_repro/scripts/experiment_utils.py",
    "kg_retrieval/scripts/retrieve_kg_subgraphs.py",
    "answer_evidence/scripts/build_answer_focused_packets.py",
    "visual_evidence/scripts/build_visual_ocr_packets.py",
    "runtime_routing/scripts/build_specialist_packets.py",
    "runtime_routing/scripts/runtime_hybrid_answering.py",
    "answer_evidence/prompts/answer_focused_policy.md",
    "visual_evidence/prompts/type_aware_visual_ocr_policy.md",
    "runtime_routing/prompts/duration_specialist_policy.md",
    "runtime_routing/prompts/arithmetic_specialist_policy.md",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}",
             "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def method_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in METHOD_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing freeze input: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.json")):
        digest.update(file.name.encode("utf-8"))
        digest.update(sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def verify_asset_fingerprint(manifest: Dict[str, Any]) -> None:
    root = GATE_ROOT / "assets" / "holdout_594"
    caption = root / "captions" / "qwen25vl_holdout.jsonl"
    if sha256_file(caption) != manifest.get("caption_cache_sha256"):
        raise RuntimeError("Frozen caption cache changed")
    directories = {
        "graphs": root / "kg_memory" / "graphs",
        "subgraphs": root / "retrieval_budget120" / "retrieved_subgraphs",
        "c_packets": root / "c_packets" / "packets",
        "d_packets": root / "d_packets" / "packets",
        "specialist_packets": root / "specialist_packets" / "packets",
    }
    expected = manifest.get("asset_directory_sha256", {})
    actual = {name: directory_sha256(path) for name, path in directories.items()}
    if expected != actual:
        changed = sorted(name for name in directories if expected.get(name) != actual.get(name))
        raise RuntimeError(f"Frozen holdout assets changed: {changed}")


def require_passed_audit(path: Path) -> None:
    report = load_json(path)
    if report.get("passed") is not True or int(report.get("hit_count", -1)) != 0:
        raise RuntimeError(f"Leakage audit did not pass: {path}")


def freeze(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Frozen config exists and will not be overwritten: {output}"
        )
    if not args.acknowledge_dev_reviewed:
        raise RuntimeError("--acknowledge-dev-reviewed is required")

    dev = GATE_ROOT / "results" / "dev_195" / "visual_reliability_gate"
    comparison_path = dev / "comparison.json"
    bootstrap_path = dev / "paired_bootstrap.json"
    audits = (
        dev / "source_visual_leakage_audit.json",
        dev / "source_specialist_leakage_audit.json",
    )
    asset_manifest_path = GATE_ROOT / "assets" / "holdout_594" / "asset_manifest.json"
    asset_audit_path = GATE_ROOT / "assets" / "holdout_594" / "baseline_prompt_audit" / "leakage_audit.json"
    for path in (comparison_path, bootstrap_path, asset_manifest_path, asset_audit_path, *audits):
        if not path.is_file():
            raise FileNotFoundError(f"Complete I development run first: {path}")
    for path in (*audits, asset_audit_path):
        require_passed_audit(path)

    comparison = load_json(comparison_path)
    bootstrap = load_json(bootstrap_path)
    protocol = load_json(HOLDOUT_ROOT / "protocol" / "protocol_manifest.json")
    assets = load_json(asset_manifest_path)
    if assets.get("status") != "ready_for_freeze" or assets.get("holdout_count") != 594:
        raise RuntimeError("Holdout assets are not complete")
    verify_asset_fingerprint(assets)
    payload = {
        "status": "frozen_for_one_formal_holdout_execution",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "models": {
            "text": args.text_model,
            "vision": args.vision_model,
        },
        "generation": {
            "temperature": 0.0,
            "do_sample": False,
            "load_in_4bit": not args.no_4bit,
        },
        "method": {
            "strategy": "i_visual_only_leakage_safe_gate_v1",
            "visual_top_k": 3,
            "target_subtypes": ["entity", "previnfo"],
            "normalized_refusal_rejection": True,
            "rank1_base_support_preservation": True,
            "arithmetic_and_duration_excluded": True,
        },
        "development_diagnostic": {
            "overall": comparison.get("overall"),
            "paired_bootstrap": bootstrap,
            "posthoc_development_only": True,
        },
        "holdout_assets": assets,
        "holdout_protocol": {
            "holdout_count": protocol.get("holdout_count"),
            "holdout_ids_sha256": protocol.get("holdout_ids_sha256"),
            "formal_executions": 1,
        },
        "file_sha256": method_hashes(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Frozen I configuration written once to {output}")


def verify(args: argparse.Namespace) -> None:
    path = Path(args.frozen_config)
    frozen = load_json(path)
    expected = frozen.get("file_sha256", {})
    actual = method_hashes()
    if expected != actual:
        changed = sorted(
            key for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        )
        raise RuntimeError(f"Frozen method files changed after freeze: {changed}")
    protocol = load_json(HOLDOUT_ROOT / "protocol" / "protocol_manifest.json")
    if (
        frozen.get("holdout_protocol", {}).get("holdout_ids_sha256")
        != protocol.get("holdout_ids_sha256")
    ):
        raise RuntimeError("Frozen holdout protocol hash changed")
    if frozen.get("holdout_protocol", {}).get("holdout_count") != 594:
        raise RuntimeError("Frozen holdout count is not 594")
    verify_asset_fingerprint(frozen.get("holdout_assets", {}))
    print(f"Frozen I configuration verified: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("--output", default=str(GATE_ROOT / "frozen_config.json"))
    create.add_argument("--text-model", required=True)
    create.add_argument("--vision-model", required=True)
    create.add_argument("--no-4bit", action="store_true")
    create.add_argument("--acknowledge-dev-reviewed", action="store_true")
    create.set_defaults(function=freeze)
    check = commands.add_parser("verify")
    check.add_argument(
        "--frozen-config", default=str(GATE_ROOT / "frozen_config.json")
    )
    check.set_defaults(function=verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()


