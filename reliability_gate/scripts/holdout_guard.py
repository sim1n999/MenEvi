"""Guard the one permitted formal Reliability-gate validation holdout execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


GATE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GATE_ROOT.parent
HOLDOUT_ROOT = PROJECT_ROOT / "holdout_validation"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_items(path: Path) -> List[Dict[str, Any]]:
    value = load_json(path)
    items = value.get("data", value) if isinstance(value, dict) else value
    if not isinstance(items, list):
        raise TypeError(f"Expected list payload: {path}")
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_from_payload(path: Path) -> List[str]:
    return sorted(str(row["question_id"]) for row in load_items(path))


def begin(args: argparse.Namespace) -> None:
    marker = Path(args.marker)
    dataset = Path(args.dataset)
    base = Path(args.base)
    frozen = Path(args.frozen_config)
    expected_ids = sorted(
        str(value)
        for value in load_json(HOLDOUT_ROOT / "protocol" / "holdout_ids.json")
    )
    if ids_from_payload(dataset) != expected_ids:
        raise RuntimeError("Holdout dataset IDs do not match the frozen protocol")
    if ids_from_payload(base) != expected_ids:
        raise RuntimeError("Baseline IDs do not match the frozen protocol")

    identity = {
        "dataset_sha256": sha256_file(dataset),
        "base_sha256": sha256_file(base),
        "frozen_config_sha256": sha256_file(frozen),
        "holdout_count": len(expected_ids),
    }
    if marker.exists():
        previous = load_json(marker)
        if not args.resume:
            raise FileExistsError(
                "Formal reliability-gate holdout already started; --resume is only for the "
                "same interrupted execution"
            )
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Resume inputs differ from the original execution")
        if previous.get("status") == "complete":
            raise RuntimeError("Formal reliability-gate holdout execution is already complete")
        print(f"Resuming the same formal reliability-gate execution: {marker}")
        return

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "experiment": "reliability_gate",
                "status": "started",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                **identity,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Formal reliability-gate holdout marker created: {marker}")


def finish(args: argparse.Namespace) -> None:
    marker = Path(args.marker)
    if not marker.is_file():
        raise FileNotFoundError("Formal reliability-gate run marker is missing")
    payload = load_json(marker)
    if payload.get("status") == "complete":
        raise RuntimeError("Formal reliability-gate holdout execution is already complete")
    result = Path(args.result)
    comparison = Path(args.comparison)
    bootstrap = Path(args.bootstrap)
    payload.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_sha256": sha256_file(result),
            "comparison_sha256": sha256_file(comparison),
            "bootstrap_sha256": sha256_file(bootstrap),
        }
    )
    marker.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Formal reliability-gate holdout marked complete: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("begin")
    start.add_argument("--dataset", required=True)
    start.add_argument("--base", required=True)
    start.add_argument("--frozen-config", required=True)
    start.add_argument("--marker", required=True)
    start.add_argument("--resume", action="store_true")
    start.set_defaults(function=begin)
    done = commands.add_parser("finish")
    done.add_argument("--marker", required=True)
    done.add_argument("--result", required=True)
    done.add_argument("--comparison", required=True)
    done.add_argument("--bootstrap", required=True)
    done.set_defaults(function=finish)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()



