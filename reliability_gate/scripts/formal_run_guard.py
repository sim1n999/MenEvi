"""Guard baseline and gated predictions inside one formal holdout execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_ROOT = ROOT / "holdout_validation"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def items(path):
    value = load(path)
    return value.get("data", value) if isinstance(value, dict) else value


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expected_ids():
    return sorted(map(str, load(HOLDOUT_ROOT / "protocol" / "holdout_ids.json")))


def ids(path):
    return sorted(str(row["question_id"]) for row in items(path))


def begin(args):
    marker = Path(args.marker)
    expected = expected_ids()
    if ids(args.dataset) != expected:
        raise RuntimeError("Formal dataset IDs differ from frozen holdout IDs")
    identity = {
        "dataset_sha256": digest(args.dataset),
        "frozen_config_sha256": digest(args.frozen_config),
        "holdout_count": len(expected),
    }
    if marker.exists():
        previous = load(marker)
        if not args.resume:
            raise FileExistsError("Formal execution already started")
        if previous.get("status") == "complete":
            raise RuntimeError("Formal execution is complete")
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Resume identity differs from original execution")
        print(f"Resuming formal reliability-gate execution: {marker}")
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "experiment": "reliability_gate", "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Formal reliability-gate execution started: {marker}")


def register_base(args):
    marker = Path(args.marker)
    payload = load(marker)
    if ids(args.base) != expected_ids():
        raise RuntimeError("Baseline IDs differ from frozen holdout IDs")
    value = digest(args.base)
    previous = payload.get("baseline_sha256")
    if previous and previous != value:
        raise RuntimeError("Baseline changed after registration")
    payload["baseline_sha256"] = value
    payload["baseline_registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registered baseline inside formal execution: {value}")


def finish(args):
    marker = Path(args.marker)
    payload = load(marker)
    if not payload.get("baseline_sha256"):
        raise RuntimeError("Baseline was not registered")
    if payload.get("status") == "complete":
        raise RuntimeError("Formal execution is already complete")
    payload.update({
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gated_sha256": digest(args.result),
        "comparison_sha256": digest(args.comparison),
        "bootstrap_sha256": digest(args.bootstrap),
    })
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Formal reliability-gate execution completed: {marker}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("begin")
    start.add_argument("--dataset", required=True)
    start.add_argument("--frozen-config", required=True)
    start.add_argument("--marker", required=True)
    start.add_argument("--resume", action="store_true")
    start.set_defaults(function=begin)
    base = commands.add_parser("register-base")
    base.add_argument("--marker", required=True)
    base.add_argument("--base", required=True)
    base.set_defaults(function=register_base)
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



