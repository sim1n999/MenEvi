from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def monitor_gpu(stop: threading.Event, samples: list[int]) -> None:
    if not shutil.which("nvidia-smi"):
        return
    command = ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"]
    while not stop.wait(0.5):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
            values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
            samples.append(sum(values))
        except (OSError, subprocess.SubprocessError, ValueError):
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")
    stop, samples = threading.Event(), []
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    start = time.time()
    monitor.start()
    completed = subprocess.run(command, check=False)
    stop.set()
    monitor.join(timeout=2)
    report = {
        "label": args.label,
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.time() - start,
        "peak_observed_gpu_memory_mib": max(samples) if samples else None,
        "gpu_samples": len(samples),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
