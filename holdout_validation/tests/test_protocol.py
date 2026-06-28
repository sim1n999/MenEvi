from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


splitter = load_module("h_splitter", "scripts/build_holdout_split.py")
auditor = load_module("h_auditor", "scripts/audit_prompt_leakage.py")


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_holdout_is_id_difference(tmp_path: Path):
    full = [
        {
            "question_id": "q1",
            "question": "one",
            "question_type": "x",
            "question_subtype": "a",
        },
        {
            "question_id": "q2",
            "question": "two",
            "question_type": "x",
            "question_subtype": "b",
        },
        {
            "question_id": "q3",
            "question": "three",
            "question_type": "x",
            "question_subtype": "c",
        },
    ]
    touched = [full[1]]
    full_path = tmp_path / "full.json"
    touched_path = tmp_path / "touched.json"
    write(full_path, full)
    write(touched_path, touched)
    manifest = splitter.build_protocol(
        full_path, touched_path, tmp_path / "out"
    )
    assert manifest["selection_uses_labels"] is False
    assert manifest["holdout_count"] == 2
    ids = json.loads((tmp_path / "out/holdout_ids.json").read_text())
    assert ids == ["q1", "q3"]


def test_prompt_audit_rejects_path_labels(tmp_path: Path):
    (tmp_path / "bad.txt").write_text(
        "image=needle_images/example.jpg", encoding="utf-8"
    )
    findings = auditor.scan(tmp_path)
    assert len(findings) == 1
    assert findings[0]["pattern"] == "needle_images"


def test_json_audit_reads_prompt_not_internal_metadata(tmp_path: Path):
    write(
        tmp_path / "request.json",
        {
            "image_path": "/internal/needle_images/example.jpg",
            "prompt": "Inspect the pixels and return JSON.",
        },
    )
    assert auditor.scan(tmp_path) == []


def test_clean_anonymous_prompt_passes(tmp_path: Path):
    (tmp_path / "clean.txt").write_text(
        "visual_01 observation=red; visual_02 observation=uncertain",
        encoding="utf-8",
    )
    assert auditor.scan(tmp_path) == []
