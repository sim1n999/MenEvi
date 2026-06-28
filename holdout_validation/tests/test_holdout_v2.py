from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compiler = load_module(
    "h_compiler_v2",
    SCRIPTS / "build_typed_evidence_v2.py",
)
runner = load_module(
    "h_runner_v2",
    SCRIPTS / "run_typed_specialists_v2.py",
)


def boundary(kind: str, date: str, score: float = 1.0):
    return {
        "boundary_id": f"b_{kind}_{date}",
        "boundary_kind": kind,
        "explicit_dates": [date],
        "observation_date": date,
        "session_id": "s1",
        "score": score,
        "evidence": f"{kind} evidence",
    }


def test_interval_candidate_rejects_reverse_dates():
    row = compiler.candidate_interval(
        1,
        "activity",
        boundary("start", "2024/02/01"),
        boundary("end", "2024/01/01"),
    )
    assert row is None


def test_interval_candidate_has_deterministic_days():
    row = compiler.candidate_interval(
        1,
        "activity",
        boundary("start", "2024/01/01"),
        boundary("end", "2024/01/11"),
    )
    assert row is not None
    assert row["duration_days"] == 10


def test_interval_selector_calculates_answer():
    packet = {
        "typed_evidence": {
            "durations": [
                {
                    "intervals": [
                        {
                            "interval_id": "duration_1_interval_01",
                            "duration_index": 1,
                            "duration_days": 10,
                        }
                    ]
                },
                {
                    "intervals": [
                        {
                            "interval_id": "duration_2_interval_01",
                            "duration_index": 2,
                            "duration_days": 3,
                        }
                    ]
                },
            ]
        }
    }
    output = (
        '{"duration_1_interval":"duration_1_interval_01",'
        '"duration_2_interval":"duration_2_interval_01"}'
    )
    answer, trace = runner.selected_interval_answer(output, packet)
    assert answer == "A"
    assert trace["calculator_output"] == "A"


def test_transaction_selector_uses_decimal_sum():
    packet = {
        "typed_evidence": {
            "groups": [
                {
                    "group_id": "txn_01",
                    "currency": "$",
                    "total_amount": "10.10",
                    "status": "completed",
                    "month_match": True,
                },
                {
                    "group_id": "txn_02",
                    "currency": "$",
                    "total_amount": "2.20",
                    "status": "completed",
                    "month_match": True,
                },
            ]
        }
    }
    output = '{"selected_group_ids":["txn_01","txn_02"]}'
    answer, trace = runner.selected_transaction_total(output, packet)
    assert answer == "$12.30"
    assert trace["calculator_output"] == "$12.30"


def test_visual_prompt_never_exposes_internal_paths():
    packet = {
        "question_subtype": "entity",
        "question": "What color is it?",
    }
    observations = {
        "data": [
            {
                "execution_mode": "model",
                "target_id": "visual_01",
                "image_id": "needle_images/private.jpg",
                "image_path": "/tmp/haystack_images/private.jpg",
                "observation": {"colors": ["red"]},
            }
        ]
    }
    prompt = runner.visual_prompt(packet, observations, "policy")
    assert "visual_01" in prompt
    assert "needle_images" not in prompt
    assert "haystack_images" not in prompt
    assert "private.jpg" not in prompt
