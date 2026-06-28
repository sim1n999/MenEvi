from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_reliability_gate.py"
spec = importlib.util.spec_from_file_location("i_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gate)


def candidate(value: str, eligible: bool = True):
    return {
        "raw_prediction": value,
        "eligible_for_merge": eligible,
    }


def test_normalized_refusal_handles_terminal_punctuation():
    assert gate.is_refusal("Insufficient information.")
    assert gate.is_refusal("  INSUFFICIENT   INFORMATION! ")


def test_rank1_supported_base_is_preserved():
    decision = gate.gate_decision(
        "Digital AV Out",
        candidate("Super Nintendo"),
        {"raw_observation": "DIGITAL AV OUT"},
    )
    assert decision["replace"] is False
    assert decision["reason"] == "preserve_rank1_supported_base"


def test_candidate_supported_by_rank1_can_replace():
    decision = gate.gate_decision(
        "Blue",
        candidate("Red"),
        {"colors": ["red"]},
    )
    assert decision["replace"] is True


def test_normalized_refusal_never_replaces():
    decision = gate.gate_decision(
        "above",
        candidate("Insufficient information."),
        {"raw_observation": "Above"},
    )
    assert decision["replace"] is False
    assert decision["reason"] == "normalized_refusal"


def test_support_uses_token_boundaries():
    assert gate.directly_supported("red", {"colors": ["red"]})
    assert not gate.directly_supported("red", {"text": "tired"})


def test_prediction_payload_can_be_wrapped_or_bare():
    rows = [{"question_id": "q1"}]
    assert gate.payload_rows(rows) == rows
    assert gate.payload_rows({"data": rows}) == rows