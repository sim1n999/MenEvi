"""Focused regression tests for Typed-evidence evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


eval_v21 = load_module("eval_v21_test", ROOT / "scripts" / "eval_v21.py")
compiler = load_module("compiler_test", ROOT / "scripts" / "build_typed_evidence.py")
specialists = load_module("specialists_test", ROOT / "scripts" / "run_typed_specialists.py")


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "question_type": "multi_session_reasoning",
            "question_subtype": "arithmetic",
            "question": "How much? Answer with the exact amount.",
            "answer": "$105",
        }

    def test_decimal_equivalence(self):
        self.assertTrue(eval_v21.exact_match("$105.00", "$105", self.item))
        self.assertTrue(eval_v21.exact_match("£67.500", "£67.50", self.item))
        self.assertFalse(eval_v21.exact_match("$105.01", "$105", self.item))

    def test_numeric_information_extraction_contract(self):
        item = {"question_type": "information_extraction", "question_subtype": "entity",
                "question": "Read the value.", "answer": "$279.00"}
        self.assertEqual("number", eval_v21.infer_contract(item))
        self.assertTrue(eval_v21.exact_match("$279", "$279.00", item))

    def test_missing_values_never_match_typed_contracts(self):
        self.assertFalse(eval_v21.exact_match("", "$105", self.item))

    def test_payload_reference_cannot_override_dataset_answer(self):
        row = eval_v21.score_row({"raw_prediction": "$999", "reference_answer": "$999"}, self.item)
        self.assertEqual(0, row["sub_em_v21"])



class SpecialistToolTests(unittest.TestCase):
    def test_purchase_tool_rejects_uncertain_event(self):
        packet = {"typed_evidence": {"events": [{
            "event_id": "purchase_01", "currency": "$", "amount": "10",
            "status": "uncertain", "month_match": True,
        }]}}
        answer, trace = specialists.selected_purchase_total('{"selected_ids":["purchase_01"]}', packet)
        self.assertIsNone(answer)
        self.assertEqual("ineligible purchase event", trace["error"])

    def test_duration_tool_rejects_cross_duration_boundary(self):
        def boundary(boundary_id, duration_index, kind, date):
            return {
                "boundary_id": boundary_id, "duration_index": duration_index,
                "boundary_kind": kind, "explicit_dates": [date], "observation_date": date,
            }

        packet = {"typed_evidence": {"durations": [
            {"boundaries": [
                boundary("duration_1_01", 1, "start", "2020/01/01"),
                boundary("duration_1_02", 1, "end", "2020/03/01"),
            ]},
            {"boundaries": [
                boundary("duration_2_01", 2, "start", "2020/01/01"),
                boundary("duration_2_02", 2, "end", "2020/02/01"),
            ]},
        ]}}
        output = json.dumps({
            "duration_1_start": "duration_2_01",
            "duration_1_end": "duration_1_02",
            "duration_2_start": "duration_2_01",
            "duration_2_end": "duration_2_02",
        })
        answer, trace = specialists.selected_duration_answer(output, packet)
        self.assertIsNone(answer)
        self.assertIn("cross-duration", trace["error"])


class CompilerTests(unittest.TestCase):
    def test_purchase_events_are_local_and_leakage_free(self):
        item = {
            "question_id": "q_test",
            "question_type": "multi_session_reasoning",
            "question_subtype": "arithmetic",
            "question": "How much total have I spent on cast iron cookware?",
            "question_date": "2024/06/01",
            "answer": "$105",
            "answer_session_ids": ["forbidden"],
        }
        graph = {"nodes": [
            {"id": "f1", "type": "Fact", "session_id": "s1", "date": "2024/05/01",
             "text": "The seller had me pay exactly $60 for the cast iron Dutch oven."},
            {"id": "f2", "type": "Fact", "session_id": "s2", "date": "2024/05/02",
             "text": "I grabbed a cast iron skillet for $45."},
            {"id": "f3", "type": "Fact", "session_id": "s3", "date": "2024/05/03",
             "text": "The list price for a knife is $150."},
        ]}
        packet = compiler.build_packet(item, graph, Path("."))
        encoded = json.dumps(packet)
        self.assertNotIn("$105", encoded)
        self.assertNotIn("forbidden", encoded)
        events = packet["typed_evidence"]["events"]
        by_amount = {event["amount"]: event for event in events}
        self.assertEqual("completed", by_amount["60"]["status"])
        self.assertEqual("completed", by_amount["45"]["status"])
        self.assertEqual("uncertain", by_amount["150"]["status"])

    def test_duration_compiler_does_not_precompute_answer(self):
        item = {
            "question_id": "q_duration", "question_type": "temporal_reasoning",
            "question_subtype": "duration_comparison",
            "question": "Which is longer? Duration 1: first job Duration 2: second job A. one B. two",
        }
        graph = {"nodes": [
            {"id": "f1", "type": "Fact", "session_id": "s1", "date": "2020/01/01",
             "text": "I started my first job on 2020/01/01."},
            {"id": "f2", "type": "Fact", "session_id": "s2", "date": "2021/01/01",
             "text": "I left my first job on 2021/01/01."},
        ]}
        packet = compiler.build_packet(item, graph, Path("."))
        self.assertNotIn("programmatic_answer", json.dumps(packet))


if __name__ == "__main__":
    unittest.main()
