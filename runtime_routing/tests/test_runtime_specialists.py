from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
E_SCRIPTS = ROOT / "runtime_routing" / "scripts"
sys.path.insert(0, str(E_SCRIPTS))

from build_specialist_packets import build_arithmetic_packet, build_duration_packet, parse_duration_labels
from runtime_hybrid_answering import calculate_selected_amounts, choose_route


class SpecialistPacketTests(unittest.TestCase):
    def test_duration_labels(self):
        question = (
            "Which duration is longer? Duration 1: The first job. "
            "Duration 2: The second job. A. First B. Second"
        )
        self.assertEqual(parse_duration_labels(question), ("The first job", "The second job"))

    def test_duration_requires_specific_overlap(self):
        item = {
            "question_id": "q1",
            "question": (
                "Duration 1: The Ojota Bus Terminal. "
                "Duration 2: The Marina Business Hub. A. One B. Two"
            ),
            "question_type": "temporal_reasoning",
            "question_subtype": "duration_comparison",
            "answer": "A",
        }
        graph = {
            "nodes": [
                {"id": "x", "type": "Fact", "text": "A generic bus was late.", "date": "2020/01/01"},
                {"id": "a", "type": "Fact", "text": "Ojota Bus Terminal opened.", "date": "2020/01/01", "session_id": "s1"},
                {"id": "b", "type": "Fact", "text": "Ojota Bus Terminal closed.", "date": "2020/02/01", "session_id": "s2"},
            ],
            "edges": [],
        }
        packet = build_duration_packet(item, graph)
        self.assertEqual(len(packet["duration_1_evidence"]), 2)
        self.assertFalse(any("generic bus" in line.lower() for line in packet["duration_1_evidence"]))
        self.assertNotIn("reference_answer", packet)

    def test_arithmetic_candidates_have_ids(self):
        item = {
            "question_id": "q2",
            "question": "How much total have I spent on board games?",
            "question_date": "2024/05/31",
            "question_type": "multi_session_reasoning",
            "question_subtype": "arithmetic",
            "answer": "$30",
        }
        graph = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "Fact",
                    "text": "I bought a board game for $10.",
                    "date": "2024/05/01",
                    "session_id": "s1",
                },
                {
                    "id": "n2",
                    "type": "Fact",
                    "text": "I purchased another board game for $20.",
                    "date": "2024/05/02",
                    "session_id": "s2",
                },
            ],
            "edges": [],
        }
        packet = build_arithmetic_packet(item, graph)
        self.assertEqual([row["candidate_id"] for row in packet["amount_candidates"]], ["a01", "a02"])
        total, selected = calculate_selected_amounts('{"selected_ids":["a01","a02"]}', packet)
        self.assertEqual(total, "$30.00")
        self.assertEqual(selected, ["a01", "a02"])
        self.assertNotEqual(packet["calculation"]["confidence"], "high")

    def test_runtime_routes(self):
        self.assertEqual(choose_route("entity", "runtime_cd")[0], "D")
        self.assertEqual(choose_route("duration_comparison", "runtime_cd")[0], "C")
        self.assertEqual(
            choose_route("duration_comparison", "runtime_specialists")[0],
            "duration_comparison_specialist",
        )
        self.assertEqual(
            choose_route("arithmetic", "runtime_specialists_override")[0],
            "arithmetic_specialist",
        )


if __name__ == "__main__":
    unittest.main()