from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graph_transform_preserves_nodes_for_no_edges() -> None:
    module = load_module("graph_transform", ROOT / "component_ablations/scripts/transform_graphs.py")
    graph = {"nodes": [{"id": "s", "type": "StateVersion"}, {"id": "i", "type": "Image"}],
             "edges": [{"source": "s", "target": "i", "type": "before"}]}
    value = module.transform(graph, "no_edges")
    assert len(value["nodes"]) == 2
    assert value["edges"] == []
    assert graph["edges"]


def test_graph_transform_removes_incident_visual_edges() -> None:
    module = load_module("graph_transform_visual", ROOT / "component_ablations/scripts/transform_graphs.py")
    graph = {"nodes": [{"id": "t", "type": "Turn"}, {"id": "i", "type": "Image"}],
             "edges": [{"source": "t", "target": "i", "type": "has_image"}]}
    value = module.transform(graph, "no_visual")
    assert [node["id"] for node in value["nodes"]] == ["t"]
    assert value["edges"] == []


def test_rule_router_is_question_only() -> None:
    module = load_module("router", ROOT / "automatic_routing/scripts/build_route_manifest.py")
    assert module.rule_predict("How much total did I spend?")[0] == "arithmetic"
    assert module.rule_predict("Which activity lasted longer?")[0] == "duration_comparison"
    assert module.parse_llm_label("entity") == "entity"
    assert module.parse_llm_label("The subtype is entity.") == "entity"
    assert module.parse_llm_label("The answer is unclear") == ""


def test_packet_contract_does_not_need_reference_answer() -> None:
    module = load_module("packet_sanitizer", ROOT / "common/scripts/sanitize_full_packets.py")
    packet = {"question": "How much total did I spend? Answer with the exact amount.",
              "question_subtype": "arithmetic", "reference_answer": "$42.00"}
    clean = module.strip_forbidden(packet)
    assert "reference_answer" not in clean
    assert module.infer_contract(clean) == "number"


def test_empty_visual_payload_helper() -> None:
    module = load_module("empty_visual", ROOT / "common/scripts/create_empty_visual_outputs.py")
    payload = module.empty_payload("dummy.json", "dummy-output")
    assert payload["data"] == []
    assert payload["empty_visual_target_set"] is True


def test_every_group_has_required_files() -> None:
    groups = "MNOPQR"
    matches = {path.name[11]: path for path in ROOT.glob("experiment_?_*") if len(path.name) > 11 and path.name[11] in groups}
    for group in groups:
        directory = matches[group]
        for name in ("EXPERIMENT_DESIGN.md", "README.md", "config.json", "run_all.sh"):
            assert (directory / name).is_file(), f"{directory}/{name}"
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
