from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = [
    "answer_refusal", "arithmetic", "counting", "duration_comparison",
    "entity", "entity_resolution", "knowledge_update", "order_ranking",
    "previnfo", "temporal_info_extraction",
]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def operational_route(subtype: str) -> str:
    if subtype in {"arithmetic", "duration_comparison"}:
        return f"specialist_{subtype}"
    if subtype in {"entity", "previnfo", "knowledge_update", "temporal_info_extraction"}:
        return "D"
    return "C"


def load_rows(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw.get("data", raw) if isinstance(raw, dict) else raw


def tokens(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def synthetic_examples(path: str) -> list[tuple[str, str]]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = str(row.get("metadata", {}).get("question_subtype", ""))
            user = next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"), "")
            matches = re.findall(r"^Question:\s*(.+)$", user, flags=re.M)
            if label in LABELS and matches:
                examples.append((matches[-1].strip(), label))
    if not examples:
        raise RuntimeError("No synthetic router examples found")
    return examples


class NaiveBayes:
    def __init__(self, examples: list[tuple[str, str]]) -> None:
        self.counts = {label: Counter() for label in LABELS}
        self.totals = Counter()
        self.docs = Counter()
        self.vocab = set()
        for text, label in examples:
            values = tokens(text)
            self.counts[label].update(values)
            self.totals[label] += len(values)
            self.docs[label] += 1
            self.vocab.update(values)
        self.n_docs = sum(self.docs.values())

    def predict(self, text: str) -> tuple[str, float]:
        values = tokens(text)
        scores = {}
        for label in LABELS:
            score = math.log((self.docs[label] + 1) / (self.n_docs + len(LABELS)))
            denominator = self.totals[label] + len(self.vocab)
            score += sum(math.log((self.counts[label][token] + 1) / denominator) for token in values)
            scores[label] = score
        ordered = sorted(scores, key=scores.get, reverse=True)
        margin = scores[ordered[0]] - scores[ordered[1]]
        return ordered[0], margin


def rule_predict(question: str) -> tuple[str, str]:
    q = question.lower()
    rules = [
        ("duration_comparison", r"longer|duration|how long|lasted longer"),
        ("order_ranking", r"order|earliest|latest|first to last|chronological"),
        ("arithmetic", r"total|altogether|how much.*spend|sum of"),
        ("counting", r"how many|number of"),
        ("knowledge_update", r"now|current|currently|updated|changed (?:my|the)"),
        ("temporal_info_extraction", r"what date|when did|which day|what time"),
        ("entity_resolution", r"which (?:one|person|item)|same (?:person|object)|who was"),
        ("previnfo", r"previous|before|earlier|used to|last time"),
        ("answer_refusal", r"do you know|did i ever|have i ever|was there any"),
    ]
    for label, pattern in rules:
        if re.search(pattern, q):
            return label, pattern
    return "entity", "default_entity"


ROUTER_PROMPT = """Classify the memory question into exactly one subtype.
Return only one label from this list:
answer_refusal, arithmetic, counting, duration_comparison, entity,
entity_resolution, knowledge_update, order_ranking, previnfo,
temporal_info_extraction.

Definitions: arithmetic asks for a numeric sum; counting asks how many;
duration_comparison compares how long two activities lasted; order_ranking
orders events; knowledge_update asks for the newest changed state; previnfo
asks for earlier stored information; temporal_info_extraction asks for a date
or time; entity_resolution links descriptions across sessions; entity extracts
a visually or textually stored entity; answer_refusal asks about absent memory.

Question: {question}
Subtype:"""


def parse_llm_label(raw: str) -> str:
    normalized = raw.lower().strip().replace("-", "_")
    exact = normalized.strip("`'\".,:;()[]{}").replace(" ", "_")
    if exact in LABELS:
        return exact
    matches = [label for label in LABELS if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", normalized)]
    return matches[0] if len(matches) == 1 else ""


def metrics(rows: list[dict]) -> dict:
    confusion = defaultdict(Counter)
    for row in rows:
        confusion[row["gold_subtype"]][row["predicted_subtype"]] += 1
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in LABELS if g != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {
        "accuracy": sum(r["gold_subtype"] == r["predicted_subtype"] for r in rows) / len(rows),
        "operational_route_accuracy": sum(
            operational_route(r["gold_subtype"]) == operational_route(r["predicted_subtype"])
            for r in rows
        ) / len(rows),
        "macro_f1": sum(x["f1"] for x in per_label.values()) / len(LABELS),
        "per_label": per_label,
        "confusion": {g: dict(confusion[g]) for g in LABELS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=["gold", "synthetic_nb", "rules", "llm_zero_shot", "universal_c"], required=True)
    parser.add_argument("--synthetic-train")
    parser.add_argument("--model")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    items = load_rows(args.dataset)
    model = NaiveBayes(synthetic_examples(args.synthetic_train)) if args.method == "synthetic_nb" else None
    generator = None
    if args.method == "llm_zero_shot":
        if not args.model:
            parser.error("--model is required for llm_zero_shot")
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "answer_evidence" / "scripts"))
        from answer_focused_kg_answering import TextGenerator
        generator = TextGenerator(args.model, load_in_4bit=not args.no_4bit, dtype=args.dtype)
    rows = []
    for item in items:
        gold, question = str(item["question_subtype"]), str(item["question"])
        if args.method == "gold": predicted, evidence = gold, "diagnostic_gold_upper_bound"
        elif args.method == "synthetic_nb": predicted, margin = model.predict(question); evidence = f"nb_margin={margin:.6f}"
        elif args.method == "rules": predicted, evidence = rule_predict(question)
        elif args.method == "llm_zero_shot":
            generated = generator.generate(ROUTER_PROMPT.format(question=question), 24)
            raw = str(generated["output"])
            predicted = parse_llm_label(raw)
            if not predicted:
                predicted, fallback = rule_predict(question)
                evidence = f"llm_unparsed={raw!r};fallback={fallback}"
            else:
                evidence = f"llm={raw!r}"
        else: predicted, evidence = "entity_resolution", "force_C_route"
        rows.append({"question_id": str(item["question_id"]), "question": question,
                     "gold_subtype": gold, "predicted_subtype": predicted,
                     "gold_operational_route": operational_route(gold),
                     "predicted_operational_route": operational_route(predicted),
                     "method": args.method, "question_only": args.method != "gold", "evidence": evidence})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"method": args.method, "count": len(rows),
                                  "metrics": metrics(rows), "routes": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
