"""Evaluation harness: accuracy, hallucination and informativeness with bootstrap CIs."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import Answer, RagPipeline
from .text import contains_fact, extract_numbers

DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "eval_set.jsonl"
UNANSWERABLE = {"unanswerable", "near_miss"}


def load_eval_set(path: str | Path = DEFAULT_EVAL) -> list[dict[str, Any]]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


@dataclass
class ItemResult:
    id: str
    category: str
    abstained: bool
    correct: bool  # the system did the right thing for this item
    hallucinated: bool  # gave an answer it should not have, or one with invented numbers
    informative: bool  # answerable question answered with all key facts
    answer: str
    reason: str


def score_item(item: dict[str, Any], result: Answer, gold_text: str) -> ItemResult:
    category = item["category"]
    if category in UNANSWERABLE:
        hallucinated = not result.abstained
        return ItemResult(
            item["id"], category, result.abstained, result.abstained, hallucinated,
            False, result.answer, result.reason,
        )  # fmt: skip
    has_facts = (not result.abstained) and all(
        contains_fact(result.answer, f) for f in item["key_facts"]
    )
    invented = (not result.abstained) and bool(
        extract_numbers(result.answer) - extract_numbers(gold_text)
    )
    return ItemResult(
        item["id"], category, result.abstained, has_facts and not invented, invented,
        has_facts, result.answer, result.reason,
    )  # fmt: skip


def bootstrap_ci(
    values: Sequence[bool], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of a binary metric."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    data = [1.0 if v else 0.0 for v in values]
    means = sorted(sum(rng.choices(data, k=n)) / n for _ in range(n_boot))
    low = means[int((alpha / 2) * n_boot)]
    high = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(data) / n, low, high


def evaluate(
    pipeline: RagPipeline,
    items: Sequence[dict[str, Any]] | None = None,
    gold_lookup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    items = list(items if items is not None else load_eval_set())
    docs = {c.doc: "" for c in pipeline.retriever.chunks}
    for c in pipeline.retriever.chunks:
        docs[c.doc] += " " + c.text
    lookup = gold_lookup or (lambda doc: docs.get(doc, ""))

    results = [
        score_item(
            it, pipeline.ask(it["question"]), lookup(it["gold_doc"]) if it["gold_doc"] else ""
        )
        for it in items
    ]
    answerable = [r for r in results if r.category == "answerable"]
    unanswerable = [r for r in results if r.category in UNANSWERABLE]

    def metric(name: str, values: list[bool]) -> dict[str, Any]:
        mean, low, high = bootstrap_ci(values)
        return {"metric": name, "value": mean, "ci95": [low, high], "n": len(values)}

    metrics = [
        metric("accuracy_overall", [r.correct for r in results]),
        metric("informativeness_answerable", [r.informative for r in answerable]),
        metric("hallucination_rate_unanswerable", [r.hallucinated for r in unanswerable]),
        metric("hallucination_rate_answerable", [r.hallucinated for r in answerable]),
        metric("abstention_correct_unanswerable", [r.abstained for r in unanswerable]),
        metric("false_abstention_answerable", [r.abstained for r in answerable]),
    ]
    return {
        "backend": pipeline.backend.name,
        "use_gate": pipeline.use_gate,
        "use_guard": pipeline.use_guard,
        "metrics": {m["metric"]: m for m in metrics},
        "items": [r.__dict__ for r in results],
    }


def format_table(reports: dict[str, dict[str, Any]]) -> str:
    names = list(next(iter(reports.values()))["metrics"])
    header = "| Metric | " + " | ".join(reports) + " |"
    lines = [header, "|---|" + "---|" * len(reports)]
    for name in names:
        cells = []
        for report in reports.values():
            m = report["metrics"][name]
            cells.append(f"{m['value']:.0%} ({m['ci95'][0]:.0%}–{m['ci95'][1]:.0%})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
