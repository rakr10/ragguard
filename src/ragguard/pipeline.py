"""The RAG pipeline: retrieve, scope-gate, generate, guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .corpus import load_corpus
from .guard import GuardResult, check_groundedness
from .llm import NO_ANSWER, Backend, make_backend
from .retrieval import BM25Retriever
from .text import content_tokens, is_covered

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    reason: str  # "answered", "out_of_scope", "no_evidence", "no_answer", "unsupported_answer"
    sources: list[str] = field(default_factory=list)
    query_coverage: float = 1.0
    guard: GuardResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagPipeline:
    def __init__(
        self,
        retriever: BM25Retriever,
        backend: Backend,
        *,
        use_gate: bool = True,
        use_guard: bool = True,
        min_coverage: float = 1.0,
        top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.backend = backend
        self.use_gate = use_gate
        self.use_guard = use_guard
        self.min_coverage = min_coverage
        self.top_k = top_k

    def query_coverage(self, question: str) -> float:
        """Share of the question's content terms that exist anywhere in the corpus."""
        terms = content_tokens(question)
        if not terms:
            return 0.0
        vocab = self.retriever.vocabulary
        return sum(1 for t in terms if is_covered(t, vocab)) / len(terms)

    def ask(self, question: str) -> Answer:
        coverage = round(self.query_coverage(question), 3)
        hits = self.retriever.search(question, k=self.top_k)
        contexts = [h.chunk for h in hits]
        sources = [c.chunk_id for c in contexts]

        if self.use_gate:
            if coverage < self.min_coverage:
                return Answer(question, NO_ANSWER, True, "out_of_scope", sources, coverage)
            if not contexts:
                return Answer(question, NO_ANSWER, True, "no_evidence", [], coverage)

        text = self.backend.generate(question, contexts).strip()
        if text == NO_ANSWER or not text:
            return Answer(question, NO_ANSWER, True, "no_answer", sources, coverage)

        verdict = None
        if self.use_guard:
            verdict = check_groundedness(text, contexts)
            if not verdict.supported:
                return Answer(
                    question, NO_ANSWER, True, "unsupported_answer", sources, coverage, verdict
                )
        return Answer(question, text, False, "answered", sources, coverage, verdict)


def build_pipeline(
    corpus_dir: str | Path | None = None,
    backend: str | Backend = "extractive",
    **options: Any,
) -> RagPipeline:
    chunks = load_corpus(corpus_dir or DEFAULT_CORPUS)
    resolved = make_backend(backend) if isinstance(backend, str) else backend
    return RagPipeline(BM25Retriever(chunks), resolved, **options)
