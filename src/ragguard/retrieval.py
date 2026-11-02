"""BM25 retrieval implemented from scratch (no external dependencies)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .corpus import Chunk
from .text import content_tokens


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class BM25Retriever:
    """Okapi BM25 over paragraph chunks."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        tokenised = [content_tokens(c.search_text) for c in self.chunks]
        self._term_freqs = [Counter(tokens) for tokens in tokenised]
        self._lengths = [len(tokens) for tokens in tokenised]
        self._avg_len = sum(self._lengths) / len(self._lengths)
        doc_freq: Counter[str] = Counter()
        for tokens in tokenised:
            doc_freq.update(set(tokens))
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}
        self.vocabulary: frozenset[str] = frozenset(doc_freq)

    def search(self, query: str, k: int = 3) -> list[Retrieved]:
        """Return up to k chunks with a positive score, best first."""
        query_terms = set(content_tokens(query))
        scored: list[Retrieved] = []
        for chunk, freqs, length in zip(self.chunks, self._term_freqs, self._lengths):
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += self._idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append(Retrieved(chunk, score))
        scored.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return scored[:k]
