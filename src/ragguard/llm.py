"""Answer-generation backends.

* ExtractiveBackend: deterministic, returns the most relevant sentences verbatim. Cannot
  hallucinate, so it is the reference for the offline evaluation.
* OverconfidentMockBackend: a deterministic stand-in for an LLM that always answers and
  sometimes invents or distorts facts. It exists to exercise the guard in tests and in the
  evaluation. It is NOT a measurement of any real model.
* AnthropicBackend: calls the Anthropic Messages API over HTTPS (needs ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from .corpus import Chunk
from .text import content_tokens, is_covered, split_sentences

NO_ANSWER = "I don't know based on the documents."


class Backend(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[Chunk]) -> str: ...


class ExtractiveBackend:
    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[Chunk]) -> str:
        if not contexts:
            return NO_ANSWER
        q_terms = set(content_tokens(question))
        scored: list[tuple[float, int, str]] = []
        order = 0
        for chunk in contexts:
            for sentence in split_sentences(chunk.text):
                sentence_terms = frozenset(content_tokens(sentence))
                overlap = sum(1 for t in q_terms if is_covered(t, sentence_terms))
                scored.append((overlap, -order, sentence))
                order += 1
        scored.sort(reverse=True)
        best = [s for s in scored if s[0] > 0][: self.max_sentences]
        if not best:
            return NO_ANSWER
        best.sort(key=lambda s: -s[1])  # restore reading order
        return " ".join(s[2] for s in best)


class OverconfidentMockBackend:
    """Always answers. Deterministic pseudo-random hallucination based on a hash of the question."""

    name = "overconfident-mock"

    def __init__(self, distort_rate: float = 0.3) -> None:
        self.distort_rate = distort_rate
        self._inner = ExtractiveBackend()

    @staticmethod
    def _roll(question: str, salt: str) -> float:
        digest = hashlib.sha256(f"{salt}:{question}".encode()).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

    def generate(self, question: str, contexts: Sequence[Chunk]) -> str:
        base = self._inner.generate(question, contexts)
        fake = 2 + int(self._roll(question, "num") * 40)
        if base == NO_ANSWER:
            return (
                f"Yes, this is available to eligible customers and normally carries a "
                f"charge of £{fake} per month, subject to approval."
            )
        if self._roll(question, "distort") < self.distort_rate:
            def bump(match: re.Match[str]) -> str:
                return str(int(match.group()) + 1 + fake % 7)

            distorted, n = re.subn(r"\d+", bump, base, count=1)
            if n:
                return distorted
            return base + f" An additional charge of £{fake} may apply."
        return base


class AnthropicBackend:
    """Minimal Messages API client. Untested against the live API in this repository's CI."""

    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.environ.get("RAGGUARD_MODEL", "claude-haiku-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def generate(self, question: str, contexts: Sequence[Chunk]) -> str:
        evidence = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in contexts)
        prompt = (
            "Answer the question using ONLY the context below. Be concise. If the context does "
            f'not contain the answer, reply exactly: "{NO_ANSWER}"\n\n'
            f"Context:\n{evidence}\n\nQuestion: {question}"
        )
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 300,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        return "".join(b.get("text", "") for b in payload.get("content", [])).strip()


def make_backend(name: str) -> Backend:
    if name == "extractive":
        return ExtractiveBackend()
    if name in {"mock", "overconfident-mock"}:
        return OverconfidentMockBackend()
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"Unknown backend: {name!r} (use extractive, mock or anthropic)")
