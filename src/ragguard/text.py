"""Small text helpers shared by retrieval, the scope gate and the groundedness guard.

Everything here is deliberately simple and dependency free so behaviour is easy to reason
about and test. A production system would use a proper tokeniser and embeddings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS = frozenset(
    {
        "a", "about", "am", "an", "and", "any", "are", "as", "at", "bank", "be", "been", "by",
        "can", "could", "cost", "costs", "did", "do", "does", "for", "from", "get", "give",
        "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "long", "many",
        "me", "much", "my", "northbridge", "of", "offer", "offers", "often", "on", "or", "our",
        "over", "please", "quickly", "should", "so", "soon", "than", "that", "the", "then",
        "there", "these", "this", "those", "to", "tell", "was", "we", "were", "what", "when",
        "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your",
    }
)  # fmt: skip

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SUFFIXES = ("ing", "ed", "es", "s", "e")
_MIN_STEM = 3


def stem(token: str) -> str:
    """Strip one common English suffix. Crude, but consistent between queries and documents."""
    for suffix in _SUFFIXES:
        if suffix == "s" and token.endswith("ss"):
            continue
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-case, drop stopwords and stem. Numbers are kept as separate digit tokens."""
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def word_tokens(text: str) -> list[str]:
    """Like content_tokens but without pure digit tokens (numbers are checked separately)."""
    return [t for t in content_tokens(text) if not t.isdigit()]


def is_covered(token: str, vocabulary: Iterable[str] | frozenset[str]) -> bool:
    """True if the token is in the vocabulary, or is a prefix of a longer vocabulary word.

    The prefix rule lets 'acknowledge' match 'acknowledgement' without a full stemmer.
    """
    vocab = vocabulary if isinstance(vocabulary, frozenset) else frozenset(vocabulary)
    if token in vocab:
        return True
    return len(token) >= 5 and any(word.startswith(token) for word in vocab)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def extract_numbers(text: str) -> set[str]:
    """Numbers as normalised strings: '£20,000' -> '20000', '39.9%' -> '39.9'."""
    return {match.replace(",", "") for match in _NUMBER_RE.findall(text)}


def normalise_for_match(text: str) -> str:
    """Lower-case, remove thousands separators and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d)", "", text.lower())).strip()


def contains_fact(answer: str, fact: str) -> bool:
    """Whole-token match so that '£5' does not match '£50' and '3%' does not match '13%'."""
    pattern = r"(?<![\w.])" + re.escape(normalise_for_match(fact)) + r"(?![\w])"
    return re.search(pattern, normalise_for_match(answer)) is not None
