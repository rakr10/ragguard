"""Groundedness guard: reject answers whose numbers or content words are not in the evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .corpus import Chunk
from .text import extract_numbers, is_covered, word_tokens


@dataclass(frozen=True)
class GuardResult:
    supported: bool
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_terms: list[str] = field(default_factory=list)
    word_support: float = 1.0


def check_groundedness(
    answer: str, contexts: Sequence[Chunk], min_word_support: float = 0.8
) -> GuardResult:
    """Check an answer against the retrieved chunks.

    Two independent checks, both must pass:
    1. Every number in the answer appears in the evidence (catches invented fees and rates).
    2. At least `min_word_support` of the answer's content words appear in the evidence.

    This is a lexical heuristic. It cannot detect a wrong claim that reuses words and numbers
    from the evidence (e.g. attaching a real number to the wrong product).
    """
    evidence = " ".join(c.text for c in contexts)
    evidence_numbers = extract_numbers(evidence)
    evidence_vocab = frozenset(word_tokens(evidence))

    bad_numbers = sorted(extract_numbers(answer) - evidence_numbers)
    answer_terms = word_tokens(answer)
    unsupported = sorted({t for t in answer_terms if not is_covered(t, evidence_vocab)})
    if answer_terms:
        supported_count = sum(1 for t in answer_terms if is_covered(t, evidence_vocab))
        support = supported_count / len(answer_terms)
    else:
        support = 1.0

    ok = not bad_numbers and support >= min_word_support
    return GuardResult(ok, bad_numbers, unsupported, round(support, 3))
