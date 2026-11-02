# ragguard

A small retrieval-augmented question-answering service that **knows when not to answer**, with an evaluation harness that measures it.

The corpus is eight short documents for a *fictional* bank ("Northbridge Bank"). No real data is used.

## What it does

1. **Retrieve**: BM25 (written from scratch, no dependencies) over paragraph chunks.
2. **Scope gate**: if any content term in the question does not exist anywhere in the corpus (e.g. "Junior ISA", "CHAPS"), abstain instead of answering from a loosely related passage.
3. **Generate**: pluggable backend: `extractive` (verbatim sentences, deterministic), `mock` (an always-answering stand-in that sometimes distorts numbers, used to exercise the guard) or `anthropic` (Messages API; see limitations).
4. **Groundedness guard**: reject an answer if it contains a number not present in the retrieved evidence, or if fewer than 80% of its content words appear in the evidence.
5. **Evaluate**: accuracy, hallucination rate, informativeness, correct/false abstention, with bootstrap 95% confidence intervals (2,000 resamples, fixed seed).

## Run it

```bash
pip install -e ".[dev]"
pytest -q
python -m ragguard ask "What is the monthly fee for the Premier account?"
python -m ragguard eval                    # extractive backend, 3 configurations
python -m ragguard eval --backend mock
uvicorn ragguard.api:app                   # POST /ask {"question": "..."}, GET /health
```

## Results

Definitions (see `src/ragguard/evaluate.py`): **hallucination** = answering a question the corpus cannot answer, *or* giving an answer containing a number that is not in the correct source document. **Informativeness** = answerable question answered with all required facts. Intervals are wide because the sets are small.

### Development set (40 questions: 24 answerable, 8 unanswerable, 8 near-miss)

The gate threshold was first set to 0.75 and then changed to 1.0 after I saw that near-miss questions slipped through. This set is therefore **not** a clean test of the gate.

Extractive backend:

| Metric | baseline (no gate, no guard) | gate only | gate + guard |
|---|---|---|---|
| accuracy_overall | 55% (40%–70%) | 85% (72%–95%) | 85% (72%–95%) |
| informativeness_answerable | 96% (88%–100%) | 92% (79%–100%) | 92% (79%–100%) |
| hallucination_rate_unanswerable | 81% (62%–100%) | 0% (0%–0%) | 0% (0%–0%) |
| hallucination_rate_answerable | 17% (4%–33%) | 17% (4%–33%) | 17% (4%–33%) |
| abstention_correct_unanswerable | 19% (0%–38%) | 100% (100%–100%) | 100% (100%–100%) |
| false_abstention_answerable | 0% (0%–0%) | 4% (0%–12%) | 4% (0%–12%) |

Mock (overconfident) backend. This shows the guard doing its job against a simulated model; it says nothing about real LLMs:

| Metric | baseline (no gate, no guard) | gate only | gate + guard |
|---|---|---|---|
| accuracy_overall | 32% (18%–48%) | 72% (57%–85%) | 72% (57%–85%) |
| informativeness_answerable | 79% (62%–92%) | 75% (58%–92%) | 67% (46%–83%) |
| hallucination_rate_unanswerable | 100% (100%–100%) | 0% (0%–0%) | 0% (0%–0%) |
| hallucination_rate_answerable | 42% (21%–62%) | 38% (21%–58%) | 12% (0%–29%) |
| abstention_correct_unanswerable | 0% (0%–0%) | 100% (100%–100%) | 100% (100%–100%) |
| false_abstention_answerable | 0% (0%–0%) | 4% (0%–12%) | 29% (12%–50%) |

### Held-out set (16 questions written after the threshold was fixed)

Extractive backend:

| Metric | baseline (no gate, no guard) | gate only | gate + guard |
|---|---|---|---|
| accuracy_overall | 44% (19%–69%) | 69% (44%–88%) | 69% (44%–88%) |
| informativeness_answerable | 88% (62%–100%) | 75% (38%–100%) | 75% (38%–100%) |
| hallucination_rate_unanswerable | 100% (100%–100%) | 38% (0%–75%) | 38% (0%–75%) |
| hallucination_rate_answerable | 12% (0%–38%) | 0% (0%–0%) | 0% (0%–0%) |
| abstention_correct_unanswerable | 0% (0%–0%) | 62% (25%–100%) | 62% (25%–100%) |
| false_abstention_answerable | 0% (0%–0%) | 25% (0%–62%) | 25% (0%–62%) |

Mock backend:

| Metric | baseline (no gate, no guard) | gate only | gate + guard |
|---|---|---|---|
| accuracy_overall | 44% (19%–69%) | 69% (44%–88%) | 81% (62%–100%) |
| informativeness_answerable | 88% (62%–100%) | 75% (38%–100%) | 75% (38%–100%) |
| hallucination_rate_unanswerable | 100% (100%–100%) | 38% (0%–75%) | 12% (0%–38%) |
| hallucination_rate_answerable | 12% (0%–38%) | 0% (0%–0%) | 0% (0%–0%) |
| abstention_correct_unanswerable | 0% (0%–0%) | 62% (25%–100%) | 88% (62%–100%) |
| false_abstention_answerable | 0% (0%–0%) | 25% (0%–62%) | 25% (0%–62%) |

The held-out numbers are worse than the development numbers, and that is the more honest picture. With 16 questions the intervals are very wide.

## Known limitations

- **The gate is lexical.** It refuses answerable questions phrased with words the corpus never uses (false abstention 4% on dev, 25% on held-out) and misses unanswerable questions built only from known words (e.g. "How do I close my account?" returned an unrelated sentence).
- **The guard is lexical.** It catches invented numbers and off-topic wording, but cannot catch a true number attached to the wrong product. It also rejects some correct paraphrased answers (see false abstention for the mock backend).
- **Retrieval errors pass through.** Some wrong answers come from retrieving the wrong paragraph (e.g. the minimum deposit for the wrong savings product).
- **The mock backend is synthetic.** Its hallucination rate is a parameter I chose, not a measurement.
- **Real LLM backends are untested.** The `anthropic` backend was written but never run against the live API, and no real model was evaluated.
- **Small, self-written evaluation sets.** No claims about general performance follow from them.
- **Docker image and CI were written but not executed** in the environment where this was built (no Docker daemon, `ruff` unavailable). `tests/lint_check.py` is a minimal stand-in lint.
- **Not deployed.** See `deploy/aws.md` for planned steps.

## Possible next steps

Embedding-based retrieval with a reranker; an entailment model for the guard; a larger, independently written evaluation set; run the Anthropic backend and report real numbers; deploy and add the URL.

## Layout

```
src/ragguard/   text, corpus, retrieval, llm, guard, pipeline, evaluate, cli, api
data/           corpus/ (8 docs), eval_set.jsonl (dev), heldout_set.jsonl
results/        full JSON reports
tests/          pytest suite + lint_check.py
deploy/aws.md   deployment notes
```
