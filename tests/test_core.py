from ragguard import build_pipeline, check_groundedness
from ragguard.corpus import Chunk, load_corpus
from ragguard.evaluate import bootstrap_ci, evaluate, load_eval_set
from ragguard.llm import NO_ANSWER, OverconfidentMockBackend
from ragguard.retrieval import BM25Retriever
from ragguard.text import contains_fact, extract_numbers, stem


def chunk(text, cid="d#1"):
    return Chunk(cid, "d", "", text)


def test_corpus_loads_and_chunks():
    chunks = load_corpus("data/corpus")
    assert len(chunks) > 30
    assert all(c.text and not c.text.startswith("#") for c in chunks)


def test_bm25_finds_relevant_chunk():
    r = BM25Retriever(load_corpus("data/corpus"))
    top = r.search("What is the monthly fee for the Premier account?", k=3)
    assert top and top[0].chunk.doc == "01_current_account_fees"


def test_bm25_no_match_returns_empty():
    r = BM25Retriever([chunk("alpha beta gamma")])
    assert r.search("zzzz qqqq") == []


def test_text_helpers():
    assert extract_numbers("£20,000 at 39.9%") == {"20000", "39.9"}
    assert contains_fact("The fee is £50.", "£50")
    assert not contains_fact("The fee is £50.", "£5")
    assert not contains_fact("rate is 13%", "3%")
    assert stem("charges") == "charg" and stem("class") == "class"


def test_guard_flags_invented_number():
    ctx = [chunk("The fee is £5 per day.")]
    assert check_groundedness("The fee is £5 per day.", ctx).supported
    bad = check_groundedness("The fee is £7 per day.", ctx)
    assert not bad.supported and bad.unsupported_numbers == ["7"]


def test_guard_flags_unsupported_words():
    ctx = [chunk("The fee is £5 per day.")]
    res = check_groundedness("Yes, premium helicopter transfers are complimentary.", ctx)
    assert not res.supported and res.word_support < 0.8


def test_gate_abstains_on_unknown_term():
    p = build_pipeline()
    a = p.ask("What is the interest rate on the Junior ISA?")
    assert a.abstained and a.reason == "out_of_scope" and a.answer == NO_ANSWER


def test_answers_supported_question():
    a = build_pipeline().ask("What is the monthly fee for the Premier account?")
    assert not a.abstained and "£25" in a.answer and a.sources


def test_guard_blocks_mock_hallucination():
    unguarded = build_pipeline(backend=OverconfidentMockBackend(distort_rate=1.0), use_guard=False)
    guarded = build_pipeline(backend=OverconfidentMockBackend(distort_rate=1.0))
    q = "What is the monthly fee for the Premier account?"
    assert "£25" not in unguarded.ask(q).answer  # distorted
    g = guarded.ask(q)
    assert g.abstained and g.reason == "unsupported_answer"


def test_bootstrap_ci_properties():
    mean, lo, hi = bootstrap_ci([True] * 8 + [False] * 2)
    assert mean == 0.8 and lo <= mean <= hi
    assert bootstrap_ci([True] * 5) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([True, False], seed=1) == bootstrap_ci([True, False], seed=1)


def test_eval_gate_improves_over_baseline():
    items = load_eval_set()
    base = evaluate(build_pipeline(use_gate=False, use_guard=False), items)
    gated = evaluate(build_pipeline(), items)
    key = "hallucination_rate_unanswerable"
    assert gated["metrics"][key]["value"] < base["metrics"][key]["value"]


def test_eval_set_is_well_formed():
    items = load_eval_set()
    assert len(items) == 40
    assert len({i["id"] for i in items}) == 40
    for i in items:
        assert (i["category"] == "answerable") == bool(i["key_facts"])
