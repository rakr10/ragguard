from starlette.testclient import TestClient

from ragguard.api import create_app
from ragguard.pipeline import build_pipeline


def client():
    return TestClient(create_app(build_pipeline()))


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


def test_ask_answers():
    r = client().post("/ask", json={"question": "What is the monthly fee for the Premier account?"})
    assert r.status_code == 200 and "£25" in r.json()["answer"]


def test_ask_abstains():
    r = client().post("/ask", json={"question": "What is the interest rate on the Junior ISA?"})
    assert r.status_code == 200 and r.json()["abstained"] is True


def test_ask_validation():
    c = client()
    assert c.post("/ask", json={"question": "  "}).status_code == 422
    assert c.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert c.post("/ask", json={"nope": 1}).status_code == 422
    assert c.post("/ask", content=b"not json").status_code == 400
