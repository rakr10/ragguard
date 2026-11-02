"""HTTP API (Starlette). Run with: uvicorn ragguard.api:app"""

from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .pipeline import RagPipeline, build_pipeline

MAX_QUESTION_CHARS = 500


def create_app(pipeline: RagPipeline | None = None) -> Starlette:
    state: dict[str, RagPipeline] = {}

    def get_pipeline() -> RagPipeline:
        if "p" not in state:
            state["p"] = pipeline or build_pipeline(
                os.environ.get("RAGGUARD_CORPUS_DIR"),
                backend=os.environ.get("RAGGUARD_BACKEND", "extractive"),
            )
        return state["p"]

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def ask(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "body must be valid JSON"}, status_code=400)
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip():
            return JSONResponse({"error": "'question' must be a non-empty string"}, status_code=422)
        if len(question) > MAX_QUESTION_CHARS:
            return JSONResponse(
                {"error": f"'question' exceeds {MAX_QUESTION_CHARS} characters"}, status_code=422
            )
        return JSONResponse(get_pipeline().ask(question.strip()).to_dict())

    return Starlette(routes=[Route("/health", health), Route("/ask", ask, methods=["POST"])])


app = create_app()
