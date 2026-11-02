"""ragguard: a small RAG pipeline with abstention, a groundedness guard and an eval harness."""

__version__ = "0.1.0"

from .guard import GuardResult, check_groundedness
from .pipeline import Answer, RagPipeline, build_pipeline

__all__ = ["Answer", "GuardResult", "RagPipeline", "build_pipeline", "check_groundedness"]
