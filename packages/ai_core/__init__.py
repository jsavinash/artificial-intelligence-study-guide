"""ai_core — shared library for the AI tutorial monorepo.

Capacity notes: pure NumPy/scikit-learn only (no torch on this machine).
All LLM access goes through get_llm() which returns a real API client when
OPENAI_API_KEY/ANTHROPIC_API_KEY are present, else a deterministic MockLLM.
"""

__version__ = "1.0.0"

from .llm import get_llm
from .metrics import accuracy, f1, mse, mae, r2, precision, recall, confusion_matrix
from .vectorstore import VectorStore, bm25_scores

__all__ = [
    "get_llm", "accuracy", "precision", "recall", "f1", "confusion_matrix",
    "mse", "mae", "r2", "VectorStore", "bm25_scores", "__version__",
]
