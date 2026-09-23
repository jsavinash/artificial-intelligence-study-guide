"""ai_core — shared library for the AI tutorial monorepo.

Compute policy: the examples are pure NumPy/scikit-learn so they run with zero
extra dependencies and expose the math. PyTorch is an *optional accelerator*
(see accelerators.py) — when installed, `train_mlp(..., backend="torch")` and
the accelerated kernels engage; when absent, everything falls back to NumPy and
still passes. Use `should_accelerate()` to decide which backend a workload
deserves. All LLM access goes through get_llm() which returns a real API client
when OPENAI_API_KEY/ANTHROPIC_API_KEY are present, else a deterministic MockLLM.
"""

__version__ = "1.1.0"

from .accelerators import (  # noqa: F401  (optional torch backend)
    HAS_MPS,
    HAS_TORCH,
    IMPORT_COST,
    backend_info,
    benchmark_backends,
    numpy_attention,
    numpy_conv2d,
    should_accelerate,
    torch_attention,
    torch_conv2d,
    train_mlp,
)
from .llm import get_llm
from .metrics import accuracy, f1, mse, mae, r2, precision, recall, confusion_matrix
from .vectorstore import VectorStore, bm25_scores

__all__ = [
    "get_llm", "accuracy", "precision", "recall", "f1", "confusion_matrix",
    "mse", "mae", "r2", "VectorStore", "bm25_scores", "__version__",
    # accelerated-computing helpers (module 27)
    "HAS_TORCH", "HAS_MPS", "IMPORT_COST", "backend_info", "should_accelerate",
    "numpy_conv2d", "torch_conv2d", "numpy_attention", "torch_attention",
    "train_mlp", "benchmark_backends",
]
