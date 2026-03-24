"""Global constants and path configuration."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PORT = 11070
BACKEND_BASE_PORT = 18100
MAX_TOKENS_DEFAULT = 4096
REQUEST_LOG_MAX = 1000  # keep last N requests in memory

MLX_HOME = Path.home() / ".mlx"
HF_CACHE = Path(os.environ.get(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
)) / "hub"
REQUEST_LOG_FILE = MLX_HOME / "requests.jsonl"
