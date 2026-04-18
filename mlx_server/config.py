"""Global constants and path configuration."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PORT = 11070
BACKEND_BASE_PORT = 18100
MAX_TOKENS_DEFAULT = 16384  # Increased for larger JSON outputs (propositions extraction)
REQUEST_LOG_MAX = 1000  # keep last N requests in memory

# Idle backend eviction: stop backends that haven't served a request for this long.
# Set MLX_IDLE_TIMEOUT=0 to disable eviction entirely.
IDLE_TIMEOUT_SECONDS = int(os.environ.get("MLX_IDLE_TIMEOUT", "1800"))  # 30 minutes

# KV cache: limit how many distinct per-session KV caches mlx_lm.server holds in memory.
# Raise this if you have many concurrent long-lived conversations; lower it to save RAM.
PROMPT_CACHE_SIZE = int(os.environ.get("MLX_PROMPT_CACHE_SIZE", "4"))

MLX_HOME = Path.home() / ".mlx"
HF_CACHE = Path(os.environ.get(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
)) / "hub"
REQUEST_LOG_FILE = MLX_HOME / "requests.jsonl"
BACKEND_PID_DIR = MLX_HOME / "backends"
