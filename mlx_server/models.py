"""Model cache inspection and binary discovery utilities."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from .config import HF_CACHE

log = logging.getLogger("mlx-gateway")


def find_mlx_server_bin() -> str | None:
    """Find the mlx_lm.server executable on PATH."""
    path = shutil.which("mlx_lm.server")
    if path:
        return path
    candidate = Path.home() / ".local" / "bin" / "mlx_lm.server"
    if candidate.exists():
        return str(candidate)
    return None


def find_embedding_python() -> str | None:
    """Find a Python interpreter that has sentence-transformers installed."""
    candidates = [
        Path.home() / ".local" / "pipx" / "venvs" / "mlx-lm" / "bin" / "python",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    for py in ("python3", "python"):
        result = subprocess.run(
            [py, "-c", "import sentence_transformers"],
            capture_output=True,
        )
        if result.returncode == 0:
            return shutil.which(py)
    return None


def is_model_cached(model_id: str) -> bool:
    cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    return cache_dir.is_dir()


def is_embedding_model(model_id: str) -> bool:
    """Return True if model_id is an embedding model rather than a generative LLM."""
    cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if cache_dir.exists():
        snapshots = cache_dir / "snapshots"
        for snapshot in (snapshots.iterdir() if snapshots.exists() else []):
            if (snapshot / "modules.json").exists():
                return True
            if (snapshot / "sentence_bert_config.json").exists():
                return True
            if (snapshot / "1_Pooling").exists():
                return True
            config_path = snapshot / "config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    arch = cfg.get("architectures", [""])[0]
                    if arch and not any(
                        kw in arch for kw in (
                            "CausalLM", "ConditionalGeneration", "LMHead", "Seq2SeqLM"
                        )
                    ):
                        return True
                except Exception:
                    pass
    lower = model_id.lower()
    return any(kw in lower for kw in (
        "embed", "minilm", "e5-", "bge-", "gte-",
        "nomic-embed", "all-mpnet", "instructor-",
    ))


def get_cached_models() -> list[dict]:
    models = []
    if not HF_CACHE.exists():
        return models
    for d in sorted(HF_CACHE.iterdir()):
        if d.is_dir() and d.name.startswith("models--"):
            model_id = d.name[len("models--"):].replace("--", "/", 1)
            try:
                size_bytes = sum(
                    f.stat().st_size for f in d.rglob("*") if f.is_file()
                )
            except OSError:
                size_bytes = 0
            models.append({
                "id": model_id,
                "size_bytes": size_bytes,
                "modified": d.stat().st_mtime,
            })
    return models
