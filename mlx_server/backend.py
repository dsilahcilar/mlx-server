"""ModelBackend and BackendManager — lifecycle management for per-model subprocesses."""
from __future__ import annotations

import http.client
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from .config import BACKEND_BASE_PORT, HF_CACHE, MAX_TOKENS_DEFAULT, MLX_HOME
from .models import (
    find_embedding_python,
    find_mlx_server_bin,
    is_embedding_model,
    is_model_cached,
)

log = logging.getLogger("mlx-gateway")


class ModelBackend:
    """Manages a single model server subprocess (LLM or embedding)."""

    def __init__(
        self,
        model_id: str,
        port: int,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        embedding: bool = False,
    ):
        self.model_id = model_id
        self.port = port
        self.max_tokens = max_tokens
        self.embedding = embedding
        self.process: subprocess.Popen | None = None
        self.ready = False
        self.failed = False
        self._loading = threading.Event()
        self._ready_event = threading.Event()
        self.last_used = time.time()
        self.error: str | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the backend process and block until it is ready or has failed."""
        if self.ready:
            return True
        self._loading.set()

        if not is_model_cached(self.model_id):
            self.error = (
                f"Model not found locally. Run: mlx-server pull {self.model_id}"
            )
            self.failed = True
            self._loading.clear()
            return False

        log_path = MLX_HOME / f"backend-{self.port}.log"
        cmd = self._build_command()
        if cmd is None:
            self._loading.clear()
            return False

        log.info(
            "Starting %s%s on internal port %d",
            self.model_id,
            " (embedding)" if self.embedding else "",
            self.port,
        )
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        except Exception as exc:
            self.error = str(exc)
            self.failed = True
            self._loading.clear()
            return False

        return self._wait_for_ready(log_path)

    def stop(self):
        if self.process and self.process.poll() is None:
            log.info("Stopping %s (PID %d)", self.model_id, self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.ready = False
        self._ready_event.clear()

    def touch(self):
        self.last_used = time.time()

    def wait_until_ready(self, timeout: float = 600) -> bool:
        return self._ready_event.wait(timeout=timeout)

    @property
    def is_loading(self) -> bool:
        return self._loading.is_set()

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_command(self) -> list[str] | None:
        if self.embedding:
            embed_python = find_embedding_python()
            if not embed_python:
                self.error = (
                    "sentence-transformers not found. "
                    "Run: python -m pip install sentence-transformers"
                )
                self.failed = True
                return None
            embed_script = Path(__file__).parent / "embedding_server.py"
            return [embed_python, str(embed_script), self.model_id, str(self.port)]

        mlx_bin = find_mlx_server_bin()
        if not mlx_bin:
            self.error = "mlx_lm.server not found. Install: pipx install mlx-lm"
            self.failed = True
            return None
        return [
            mlx_bin,
            "--model", self.model_id,
            "--port", str(self.port),
            "--max-tokens", str(self.max_tokens),
            "--chat-template-args", json.dumps({"enable_thinking": False}),
        ]

    def _wait_for_ready(
        self,
        log_path: Path,
        retries: int = 100,
        interval: float = 3.0,
    ) -> bool:
        """Poll /v1/models until the backend responds 200 or we give up."""
        for _ in range(retries):
            if self.process.poll() is not None:
                self.error = self._read_error_from_log(log_path)
                self.failed = True
                self._loading.clear()
                log.error("Backend %s failed: %s", self.model_id, self.error)
                return False
            try:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", self.port, timeout=2
                )
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    self.ready = True
                    self._loading.clear()
                    self._ready_event.set()
                    self.last_used = time.time()
                    log.info(
                        "✅ %s ready on port %d (PID %d)",
                        self.model_id,
                        self.port,
                        self.process.pid,
                    )
                    return True
            except Exception:
                pass
            time.sleep(interval)

        self.error = "Timeout (5 min) loading model"
        self.failed = True
        self._loading.clear()
        log.error("Backend %s timed out", self.model_id)
        return False

    def _read_error_from_log(self, log_path: Path) -> str:
        try:
            lines = log_path.read_text().strip().splitlines()
            error_line = next(
                (
                    l.strip() for l in reversed(lines)
                    if any(kw in l for kw in (
                        "Error:", "error:", "Exception:", "Traceback"
                    ))
                ),
                None,
            )
            return error_line or next(
                (l.strip() for l in reversed(lines) if l.strip()),
                f"exited with code {self.process.returncode}",
            )
        except Exception:
            return f"exited with code {self.process.returncode}"


class BackendManager:
    """Thread-safe registry of active model backends."""

    def __init__(self):
        self.backends: dict[str, ModelBackend] = {}
        self._next_port = BACKEND_BASE_PORT
        self._lock = threading.Lock()

    def get_or_start(
        self, model_id: str
    ) -> tuple[ModelBackend | None, str | None]:
        """Return ``(backend, None)`` on success or ``(None, error_msg)`` on failure."""
        waiting_on = None

        with self._lock:
            if model_id in self.backends:
                b = self.backends[model_id]
                if b.ready:
                    b.touch()
                    return b, None
                if b.failed:
                    del self.backends[model_id]
                elif b.is_loading:
                    waiting_on = b

            if model_id not in self.backends and waiting_on is None:
                if not is_model_cached(model_id):
                    return None, (
                        f"Model '{model_id}' not found locally. "
                        f"Run: mlx-server pull {model_id}"
                    )
                port = self._next_port
                self._next_port += 1
                embedding = is_embedding_model(model_id)
                self.backends[model_id] = ModelBackend(
                    model_id, port, embedding=embedding
                )

        if waiting_on is not None:
            if waiting_on.wait_until_ready(timeout=600):
                waiting_on.touch()
                return waiting_on, None
            return None, waiting_on.error or "Timeout waiting for model"

        backend = self.backends.get(model_id)
        if backend is None:
            return None, "Internal error"

        if backend.start():
            return backend, None
        return None, backend.error or "Failed to start model"

    def stop_model(self, model_id: str) -> bool:
        with self._lock:
            b = self.backends.pop(model_id, None)
        if b:
            b.stop()
            return True
        return False

    def stop_all(self):
        with self._lock:
            backends = list(self.backends.values())
            self.backends.clear()
        for b in backends:
            b.stop()

    def list_loaded(self) -> list[dict]:
        with self._lock:
            result = []
            for b in self.backends.values():
                info: dict = {
                    "model": b.model_id,
                    "port": b.port,
                    "ready": b.ready,
                    "loading": b.is_loading,
                    "failed": b.failed,
                    "type": "embedding" if b.embedding else "llm",
                }
                if b.failed and b.error:
                    info["error"] = b.error
                if b.process and b.process.poll() is None:
                    info["pid"] = b.process.pid
                if b.ready:
                    cache_dir = HF_CACHE / f"models--{b.model_id.replace('/', '--')}"
                    if cache_dir.exists():
                        try:
                            size = sum(
                                f.stat().st_size
                                for f in cache_dir.rglob("*") if f.is_file()
                            )
                            info["size_gb"] = round(size / (1024 ** 3), 1)
                        except OSError:
                            pass
                    idle = time.time() - b.last_used
                    info["idle"] = (
                        f"{int(idle)}s" if idle < 60 else f"{int(idle / 60)}m"
                    )
                result.append(info)
            return result


manager = BackendManager()
