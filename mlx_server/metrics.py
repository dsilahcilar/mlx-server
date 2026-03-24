"""Per-model metrics and request log."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

from .config import MLX_HOME, REQUEST_LOG_FILE, REQUEST_LOG_MAX

log = logging.getLogger("mlx-gateway")


class ModelMetrics:
    """Per-model request counters and token stats."""

    __slots__ = (
        "requests", "errors", "prompt_tokens", "completion_tokens",
        "cached_tokens", "latency_total_ms", "latency_min_ms", "latency_max_ms",
    )

    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.latency_total_ms = 0.0
        self.latency_min_ms = float("inf")
        self.latency_max_ms = 0.0

    def record(
        self,
        latency_ms: float,
        prompt: int,
        completion: int,
        cached: int,
        error: bool,
    ):
        self.requests += 1
        if error:
            self.errors += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        self.latency_total_ms += latency_ms
        self.latency_min_ms = min(self.latency_min_ms, latency_ms)
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_total_ms / self.requests if self.requests else 0.0

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_avg_ms": round(self.avg_latency_ms, 1),
            "latency_min_ms": round(self.latency_min_ms, 1) if self.requests else 0,
            "latency_max_ms": round(self.latency_max_ms, 1),
        }


class MetricsCollector:
    """Thread-safe metrics store and persistent request log."""

    def __init__(self):
        self._lock = threading.Lock()
        self._models: dict[str, ModelMetrics] = {}
        self._recent: deque[dict] = deque(maxlen=REQUEST_LOG_MAX)
        self._start_time = time.time()
        MLX_HOME.mkdir(parents=True, exist_ok=True)
        self._preload()

    def _preload(self):
        """Replay the tail of requests.jsonl into in-memory stats on startup."""
        if not REQUEST_LOG_FILE.exists():
            return
        try:
            lines = REQUEST_LOG_FILE.read_text().splitlines()
            for line in lines[-REQUEST_LOG_MAX:]:
                try:
                    entry = json.loads(line)
                    self._recent.append(entry)
                    model_id = entry.get("model", "")
                    if model_id:
                        if model_id not in self._models:
                            self._models[model_id] = ModelMetrics()
                        self._models[model_id].record(
                            latency_ms=entry.get("latency_ms", 0),
                            prompt=entry.get("prompt_tokens", 0),
                            completion=entry.get("completion_tokens", 0),
                            cached=entry.get("cached_tokens", 0),
                            error=entry.get("error", False),
                        )
                except (json.JSONDecodeError, KeyError):
                    pass
        except OSError:
            pass

    def record(
        self,
        model_id: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        error: bool = False,
        endpoint: str = "",
        stream: bool = False,
    ):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_id,
            "endpoint": endpoint,
            "stream": stream,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "error": error,
        }
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = ModelMetrics()
            self._models[model_id].record(
                latency_ms, prompt_tokens, completion_tokens, cached_tokens, error
            )
            self._recent.append(entry)
        try:
            with open(REQUEST_LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def get_summary(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            total = sum(m.requests for m in self._models.values())
            errors = sum(m.errors for m in self._models.values())
            return {
                "uptime_seconds": int(uptime),
                "total_requests": total,
                "total_errors": errors,
                "models": {k: v.to_dict() for k, v in self._models.items()},
            }

    def get_recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            entries = list(self._recent)
        return entries[-n:]


metrics = MetricsCollector()
