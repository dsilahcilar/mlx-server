"""Ollama-compatible endpoint handlers (/api/chat, /api/generate, /api/tags, etc.)."""
from __future__ import annotations

import http.client
import json
import logging
import re
import time
from datetime import datetime, timezone

from ..aliases import _generate_ollama_alias, resolve_model
from ..backend import manager
from ..config import HF_CACHE
from ..metrics import metrics
from ..models import get_cached_models, is_model_cached

log = logging.getLogger("mlx-gateway")


class OllamaHandlerMixin:
    """Handles ``/api/*`` endpoints using Ollama's wire format."""

    # ── List endpoints ────────────────────────────────────────────────────

    def _ollama_tags(self):
        """GET /api/tags — list locally available models."""
        models = [self._ollama_model_entry(m) for m in get_cached_models()]
        self._json(200, {"models": models})

    def _ollama_ps(self):
        """GET /api/ps — list currently loaded models."""
        models = []
        for b in manager.list_loaded():
            if not b.get("ready"):
                continue
            alias = _generate_ollama_alias(b["model"]) or b["model"]
            size_bytes = int(b.get("size_gb", 0) * 1024 ** 3)
            models.append({
                "name": alias,
                "model": alias,
                "size": size_bytes,
                "digest": f"sha256:{b['model']}",
                "details": {"format": "mlx"},
                "expires_at": datetime.now(timezone.utc).isoformat(),
                "size_vram": size_bytes,
            })
        self._json(200, {"models": models})

    # ── Chat / generate ───────────────────────────────────────────────────

    def _ollama_chat(self):
        """POST /api/chat — Ollama chat, proxied through OpenAI completions."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        ollama_model = body.get("model", "")
        model_id = resolve_model(ollama_model)
        is_stream = body.get("stream", True)

        openai_body: dict = {
            "model": model_id,
            "messages": body.get("messages", []),
            "stream": is_stream,
        }
        if is_stream:
            openai_body["stream_options"] = {"include_usage": True}

        opts = body.get("options", {})
        for oai_key, ollama_key in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stop", "stop"),
        ):
            if ollama_key in opts:
                openai_body[oai_key] = opts[ollama_key]
        if "num_predict" in opts:
            openai_body["max_tokens"] = opts["num_predict"]

        self._ollama_proxy_chat(
            model_id, ollama_model, openai_body, is_stream, "/api/chat"
        )

    def _ollama_generate(self):
        """POST /api/generate — Ollama generate, converted to chat completions."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        ollama_model = body.get("model", "")
        model_id = resolve_model(ollama_model)
        is_stream = body.get("stream", True)

        openai_body: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": body.get("prompt", "")}],
            "stream": is_stream,
        }
        if is_stream:
            openai_body["stream_options"] = {"include_usage": True}

        opts = body.get("options", {})
        if "temperature" in opts:
            openai_body["temperature"] = opts["temperature"]
        if "num_predict" in opts:
            openai_body["max_tokens"] = opts["num_predict"]

        self._ollama_proxy_generate(
            model_id, ollama_model, openai_body, is_stream, "/api/generate"
        )

    def _ollama_show(self):
        """POST /api/show — model info in Ollama format."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        model_name = body.get("model") or body.get("name", "")
        model_id = resolve_model(model_name)

        if not is_model_cached(model_id):
            self._json(404, {"error": f"model '{model_name}' not found"})
            return

        cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
        details = {
            "format": "mlx", "family": "",
            "parameter_size": "", "quantization_level": "",
        }
        model_info = {}

        config_files = list(cache_dir.glob("snapshots/*/config.json"))
        if config_files:
            try:
                cfg = json.loads(config_files[0].read_text())
                details["family"] = cfg.get("model_type", "")
                quant = cfg.get("quantization", {})
                if isinstance(quant, dict):
                    details["quantization_level"] = f"Q{quant.get('bits', '?')}"
                model_info = {
                    "hidden_size": cfg.get("hidden_size"),
                    "num_hidden_layers": cfg.get("num_hidden_layers"),
                    "num_attention_heads": cfg.get("num_attention_heads"),
                    "vocab_size": cfg.get("vocab_size"),
                    "context_length": cfg.get("max_position_embeddings"),
                }
            except (json.JSONDecodeError, OSError):
                pass

        self._json(200, {
            "model_info": model_info,
            "details": details,
            "modified_at": datetime.fromtimestamp(
                cache_dir.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })

    # ── Internal proxy helpers ────────────────────────────────────────────

    def _ollama_proxy_chat(
        self,
        model_id: str,
        ollama_model: str,
        openai_body: dict,
        is_stream: bool,
        endpoint: str,
    ):
        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(404, {"error": err or "model not found"})
            return

        openai_raw = json.dumps(openai_body).encode()
        t_start = time.monotonic()
        error_occurred = False
        prompt_tokens = completion_tokens = cached_tokens = 0

        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", "/v1/chat/completions", body=openai_raw,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(openai_raw)),
                },
            )
            resp = conn.getresponse()

            if resp.status != 200:
                error_occurred = True
                self._forward_error_response(resp)
                conn.close()
                return

            if is_stream:
                prompt_tokens, completion_tokens, cached_tokens = (
                    self._ollama_stream_chat(resp, ollama_model)
                )
            else:
                prompt_tokens, completion_tokens, cached_tokens = (
                    self._ollama_nonstream_chat(resp, ollama_model)
                )
            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            error_occurred = True
        except Exception as exc:
            error_occurred = True
            log.exception("Ollama chat proxy error")
            try:
                self._json(502, {"error": str(exc)})
            except Exception:
                pass
        finally:
            metrics.record(
                model_id=model_id,
                latency_ms=(time.monotonic() - t_start) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                error=error_occurred,
                endpoint=endpoint,
                stream=is_stream,
            )

    def _ollama_proxy_generate(
        self,
        model_id: str,
        ollama_model: str,
        openai_body: dict,
        is_stream: bool,
        endpoint: str,
    ):
        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(404, {"error": err or "model not found"})
            return

        openai_raw = json.dumps(openai_body).encode()
        t_start = time.monotonic()
        error_occurred = False
        prompt_tokens = completion_tokens = cached_tokens = 0

        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", "/v1/chat/completions", body=openai_raw,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(openai_raw)),
                },
            )
            resp = conn.getresponse()

            if resp.status != 200:
                error_occurred = True
                self._forward_error_response(resp)
                conn.close()
                return

            if is_stream:
                prompt_tokens, completion_tokens, cached_tokens = (
                    self._ollama_stream_generate(resp, ollama_model)
                )
            else:
                prompt_tokens, completion_tokens, cached_tokens = (
                    self._ollama_nonstream_generate(resp, ollama_model)
                )
            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            error_occurred = True
        except Exception as exc:
            error_occurred = True
            log.exception("Ollama generate proxy error")
            try:
                self._json(502, {"error": str(exc)})
            except Exception:
                pass
        finally:
            metrics.record(
                model_id=model_id,
                latency_ms=(time.monotonic() - t_start) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                error=error_occurred,
                endpoint=endpoint,
                stream=is_stream,
            )

    # ── Streaming / non-streaming converters ─────────────────────────────

    def _ollama_stream_chat(
        self, resp, model_name: str
    ) -> tuple[int, int, int]:
        """Convert an OpenAI SSE stream to Ollama NDJSON chat stream."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        t_start = time.monotonic()
        eval_count = 0
        prompt_tokens = completion_tokens = cached_tokens = 0

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    oai = json.loads(data_str)
                    usage = oai.get("usage") or {}
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        cached_tokens = (
                            (usage.get("prompt_tokens_details") or {})
                            .get("cached_tokens", 0)
                        )
                    choices = oai.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content", "")
                    finish = choices[0].get("finish_reason")
                    if content:
                        eval_count += 1
                        out = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        })
                        self.wfile.write(out.encode() + b"\n")
                        self.wfile.flush()
                    if finish == "stop":
                        duration_ns = int((time.monotonic() - t_start) * 1e9)
                        final = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "total_duration": duration_ns,
                            "eval_count": eval_count,
                        })
                        self.wfile.write(final.encode() + b"\n")
                        self.wfile.flush()
                except json.JSONDecodeError:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass

        return prompt_tokens, completion_tokens, cached_tokens

    def _ollama_nonstream_chat(
        self, resp, model_name: str
    ) -> tuple[int, int, int]:
        """Convert an OpenAI non-streaming response to Ollama chat format."""
        data = json.loads(resp.read())
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )
        self._json(200, {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 0,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        })
        return prompt_tokens, completion_tokens, cached_tokens

    def _ollama_stream_generate(
        self, resp, model_name: str
    ) -> tuple[int, int, int]:
        """Convert an OpenAI SSE stream to Ollama generate NDJSON stream."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        t_start = time.monotonic()
        prompt_tokens = completion_tokens = cached_tokens = 0

        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    oai = json.loads(data_str)
                    usage = oai.get("usage") or {}
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        cached_tokens = (
                            (usage.get("prompt_tokens_details") or {})
                            .get("cached_tokens", 0)
                        )
                    choices = oai.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content", "")
                    finish = choices[0].get("finish_reason")
                    if content:
                        out = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "response": content,
                            "done": False,
                        })
                        self.wfile.write(out.encode() + b"\n")
                        self.wfile.flush()
                    if finish == "stop":
                        duration_ns = int((time.monotonic() - t_start) * 1e9)
                        final = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "response": "",
                            "done": True,
                            "total_duration": duration_ns,
                        })
                        self.wfile.write(final.encode() + b"\n")
                        self.wfile.flush()
                except json.JSONDecodeError:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass

        return prompt_tokens, completion_tokens, cached_tokens

    def _ollama_nonstream_generate(
        self, resp, model_name: str
    ) -> tuple[int, int, int]:
        """Convert an OpenAI non-streaming response to Ollama generate format."""
        data = json.loads(resp.read())
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )
        self._json(200, {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": content,
            "done": True,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        })
        return prompt_tokens, completion_tokens, cached_tokens

    # ── Shared helpers ────────────────────────────────────────────────────

    def _ollama_model_entry(self, model: dict) -> dict:
        mid = model["id"]
        alias = _generate_ollama_alias(mid) or mid
        ts = datetime.fromtimestamp(model["modified"], tz=timezone.utc).isoformat()

        details = {
            "format": "mlx", "family": "",
            "parameter_size": "", "quantization_level": "",
        }
        cache_dir = HF_CACHE / f"models--{mid.replace('/', '--')}"
        config_files = list(cache_dir.glob("snapshots/*/config.json"))
        if config_files:
            try:
                cfg = json.loads(config_files[0].read_text())
                details["family"] = cfg.get("model_type", "")
                quant = cfg.get("quantization", {})
                if isinstance(quant, dict):
                    details["quantization_level"] = f"Q{quant.get('bits', '?')}"
                size_match = re.search(r'(\d+(?:\.\d+)?[Bb])', mid.split("/")[-1])
                if size_match:
                    details["parameter_size"] = size_match.group(1)
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "name": alias,
            "model": alias,
            "modified_at": ts,
            "size": model["size_bytes"],
            "digest": f"sha256:{mid}",
            "details": details,
        }

    def _forward_error_response(self, resp):
        err_body = resp.read()
        self.send_response(resp.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(err_body)))
        self.end_headers()
        self.wfile.write(err_body)
