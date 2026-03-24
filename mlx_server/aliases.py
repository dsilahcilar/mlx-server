"""Model alias generation and resolution (Ollama-style names → HuggingFace IDs)."""
from __future__ import annotations

import json
import re

from .config import MLX_HOME
from .models import get_cached_models


def _generate_ollama_alias(model_id: str) -> str | None:
    """Generate an Ollama-style alias (e.g. ``qwen3:8b``) from an HF model ID."""
    name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Strip quantization/format suffixes (applied twice to handle stacked suffixes)
    quant_pattern = r'[-_](4bit|8bit|6bit|MLX|mlx|qat|MXFP4|Q8|bf16|fp16|fp32)$'
    name = re.sub(quant_pattern, '', name)
    name = re.sub(quant_pattern, '', name)

    # Strip instruction-tuning / variant suffixes
    name = re.sub(
        r'[-_](Instruct|instruct|Chat|chat|it|IT|Base|base|Preview|preview)$',
        '',
        name,
    )

    size_match = re.search(r'[-_](\d+(?:\.\d+)?[Bb])(?:[-_]|$)', name)
    if not size_match:
        return None

    size = size_match.group(1).lower()
    family = name[:size_match.start()].rstrip('-_').lower()
    return f"{family}:{size}" if family else None


def build_alias_map() -> dict[str, str]:
    """Build a mapping from alias names to canonical HuggingFace model IDs."""
    aliases: dict[str, str] = {}

    alias_file = MLX_HOME / "aliases.json"
    if alias_file.exists():
        try:
            aliases.update(json.loads(alias_file.read_text()))
        except (json.JSONDecodeError, OSError):
            pass

    for model in get_cached_models():
        mid = model["id"]
        aliases[mid] = mid  # full name always resolves to itself

        if "/" in mid:
            short = mid.split("/", 1)[1]
            aliases.setdefault(short, mid)
            aliases.setdefault(short.lower(), mid)

        ollama = _generate_ollama_alias(mid)
        if ollama:
            aliases.setdefault(ollama, mid)

    return aliases


def resolve_model(name: str) -> str:
    """Resolve a model alias to its canonical HuggingFace ID.

    Falls back to returning *name* unchanged so the backend can surface a
    clear error if the model is not cached locally.
    """
    aliases = build_alias_map()
    return aliases.get(name) or aliases.get(name.lower()) or name
