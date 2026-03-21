# mlx-server

An OS-level CLI for managing local [MLX](https://github.com/ml-explore/mlx) inference server instances on Apple Silicon. Works like `ollama` — start, stop, and inspect models from anywhere in your terminal.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python with `mlx-lm` installed: `pipx install mlx-lm`

## Installation

```bash
sudo cp mlx-server /usr/local/bin/mlx-server
sudo chmod +x /usr/local/bin/mlx-server
```

## Usage

```bash
# Search available models on HuggingFace (mlx-community)
mlx-server search
mlx-server search Qwen3

# List locally downloaded models
mlx-server models

# Start a model (downloads if not cached)
mlx-server start
mlx-server start mlx-community/Qwen3-8B-4bit --port 8089
mlx-server start mlx-community/Qwen3-32B-4bit --port 8088 --max-tokens 8192

# Run multiple models simultaneously on different ports
mlx-server start mlx-community/Qwen3-32B-4bit --port 8088
mlx-server start mlx-community/Qwen3-8B-4bit  --port 8089

# Check running instances
mlx-server status

# View logs
mlx-server logs --port 8088
mlx-server logs mlx-community/Qwen3-8B-4bit -f

# Stop instances
mlx-server stop --port 8089
mlx-server stop mlx-community/Qwen3-32B-4bit
mlx-server stop          # stops all
```

## OpenAI-Compatible API

Each instance exposes an OpenAI-compatible REST API:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-32B-4bit","messages":[{"role":"user","content":"Hello"}]}'
```

Works with any OpenAI client — Python `openai` SDK, LangChain, Spring AI, etc. Just set `base_url` to the local port.

## Defaults

| Option | Default |
|---|---|
| Model | `mlx-community/Qwen3-32B-4bit` |
| Port | `8088` |
| Max tokens | `4096` |
| Thinking (CoT) | disabled |

## Notes

- Instance state tracked in `~/.mlx/` (PID, meta, log files per port)
- Uses `HF_HUB_OFFLINE=1` to load from local cache — no network calls at startup
- Models cached in `~/.cache/huggingface/hub/`
