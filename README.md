# mlx-server

An Ollama-style CLI for running LLMs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

**One URL, many models.** A single gateway routes OpenAI-compatible API requests to the right model backend — just like calling `api.openai.com`, but everything runs on your Mac.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- `mlx-lm`: `pipx install mlx-lm`

## Installation

```bash
git clone <repo-url>
cd mlx-server
./install.sh
```

## Quick Start

```bash
# Start the gateway server
mlx-server serve

# Pull and run a model (starts interactive chat)
mlx-server run mlx-community/Qwen3-32B-4bit

# From another terminal — use the API
curl http://127.0.0.1:11070/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-32B-4bit","messages":[{"role":"user","content":"Hello"}]}'
```

## Commands

| Command | Description |
|---|---|
| `serve [--port N]` | Start the gateway server (default port: 11070) |
| `shutdown` | Stop the gateway and all loaded models |
| `run <model>` | Load a model and start interactive chat |
| `stop <model>` | Unload a model from memory |
| `pull <model>` | Download a model from HuggingFace |
| `rm <model>` | Delete a downloaded model |
| `show <model>` | Show model details (architecture, size, quantization) |
| `ps` | List currently loaded models |
| `list` | List locally downloaded models |
| `search [query]` | Search mlx-community models on HuggingFace |
| `logs [-f]` | View gateway logs |

## Multi-Model Support

Load multiple models simultaneously — the gateway routes by the `model` field:

```bash
mlx-server serve

# Load both models (each gets its own backend process)
mlx-server run mlx-community/Qwen3-32B-4bit    # loads on demand
mlx-server run mlx-community/Qwen3-8B-4bit     # loads alongside

# API calls routed by model name — same URL
curl http://127.0.0.1:11070/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3-32B-4bit","messages":[...]}'

curl http://127.0.0.1:11070/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3-8B-4bit","messages":[...]}'

# Check what's loaded
mlx-server ps

# Unload one
mlx-server stop mlx-community/Qwen3-8B-4bit
```

Models are also loaded **on demand**: just send an API request with any locally available model and the gateway starts the backend automatically.

## OpenAI-Compatible API

The gateway implements the OpenAI API standard:

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings |
| `GET /v1/models` | List available models |

Works with any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11070/v1", api_key="mlx")
response = client.chat.completions.create(
    model="mlx-community/Qwen3-32B-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```typescript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:11070/v1", apiKey: "mlx" });
```

Also works with LangChain, Spring AI, Cursor, Continue.dev, and any tool that supports custom OpenAI base URLs.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLX_PORT` | `11070` | Gateway port |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache location |

## Architecture

```
┌─────────────┐
│   Clients    │  (curl, Python SDK, Spring AI, etc.)
└──────┬───────┘
       │ :11070
┌──────▼───────┐
│   Gateway    │  Single entry point, routes by "model" field
│  (Python)    │
└──┬───────┬───┘
   │       │
┌──▼──┐ ┌──▼──┐
│ :181│ │ :181│  Internal mlx_lm.server backends
│ 00  │ │ 01  │  (one per model, auto-managed)
│Qwen │ │Qwen │
│ 32B │ │  8B │
└─────┘ └─────┘
```

## Notes

- State tracked in `~/.mlx/` (gateway PID, backend logs)
- Models cached in `~/.cache/huggingface/hub/`
- Backends start with `HF_HUB_OFFLINE=1` for instant loading from cache
- Chain-of-thought (thinking) disabled by default for cleaner output
- Max tokens default: 4096 per response
