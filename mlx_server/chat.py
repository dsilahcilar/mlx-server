"""Interactive multi-turn chat REPL."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def chat_repl(base_url: str, model: str):
    """Run an interactive multi-turn chat session with streaming output."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Respond in clear, natural language."},
    ]
    print(f"Running {model}. Send a message or type /bye to exit.\n")

    while True:
        try:
            prompt = input("\033[1;32m>>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue
        if prompt.strip() in ("/bye", "/exit", "/quit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )

        full_response = ""
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            for raw_line in resp:
                line = raw_line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        choices = chunk.get("choices") or []
                        if choices:
                            content = (
                                choices[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_response += content
                    except json.JSONDecodeError:
                        pass
            print("\n")
        except KeyboardInterrupt:
            print("\n")
            messages.pop()
            continue
        except urllib.error.HTTPError as e:
            print(f"\nError ({e.code}): {e.read().decode()}\n")
            messages.pop()
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": full_response})
