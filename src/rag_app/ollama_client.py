from __future__ import annotations

from typing import Any

from .config import AppConfig


class OllamaConfigurationError(RuntimeError):
    pass


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        return ((response.get("message") or {}).get("content") or "").strip()
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(getattr(message, "content", "") or "").strip()


def chat(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
) -> str:
    from ollama import Client

    headers = {}
    if config.uses_direct_ollama_cloud:
        if not config.ollama_api_key:
            raise OllamaConfigurationError(
                "OLLAMA_API_KEY is required when OLLAMA_HOST is https://ollama.com."
            )
        headers["Authorization"] = f"Bearer {config.ollama_api_key}"

    client = Client(host=config.ollama_host, headers=headers)
    options = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_ctx": config.num_ctx,
    }
    if options_override:
        options.update(options_override)

    request: dict[str, Any] = {
        "model": config.ollama_model,
        "messages": messages,
        "stream": False,
        "keep_alive": config.ollama_keep_alive,
        "options": options,
    }
    if response_format:
        request["format"] = response_format

    response = client.chat(**request)
    content = _message_content(response)
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content
