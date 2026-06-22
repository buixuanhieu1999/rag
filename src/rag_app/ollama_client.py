from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .config import AppConfig


class OllamaConfigurationError(RuntimeError):
    pass


def _message_content_raw(response: Any) -> str:
    if isinstance(response, dict):
        return str((response.get("message") or {}).get("content") or "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_content(response: Any) -> str:
    return _message_content_raw(response).strip()


def _build_chat_request(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
    stream: bool,
) -> tuple[Any, dict[str, Any]]:
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
        "stream": stream,
        "keep_alive": config.ollama_keep_alive,
        "options": options,
    }
    if response_format:
        request["format"] = response_format

    return client, request


def chat(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
) -> str:
    client, request = _build_chat_request(
        config,
        messages,
        response_format=response_format,
        options_override=options_override,
        stream=False,
    )
    response = client.chat(**request)
    content = _message_content(response)
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content


def chat_stream(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
) -> Iterator[str]:
    client, request = _build_chat_request(
        config,
        messages,
        response_format=response_format,
        options_override=options_override,
        stream=True,
    )
    saw_content = False
    for chunk in client.chat(**request):
        content = _message_content_raw(chunk)
        if not content:
            continue
        saw_content = True
        yield content
    if not saw_content:
        raise RuntimeError("Ollama returned an empty response.")
