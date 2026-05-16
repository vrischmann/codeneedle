"""Minimal OpenAI-compatible chat-completions client."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Usage:
    """Token usage and server-side timing data returned by the API."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # llama-server returns detailed timings in a top-level `timings` field.
    timings: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_per_second(self) -> float | None:
        """Prompt processing throughput (tokens/s), if reported by the server."""
        v = self.timings.get("prompt_per_second")
        return float(v) if v is not None else None

    @property
    def predicted_per_second(self) -> float | None:
        """Generation throughput (tokens/s), if reported by the server."""
        v = self.timings.get("predicted_per_second")
        return float(v) if v is not None else None

    @property
    def cache_n(self) -> int | None:
        """Number of prompt tokens reused from KV cache (llama-server only)."""
        v = self.timings.get("cache_n")
        return int(v) if v is not None else None


@dataclass
class ChatResponse:
    """Response from chat_complete: content + usage metadata."""
    content: str
    usage: Usage = field(default_factory=Usage)


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 6000  # generous — reasoning models can spend most of this on CoT
    timeout: float = 600.0
    # Reasoning-suppression techniques. Each works on a different subset of
    # reasoning models — see configs/CONFIG_README.md for the matrix. Combine
    # freely; harmless flags are just ignored by models that don't recognize them.
    reasoning_effort: str | None = None    # sends `reasoning_effort: <value>` in request body (e.g. "none", "low")
    prefill_no_think: bool = False         # appends an assistant message containing `<think>\n</think>\n\n`
    stop: list[str] | None = None          # stop sequences sent to the server; useful for models that parrot the prompt back (Gemma 4)
    use_max_completion_tokens: bool = False  # send `max_completion_tokens` instead of `max_tokens` (required by OpenAI GPT-5 family)
    extra_body: dict[str, Any] = field(default_factory=dict)  # extra fields merged into the request body (e.g. enable_thinking for vLLM)


@dataclass
class DebugCapture:
    """Full request/response capture for debug logging."""
    request_url: str
    request_payload: dict
    response_status: int
    response_data: dict


def chat_complete(cfg: ClientConfig, system: str | None, user: str, *, debug: bool = False) -> tuple[ChatResponse, DebugCapture | None]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if cfg.prefill_no_think:
        # Pre-filling the assistant turn with an empty think block is the only
        # technique that reliably skips CoT on Qwen3.5/3.6 — the model sees
        # <think/> and continues from there with the actual answer.
        messages.append({"role": "assistant", "content": "\n\n"})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": False,
    }
    if cfg.use_max_completion_tokens:
        payload["max_completion_tokens"] = cfg.max_tokens
    else:
        payload["max_tokens"] = cfg.max_tokens
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=cfg.timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()

    debug_cap: DebugCapture | None = None
    if debug:
        debug_cap = DebugCapture(
            request_url=url,
            request_payload=payload,
            response_status=r.status_code,
            response_data=data,
        )

    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        # vLLM reasoning models (Qwen3.x, DeepSeek-R1, …) put CoT tokens in
        # a separate `reasoning` field and set `content` to null when the
        # entire budget is spent on thinking. Fall back to reasoning so the
        # caller gets *something* rather than None.
        content = msg.get("reasoning") or ""

    # Extract usage data (both servers return this).
    raw_usage = data.get("usage") or {}
    prompt_tokens = raw_usage.get("prompt_tokens", 0) or 0
    completion_tokens = raw_usage.get("completion_tokens", 0) or 0

    # llama-server returns a top-level `timings` object with detailed perf data.
    timings: dict[str, Any] = data.get("timings") or {}

    return ChatResponse(
        content=content,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timings=timings,
        ),
    ), debug_cap
