"""
Provider-agnostic LLM client. One interface, swappable adapters, selected by
environment variables so nothing is hardcoded and you can drop in whatever
provider/model you decide on later.

Supported out of the box:
  - openai     (OpenAI / any OpenAI-compatible chat-completions endpoint)
  - anthropic  (Anthropic Messages API)
  - ollama     (local models via http://localhost:11434)

Configuration (env vars):
  LLM_PROVIDER   one of: openai | anthropic | ollama            (default: openai)
  LLM_MODEL      model id, e.g. gpt-4o, claude-sonnet-4, llama3  (provider default otherwise)
  LLM_API_KEY    api key (or provider-specific OPENAI_API_KEY / ANTHROPIC_API_KEY)
  LLM_BASE_URL   override endpoint (e.g. an OpenAI-compatible gateway, or Ollama host)
  LLM_TEMPERATURE  float, default 0.2 (low = more consistent decisions)

The client only needs `requests` (already a dependency). No provider SDKs
required, which keeps install light and avoids version churn.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    timeout: int = 60

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        _load_project_env()
        cfg = cls(
            provider=os.getenv("LLM_PROVIDER", "openai").lower(),
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        # provider-specific key fallbacks
        if not cfg.api_key:
            cfg.api_key = os.getenv(
                {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
                .get(cfg.provider, "LLM_API_KEY")
            )
        return cfg


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class BaseAdapter:
    default_model = ""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    default_model = "gpt-4o"

    def complete(self, system: str, user: str) -> str:
        base = self.cfg.base_url or "https://api.openai.com/v1"
        if not self.cfg.api_key:
            raise LLMError("OpenAI provider needs an API key (LLM_API_KEY or OPENAI_API_KEY).")
        resp = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.cfg.model or self.default_model,
                "temperature": self.cfg.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.cfg.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"OpenAI API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicAdapter(BaseAdapter):
    default_model = "claude-sonnet-4-20250514"

    def complete(self, system: str, user: str) -> str:
        base = self.cfg.base_url or "https://api.anthropic.com/v1"
        if not self.cfg.api_key:
            raise LLMError("Anthropic provider needs an API key (LLM_API_KEY or ANTHROPIC_API_KEY).")
        resp = requests.post(
            f"{base.rstrip('/')}/messages",
            headers={"x-api-key": self.cfg.api_key,
                     "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={
                "model": self.cfg.model or self.default_model,
                "max_tokens": 1500,
                "temperature": self.cfg.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=self.cfg.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class OllamaAdapter(BaseAdapter):
    default_model = "llama3"

    def complete(self, system: str, user: str) -> str:
        base = self.cfg.base_url or "http://localhost:11434"
        resp = requests.post(
            f"{base.rstrip('/')}/api/chat",
            json={
                "model": self.cfg.model or self.default_model,
                "stream": False,
                "options": {"temperature": self.cfg.temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.cfg.timeout,
        )
        if resp.status_code >= 400:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["message"]["content"]


ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
}


class LLMClient:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig.from_env()
        if self.cfg.provider not in ADAPTERS:
            raise LLMError(
                f"Unknown provider {self.cfg.provider!r}. Options: {list(ADAPTERS)}"
            )
        self.adapter = ADAPTERS[self.cfg.provider](self.cfg)

    @property
    def model(self) -> str:
        return self.cfg.model or self.adapter.default_model

    def complete(self, system: str, user: str) -> str:
        return self.adapter.complete(system, user)

    def complete_json(self, system: str, user: str) -> dict:
        """Call the model and parse a JSON object out of the reply, robustly:
        strips markdown fences and grabs the first {...} block if the model
        wrapped its JSON in prose."""
        raw = self.complete(system, user)
        return extract_json(raw)


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM reply."""
    if not text:
        raise LLMError("empty LLM response")
    # strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # fall back to the first balanced-looking object
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if (start != -1 and end > start) else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMError(f"could not parse JSON from LLM reply: {e}. Raw: {text[:300]}")
