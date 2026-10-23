"""OpenRouter chat client (the only LLM provider used by this project).

Reads OPENROUTER_API_KEY / OPENROUTER_MODEL / OPENROUTER_BASE_URL from the environment or `.env`.
Never logs the key. Bounded retries with backoff on 429/5xx, request timeout, and an optional
on-disk JSONL cache keyed by (model, messages, temperature, replicate) so interrupted runs resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class LLMConfigError(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    pass


@dataclass
class OpenRouterSettings:
    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls, model: str | None = None, env_file: str | os.PathLike | None = ".env"):
        try:
            from dotenv import load_dotenv
            if env_file and Path(env_file).exists():
                load_dotenv(env_file, override=False)
        except ImportError:
            pass
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        model = (model or os.environ.get("OPENROUTER_MODEL", "")).strip()
        base = os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL
        missing = [n for n, v in (("OPENROUTER_API_KEY", key), ("OPENROUTER_MODEL", model)) if not v]
        if missing:
            raise LLMConfigError(f"missing {', '.join(missing)}: fill them in .env (see .env.example)")
        if "api.openai.com" in base or "api.anthropic.com" in base:
            raise LLMConfigError("OPENROUTER_BASE_URL must point to OpenRouter")
        return cls(key, model, base.rstrip("/"))

    def __repr__(self):   # never print the key
        return f"OpenRouterSettings(model={self.model!r}, base_url={self.base_url!r}, api_key=***)"


def env_models(var: str, fallback: str | None = None) -> list[str]:
    """Comma-separated model list from an env var (e.g. OPENROUTER_SCORING_MODELS)."""
    raw = os.environ.get(var, "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    fb = fallback or os.environ.get("OPENROUTER_MODEL", "").strip()
    return [fb] if fb else []


class ChatClient:
    """Interface: chat(messages, model=None, temperature=1.0, replicate=0, max_tokens=None) -> str."""

    calls = 0

    def chat(self, messages, model=None, temperature=1.0, replicate=0, max_tokens=None) -> str:
        raise NotImplementedError


class OpenRouterClient(ChatClient):
    def __init__(self, settings: OpenRouterSettings, timeout=60.0, max_retries=4, backoff=2.0,
                 cache_path: str | os.PathLike | None = ".cache/llm_cache.jsonl"):
        self.s = settings
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["text"]
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def _key(model, messages, temperature, replicate, max_tokens):
        blob = json.dumps([model, messages, temperature, replicate, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, model=None, temperature=1.0, replicate=0, max_tokens=None) -> str:
        import requests

        model = model or self.s.model
        key = self._key(model, messages, temperature, replicate, max_tokens)
        if key in self._cache:
            return self._cache[key]
        body = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self.s.api_key}", "Content-Type": "application/json",
                   "X-Title": "proxyattack-research-harness"}
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._lock:
                    self.calls += 1
                r = requests.post(f"{self.s.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    retry_after = float(r.headers.get("Retry-After", self.backoff * 2 ** attempt))
                    time.sleep(min(retry_after, 60))
                    continue
                if r.status_code >= 400:
                    raise LLMCallError(f"OpenRouter HTTP {r.status_code}: {r.text[:200]}")
                data = r.json()
                text = data["choices"][0]["message"]["content"] or ""
                if self.cache_path:
                    with self._lock:
                        self._cache[key] = text
                        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with self.cache_path.open("a") as f:
                            f.write(json.dumps({"key": key, "model": model, "text": text}) + "\n")
                return text
            except (requests.Timeout, requests.ConnectionError, KeyError, ValueError) as exc:
                last = type(exc).__name__
                time.sleep(self.backoff * 2 ** attempt)
        with self._lock:
            self.failures += 1
        raise LLMCallError(f"OpenRouter call failed after {self.max_retries + 1} attempts ({last})")


class MockClient(ChatClient):
    """Offline stand-in for tests and dry runs: `responder(messages, model, temperature, replicate)`."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = 0
        self.failures = 0

    def chat(self, messages, model=None, temperature=1.0, replicate=0, max_tokens=None) -> str:
        self.calls += 1
        return self.responder(messages, model, temperature, replicate)
