"""
Players Data — IBM CIC Germany
Ollama Client  (qwen2.5:14b)

Thread-safe, production-grade HTTP client for local Ollama inference.

Features
────────
  • Sync and async interfaces (OllamaClient / AsyncOllamaClient)
  • Per-call timeout with hard-abort via httpx
  • Automatic retry with exponential back-off (sync only; async uses asyncio.wait_for)
  • LRU response cache keyed on (prompt_hash, model, max_tokens)
  • Health-check / model-availability probe at startup
  • Structured JSON output mode (json_mode=True) with strict parse + fallback
  • Streaming support (iter_tokens generator)
  • All calls log latency at DEBUG; slow calls warn at ≥ SLA threshold

Environment variables
─────────────────────
  OLLAMA_BASE_URL   default: http://localhost:11434
  OLLAMA_TIMEOUT_S  default: 30          (hard per-request timeout)
  OLLAMA_NLG_TIMEOUT_S  default: 2       (tight SLA for serve-mode NLG)
  OLLAMA_RETRIES    default: 2
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Generator, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
_BASE_URL         = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
_TIMEOUT_S        = float(os.getenv("OLLAMA_TIMEOUT_S",     "30"))
_NLG_TIMEOUT_S    = float(os.getenv("OLLAMA_NLG_TIMEOUT_S", "2"))
_RETRIES          = int(os.getenv("OLLAMA_RETRIES",          "2"))
_DEFAULT_MODEL    = "qwen2.5:14b"
_SLOW_CALL_MS     = 500   # warn if a call takes longer than this


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OllamaResponse:
    text: str
    model: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ms: float = 0.0
    cached: bool = False

    def as_json(self) -> Optional[Dict[str, Any]]:
        """Parse .text as JSON; return None if it fails."""
        try:
            return json.loads(self.text.strip())
        except json.JSONDecodeError:
            # Strip markdown fences that some models add
            stripped = self.text.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                inner = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                )
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
        logger.warning("OllamaResponse.as_json(): could not parse: %.120s", self.text)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal LRU cache (thread-safe via functools.lru_cache on immutable key)
# ─────────────────────────────────────────────────────────────────────────────
def _cache_key(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    raw = f"{model}|{max_tokens}|{temperature:.3f}|{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


class _ResponseCache:
    """Simple bounded dict cache with thread lock."""

    def __init__(self, maxsize: int = 256):
        self._store: Dict[str, OllamaResponse] = {}
        self._order: List[str] = []
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[OllamaResponse]:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: OllamaResponse) -> None:
        with self._lock:
            if key in self._store:
                return
            if len(self._order) >= self._maxsize:
                evict = self._order.pop(0)
                self._store.pop(evict, None)
            self._store[key] = value
            self._order.append(key)


_GLOBAL_CACHE = _ResponseCache(maxsize=512)


# ─────────────────────────────────────────────────────────────────────────────
# Sync Client
# ─────────────────────────────────────────────────────────────────────────────
class OllamaClient:
    """
    Thread-safe synchronous Ollama client.

    Instantiate once (module level or as a singleton) and reuse across threads.
    Uses a persistent httpx.Client for connection pooling.
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        timeout_s: float = _TIMEOUT_S,
        max_retries: int = _RETRIES,
        cache: bool = True,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "httpx is required for OllamaClient. "
                "Install with: pip install httpx"
            )
        self._httpx = httpx
        self.base_url      = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s     = timeout_s
        self.max_retries   = max_retries
        self._cache_enabled = cache
        self._cache = _GLOBAL_CACHE
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._lock = threading.Lock()  # for client re-creation only

    # ── Health check ──────────────────────────────────────────────────────────
    def is_available(self, model: Optional[str] = None) -> bool:
        """Return True if Ollama is reachable and the requested model is loaded."""
        try:
            r = self._client.get("/api/tags", timeout=3.0)
            if r.status_code != 200:
                return False
            if model:
                tags = r.json().get("models", [])
                names = {m["name"].split(":")[0] for m in tags}
                names |= {m["name"] for m in tags}
                if model.split(":")[0] not in names and model not in names:
                    logger.warning("Model '%s' not found in Ollama; available: %s", model, names)
                    return False
            return True
        except Exception as exc:
            logger.debug("Ollama health-check failed: %s", exc)
            return False

    # ── Core generate ─────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        json_mode: bool = False,
        timeout_s: Optional[float] = None,
        use_cache: bool = True,
    ) -> OllamaResponse:
        """
        Generate a completion.  Retries up to self.max_retries times on
        transient errors (connection reset, timeout, 5xx).
        """
        model = model or self.default_model
        t_out = timeout_s or self.timeout_s

        # Cache lookup
        if self._cache_enabled and use_cache:
            full_prompt = (system or "") + "\n" + prompt
            ck = _cache_key(full_prompt, model, max_tokens, temperature)
            cached = self._cache.get(ck)
            if cached is not None:
                logger.debug("Cache hit for model=%s prompt_len=%d", model, len(prompt))
                return cached

        payload: Dict[str, Any] = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict":  max_tokens,
                "temperature":  temperature,
                "top_p":        top_p,
                "num_ctx":      4096,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.post(
                    "/api/generate",
                    json=payload,
                    timeout=self._httpx.Timeout(t_out, connect=5.0),
                )
                resp.raise_for_status()
                data = resp.json()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                result = OllamaResponse(
                    text=data.get("response", ""),
                    model=model,
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    eval_count=data.get("eval_count", 0),
                    total_duration_ms=elapsed_ms,
                )

                if elapsed_ms > _SLOW_CALL_MS:
                    logger.warning(
                        "Slow Ollama call: model=%s  %.0f ms  tokens=%d",
                        model, elapsed_ms, result.eval_count,
                    )
                else:
                    logger.debug(
                        "Ollama generate: model=%s  %.0f ms  tokens=%d",
                        model, elapsed_ms, result.eval_count,
                    )

                if self._cache_enabled and use_cache:
                    self._cache.set(ck, result)  # type: ignore[possibly-undefined]

                return result

            except (self._httpx.TimeoutException, self._httpx.NetworkError) as exc:
                last_exc = exc
                wait = 0.2 * (2 ** attempt)
                logger.warning(
                    "Ollama transient error (attempt %d/%d): %s — retrying in %.1f s",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
            except Exception as exc:
                raise RuntimeError(f"Ollama generate failed: {exc}") from exc

        raise RuntimeError(f"Ollama unreachable after {self.max_retries + 1} attempts: {last_exc}")

    # ── Streaming generator ───────────────────────────────────────────────────
    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """Yield tokens as they arrive (Server-Sent Events from /api/generate)."""
        model = model or self.default_model
        payload: Dict[str, Any] = {
            "model":  model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx":     4096,
            },
        }
        if system:
            payload["system"] = system

        with self._client.stream("POST", "/api/generate", json=payload,
                                 timeout=self._httpx.Timeout(self.timeout_s)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    # ── Chat interface ────────────────────────────────────────────────────────
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
        timeout_s: Optional[float] = None,
    ) -> OllamaResponse:
        """OpenAI-compatible chat interface (/api/chat)."""
        model   = model or self.default_model
        t_out   = timeout_s or self.timeout_s
        payload: Dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx":     4096,
            },
        }
        if json_mode:
            payload["format"] = "json"

        t0   = time.perf_counter()
        resp = self._client.post(
            "/api/chat",
            json=payload,
            timeout=self._httpx.Timeout(t_out, connect=5.0),
        )
        resp.raise_for_status()
        data       = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = data.get("message", {}).get("content", "")
        return OllamaResponse(
            text=text,
            model=model,
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            total_duration_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Async Client
# ─────────────────────────────────────────────────────────────────────────────
class AsyncOllamaClient:
    """
    Async Ollama client for use inside asyncio event loops.
    Safe to share across coroutines; uses a single httpx.AsyncClient.
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        default_model: str = _DEFAULT_MODEL,
        timeout_s: float = _TIMEOUT_S,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required. Install with: pip install httpx")
        self._httpx        = httpx
        self.base_url      = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout_s     = timeout_s
        self._client: Optional[Any] = None   # lazily created inside the loop

    async def _get_client(self):
        if self._client is None:
            self._client = self._httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._httpx.Timeout(self.timeout_s, connect=5.0),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
        json_mode: bool = False,
        timeout_s: Optional[float] = None,
    ) -> OllamaResponse:
        client  = await self._get_client()
        model   = model or self.default_model
        t_out   = timeout_s or self.timeout_s
        payload: Dict[str, Any] = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx":     4096,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        t0   = time.perf_counter()
        resp = await asyncio.wait_for(
            client.post("/api/generate", json=payload),
            timeout=t_out,
        )
        resp.raise_for_status()
        data       = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return OllamaResponse(
            text=data.get("response", ""),
            model=model,
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            total_duration_ms=elapsed_ms,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy init)
# ─────────────────────────────────────────────────────────────────────────────
_singleton_lock   = threading.Lock()
_singleton_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Return the module-level singleton OllamaClient (created once, thread-safe)."""
    global _singleton_client
    if _singleton_client is None:
        with _singleton_lock:
            if _singleton_client is None:
                _singleton_client = OllamaClient()
                logger.info(
                    "OllamaClient singleton created: url=%s model=%s",
                    _BASE_URL, _DEFAULT_MODEL,
                )
    return _singleton_client