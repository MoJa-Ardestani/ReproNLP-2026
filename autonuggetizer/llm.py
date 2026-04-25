"""
LLM client wrapper — supports OpenAI, Gemini, Qwen, LLaMA, and Claude.

Credentials are loaded from experiments/credentials.yaml (fill once, use forever).
Falls back to environment variables if the file is missing or a field is empty.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


@dataclass(frozen=True)
class ChatCompletionResult:
    """One chat completion: decoded text plus usage and wall time for the successful API call."""

    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_sec: float = 0.0

    def __str__(self) -> str:
        return self.text
BACKOFF_BASE = 2.0

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "qwen": "Qwen/Qwen2.5-32B-Instruct",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "claude": "claude-sonnet-4-20250514",
    "azure": "gpt-4o",
}

_CREDS_PATH = Path(__file__).resolve().parent.parent / "credentials.yaml"
_creds_cache: Optional[dict] = None


def _load_credentials() -> dict:
    global _creds_cache
    if _creds_cache is not None:
        return _creds_cache

    if _CREDS_PATH.exists():
        with open(_CREDS_PATH) as f:
            _creds_cache = yaml.safe_load(f) or {}
        logger.info("Loaded credentials from %s", _CREDS_PATH)
    else:
        _creds_cache = {}
        logger.info("No credentials.yaml found, using environment variables.")

    return _creds_cache


def _get_cred(provider: str, field: str, env_var: str) -> str:
    """Get a credential: credentials.yaml first, then env var."""
    creds = _load_credentials()
    value = (creds.get(provider) or {}).get(field, "")
    if value:
        return value
    return os.environ.get(env_var, "")


def _call_openai_compatible(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
    api_key: str,
    base_url: Optional[str] = None,
) -> ChatCompletionResult:
    from openai import OpenAI, RateLimitError

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            pu = getattr(resp, "usage", None)
            pt = getattr(pu, "prompt_tokens", None) if pu else None
            ct = getattr(pu, "completion_tokens", None) if pu else None
            return ChatCompletionResult(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
            )
        except RateLimitError:
            wait = BACKOFF_BASE ** attempt
            logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs…",
                           attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries (rate limiting).")


def _call_azure(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
) -> ChatCompletionResult:
    from openai import AzureOpenAI, RateLimitError

    client = AzureOpenAI(
        api_key=_get_cred("azure", "api_key", "AZURE_OPENAI_API_KEY"),
        api_version=_get_cred("azure", "api_version", "AZURE_OPENAI_API_VERSION") or "2024-06-01",
        azure_endpoint=_get_cred("azure", "endpoint", "AZURE_OPENAI_ENDPOINT"),
    )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            pu = getattr(resp, "usage", None)
            pt = getattr(pu, "prompt_tokens", None) if pu else None
            ct = getattr(pu, "completion_tokens", None) if pu else None
            return ChatCompletionResult(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
            )
        except RateLimitError:
            wait = BACKOFF_BASE ** attempt
            logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs…",
                           attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries (rate limiting).")


def _call_claude(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> ChatCompletionResult:
    import anthropic

    client = anthropic.Anthropic(api_key=_get_cred("claude", "api_key", "ANTHROPIC_API_KEY"))

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
            )
            latency = time.perf_counter() - t0
            text = (resp.content[0].text if resp.content else "").strip()
            u = getattr(resp, "usage", None)
            pt = getattr(u, "input_tokens", None) if u else None
            ct = getattr(u, "output_tokens", None) if u else None
            return ChatCompletionResult(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
            )
        except anthropic.RateLimitError:
            wait = BACKOFF_BASE ** attempt
            logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs…",
                           attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries (rate limiting).")


def _call_gemini(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> ChatCompletionResult:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_cred("gemini", "api_key", "GEMINI_API_KEY"))

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            latency = time.perf_counter() - t0
            text = (resp.text or "").strip()
            um = getattr(resp, "usage_metadata", None)
            pt = getattr(um, "prompt_token_count", None) if um else None
            ct = getattr(um, "candidates_token_count", None) if um else None
            return ChatCompletionResult(
                text=text,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_sec=latency,
            )
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "resource" in str(e).lower():
                wait = BACKOFF_BASE ** attempt
                logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs…",
                               attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries (rate limiting).")


def chat_completion(
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    provider: str = "openai",
    seed: Optional[int] = 42,
) -> ChatCompletionResult:
    """
    Unified chat-completion across all supported providers.

    Set the model to "default" or None to use the provider's default model.
    Returns ChatCompletionResult (.text, token fields, .latency_sec for the HTTP/API call).
    """
    if not model or model == "default":
        creds = _load_credentials()
        cred_model = (creds.get(provider) or {}).get("model", "")
        model = cred_model if cred_model else DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "openai":
        return _call_openai_compatible(
            system, user, model, temperature, max_tokens, seed,
            api_key=_get_cred("openai", "api_key", "OPENAI_API_KEY"),
        )

    elif provider == "azure":
        return _call_azure(system, user, model, temperature, max_tokens, seed)

    elif provider == "gemini":
        return _call_gemini(system, user, model, temperature, max_tokens)

    elif provider == "qwen":
        return _call_openai_compatible(
            system, user, model, temperature, max_tokens, seed,
            api_key=_get_cred("qwen", "api_key", "QWEN_API_KEY"),
            base_url=_get_cred("qwen", "base_url", "QWEN_BASE_URL"),
        )

    elif provider == "llama":
        return _call_openai_compatible(
            system, user, model, temperature, max_tokens, seed,
            api_key=_get_cred("llama", "api_key", "LLAMA_API_KEY"),
            base_url=_get_cred("llama", "base_url", "LLAMA_BASE_URL"),
        )

    elif provider == "claude":
        return _call_claude(system, user, model, temperature, max_tokens)

    else:
        raise ValueError(f"Unknown provider: {provider}")
