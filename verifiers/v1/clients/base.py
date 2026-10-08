"""Shared client plumbing: transport defaults and URL/key utilities."""

import re

import httpx
from openai import AsyncOpenAI

from verifiers.v1.configs.client import BaseClientConfig, resolve_api_key

# No read timeout: agentic completions are slow and the rollout timeout is the real
# backstop. The connect bound stays so an unreachable endpoint still fails, after the
# connection retries below.
DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=None, pool=None)
CONNECT_ATTEMPTS = 6
"""Attempts to establish a connection before a request fails. A provider busy with a deep
queue (for example a local vLLM server serving many concurrent rollouts) can be slow to
accept new connections; one slow accept used to fail the call and end its rollout."""
CONNECT_BACKOFF_SECONDS = 0.5
"""First wait between connection attempts; it doubles after each failed attempt."""
# Idle connections expire before common providers close them (uvicorn, which serves vLLM,
# closes idle keep-alive connections after 5 s). Reusing a connection the server is closing
# fails the request with "server disconnected" after it was sent, which cannot be retried.
KEEPALIVE_EXPIRY_SECONDS = 2.0
DEFAULT_LIMITS = httpx.Limits(
    max_connections=1000,
    max_keepalive_connections=100,
    keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
)
MAX_RETRIES = 0
"""No client-side retries: failures surface to the harness SDK and the trace instead of
being silently reattempted."""

# An API version path segment (`v1`, `v2`, ...) — the only kind `join_url` dedups.
VERSION_SEGMENT = re.compile(r"v\d+")


def build_async_openai(config: BaseClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        default_headers=config.headers or None,
        timeout=DEFAULT_TIMEOUT,
        max_retries=MAX_RETRIES,
        http_client=httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=DEFAULT_LIMITS),
    )


def join_url(base_url: str, path: str) -> str:
    """Join `base_url` with a dialect path without repeating the API version segment:
    `.../api/v1` + `/v1/messages` -> `.../api/v1/messages`. Only version-shaped segments
    dedup, so a base ending in `/chat` doesn't swallow `/chat/completions`."""
    head = path.split("/")[1] if path.startswith("/") else ""
    base = base_url.rstrip("/")
    if VERSION_SEGMENT.fullmatch(head) and base.endswith(f"/{head}"):
        base = base[: -len(head) - 1]
    return base + path
