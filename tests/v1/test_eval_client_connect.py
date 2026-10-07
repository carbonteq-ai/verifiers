"""The eval client retries a connection that never reached the provider, and nothing else."""

import asyncio

import httpx
import pytest

from verifiers.v1.clients import eval as eval_client
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.errors import ProviderError


def client_with(handler, monkeypatch) -> EvalClient:
    monkeypatch.setattr(eval_client, "CONNECT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setenv("VF_CONNECT_TEST_KEY", "fixture")
    client = EvalClient(
        EvalClientConfig(
            base_url="http://provider.test/v1", api_key_var="VF_CONNECT_TEST_KEY"
        )
    )
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_slow_connects_are_retried_until_the_provider_accepts(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) < 3:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        return httpx.Response(200, json={"ok": True})

    client = client_with(handler, monkeypatch)
    response = asyncio.run(
        client._request("http://provider.test/v1/chat/completions", {}, httpx.Headers())
    )
    assert response.json() == {"ok": True}
    assert len(calls) == 3


def test_a_provider_that_never_accepts_fails_after_the_bounded_attempts(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    client = client_with(handler, monkeypatch)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            client._request(
                "http://provider.test/v1/chat/completions", {}, httpx.Headers()
            )
        )
    assert raised.value.status_code == 503
    assert len(calls) == eval_client.CONNECT_ATTEMPTS


def test_failures_after_the_request_was_sent_are_not_retried(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("read timed out", request=request)

    client = client_with(handler, monkeypatch)
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            client._request(
                "http://provider.test/v1/chat/completions", {}, httpx.Headers()
            )
        )
    assert raised.value.status_code == 504
    assert len(calls) == 1


def test_idle_connections_expire_before_the_provider_closes_them():
    # uvicorn (vLLM's server) closes idle keep-alive connections after 5 s.
    from verifiers.v1.clients.base import DEFAULT_LIMITS

    assert (
        DEFAULT_LIMITS.keepalive_expiry is not None
        and DEFAULT_LIMITS.keepalive_expiry < 5.0
    )
