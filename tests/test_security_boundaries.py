"""Checks that public requests and excess calls cannot reach sensitive work."""

import asyncio
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from test_server import admin, basic, bearer, make_runtime
from voice_agent_runtime.cli import serve
from voice_agent_runtime.config import Settings
from voice_agent_runtime.server import build_app


def test_public_health_is_minimal_and_metrics_require_operator_token():
    async def check():
        runtime = make_runtime()
        runtime.metrics.observe("latency_ms", 615)
        async with TestClient(TestServer(build_app(runtime))) as client:
            public = await (await client.get("/health")).json()
            assert public == {"status": "ok"}
            assert (await client.get("/metrics")).status == 401
            # a PBX sees the service latency and its own usage, nothing about other PBXs
            detailed = await (await client.get("/health", headers=bearer())).json()
            assert detailed["stages"]["latency_ms"]["p50"] == 615
            assert detailed["tenant"]["id"] == "pbx-acme" and "tenants" not in detailed
            assert "bundles" not in detailed and "pending_results" not in detailed
            assert (await client.get("/metrics", headers=bearer())).status == 401
            # the operator token sees everything
            assert (await client.get("/metrics", headers=admin())).status == 200
            full = await (await client.get("/health", headers=admin())).json()
            assert "tenants" in full and full["max_calls"] == runtime.settings.max_calls
    asyncio.run(check())


def test_metrics_stay_closed_without_operator_token():
    async def check():
        runtime = make_runtime(admin_token="")
        async with TestClient(TestServer(build_app(runtime))) as client:
            assert (await client.get("/metrics", headers={"Authorization": "Bearer "})).status == 401
            assert (await client.get("/metrics", headers=bearer())).status == 401
    asyncio.run(check())


def test_plaintext_runtime_requires_explicit_trusted_proxy(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("VA_TLS_CERT", raising=False)
    monkeypatch.delenv("VA_TLS_KEY", raising=False)
    monkeypatch.delenv("VA_TRUSTED_PROXY", raising=False)
    with patch("voice_agent_runtime.cli.logs.setup") as logging_setup:
        assert serve() == 2
    logging_setup.assert_not_called()
    proxy = Settings.from_env({"LLM_API_KEY": "test-key", "VA_TRUSTED_PROXY": "true"})
    assert proxy.trusted_proxy and not proxy.tls_enabled


def test_excess_media_connection_is_rejected_before_session_creation():
    async def check():
        runtime = make_runtime(max_calls=1)
        runtime.metrics.active_calls = 1
        async with TestClient(TestServer(build_app(runtime))) as client:
            response = await client.get("/media?callid=extra&agent=3", headers=basic())
            assert response.status == 503
            assert runtime.metrics.active_calls == 1
            assert not runtime.results.is_active("pbx-acme", "extra")
    asyncio.run(check())
