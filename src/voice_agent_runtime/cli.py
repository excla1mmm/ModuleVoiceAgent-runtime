"""Command line: ``serve``, ``download-models``, ``hash-token``, ``healthcheck``, ``status``, ``drain``.

``healthcheck --ready`` and ``drain`` are the steps of a blue-green deploy (``scripts/deploy.sh``), run inside
the container: wait until the new instance is ready, then let the old one finish its calls."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import signal
import sys
import time
import urllib.error
import urllib.request

from aiohttp import web

from . import logs
from .config import ConfigError, Settings
from .llm import ProviderRouter
from .metrics import Metrics
from .server import Runtime, build_app, drain, ssl_context
from .stores import BundleStore, ResultStore
from .tenants import TenantRegistry, token_hash
from .usage import UsageMeter

log = logging.getLogger("voice_agent")


def serve() -> int:
    try:
        settings = Settings.from_env()
        if not settings.tls_enabled and not settings.trusted_proxy:
            raise ConfigError("TLS certificate and key are required unless VA_TRUSTED_PROXY=true")
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    logs.setup(settings.log_level)
    from .speech import SpeechEngines

    tenants = TenantRegistry(settings.tenants_file)
    if not len(tenants):
        log.warning("no tenants loaded from %s: every request will be rejected", settings.tenants_file)
    runtime = Runtime(settings, tenants, SpeechEngines(settings), BundleStore(),
                      ResultStore(settings.result_ttl_s, settings.max_results_per_tenant), Metrics(),
                      usage=UsageMeter(settings.usage_db))
    if settings.secondary is not None and settings.failover_after:
        runtime.router = ProviderRouter(settings.primary, settings.secondary, settings.hedge_ms / 1000,
                                        settings.failover_after, settings.failover_window_s,
                                        settings.recovery_probe_s, settings.recovery_probes)
    asyncio.run(run_server(runtime))
    return 0


async def run_server(runtime: Runtime) -> None:
    """Serves until SIGTERM/SIGINT, then drains: active calls are hung up with a fallback result and the
    PBXs get ``VA_DRAIN_SECONDS`` to collect it before the listener closes (aiohttp itself would close the
    listener first, so the drain is done here)."""
    settings = runtime.settings
    runner = web.AppRunner(build_app(runtime), access_log=None, shutdown_timeout=5.0)
    await runner.setup()
    await web.TCPSite(runner, settings.host, settings.port, ssl_context=ssl_context(settings)).start()
    log.info("listening on %s:%d (%s)", settings.host, settings.port, "TLS" if settings.tls_enabled else "plain HTTP")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
        log.info("stop requested, draining for up to %ds", settings.drain_s)
        await drain(runtime, settings.drain_s)
    finally:
        if runtime.router is not None:
            runtime.router.close()
        await runner.cleanup()


def download() -> int:
    import os
    from pathlib import Path

    from .config import voices_from_env
    from .speech import download_models

    try:
        voices = voices_from_env(os.environ)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    download_models(Path(os.environ.get("VA_MODELS_DIR", "/models")), voices)
    print("models downloaded")
    return 0


def hash_token(token: str | None) -> int:
    token = token or getpass.getpass("PBX token: ")
    print(token_hash(token.strip()))
    return 0


def _local_url(path: str) -> str:
    scheme = "https" if os.environ.get("VA_TLS_CERT") and os.environ.get("VA_TLS_KEY") else "http"
    return f"{scheme}://127.0.0.1:{os.environ.get('VA_LISTEN_PORT', '8443')}{path}"


def _request(url: str, method: str = "GET", token: str = "") -> dict:
    import ssl

    ctx = ssl._create_unverified_context()  # the container checks itself; the certificate is for clients
    request = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(request, timeout=5, context=ctx) as resp:
        return json.loads(resp.read())


def healthcheck(url: str, expected: str = "ok") -> int:
    try:
        return 0 if _request(url).get("status") == expected else 1
    except (OSError, ValueError):  # /ready answers 503 (HTTPError) while starting or draining
        return 1


def _admin_token() -> str:
    try:
        token = Settings.from_env().admin_token
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return ""
    if not token:
        print("VA_ADMIN_TOKEN is required", file=sys.stderr)
    return token


def status() -> int:
    """One JSON line about this instance: active calls, unclaimed results, ready, draining, version."""
    token = _admin_token()
    if not token:
        return 2
    try:
        health = _request(_local_url("/health"), token=token)
    except (OSError, ValueError) as exc:
        print(f"health failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: health.get(k) for k in ("active_calls", "pending_results", "ready", "status", "version",
                                                 "boot_id", "bundles")}))
    return 0


def drain_calls(wait_s: int, settle_s: int) -> int:
    """Stops taking new calls, waits up to ``wait_s`` for the active calls to end and then up to ``settle_s``
    for the PBXs to collect their results. Exit 0: nothing left; 3: timed out (stopping now drops what is left)."""
    token = _admin_token()
    if not token:
        return 2
    try:
        state = _request(_local_url("/internal/drain"), "POST", token)
    except (OSError, ValueError) as exc:
        print(f"drain failed: {exc}", file=sys.stderr)
        return 1
    started, calls_done_at = time.monotonic(), None
    while True:
        calls, results = int(state.get("active_calls", 0)), int(state.get("pending_results", 0))
        print(json.dumps({"active_calls": calls, "pending_results": results}), flush=True)
        now = time.monotonic()
        if calls == 0:
            calls_done_at = calls_done_at or now
            if results == 0:
                return 0
            if now - calls_done_at >= settle_s:
                return 3
        elif now - started >= wait_s:
            return 3
        time.sleep(2)
        try:
            state = _request(_local_url("/health"), token=token)
        except (OSError, ValueError) as exc:
            print(f"health failed: {exc}", file=sys.stderr)
            return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice-agent-runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the HTTPS/WSS server")
    sub.add_parser("download-models", help="fetch the VAD, STT and TTS models into VA_MODELS_DIR")
    p_hash = sub.add_parser("hash-token", help="print the SHA-256 of a PBX token for tenants.json")
    p_hash.add_argument("token", nargs="?")
    p_health = sub.add_parser("healthcheck", help="exit 0 when /health answers ok (--ready: /ready answers ready)")
    p_health.add_argument("--url")
    p_health.add_argument("--ready", action="store_true", help="check readiness for calls instead of liveness")
    sub.add_parser("status", help="print active calls, unclaimed results and readiness as JSON (operator token)")
    p_drain = sub.add_parser("drain", help="stop taking calls and wait until the active ones end (blue-green deploy)")
    p_drain.add_argument("--wait", type=int, default=1860, help="max seconds for active calls (default 1860)")
    p_drain.add_argument("--settle", type=int, default=60, help="max seconds for results after the calls (default 60)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve()
    if args.command == "download-models":
        return download()
    if args.command == "hash-token":
        return hash_token(args.token)
    if args.command == "status":
        return status()
    if args.command == "drain":
        return drain_calls(args.wait, args.settle)
    if args.ready:
        return healthcheck(args.url or _local_url("/ready"), "ready")
    return healthcheck(args.url or _local_url("/health"))


if __name__ == "__main__":
    sys.exit(main())
