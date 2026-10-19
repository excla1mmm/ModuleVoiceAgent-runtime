"""HTTPS API and the media WebSocket.

Endpoints (``/v1/*`` require ``Authorization: Bearer <PBX id>.<token>``):

* ``GET  /health``                         public status; a PBX Bearer adds its own usage and the service
                                           latency, the operator token (``VA_ADMIN_TOKEN``) everything
* ``GET  /metrics``                        Prometheus text format (operator token)
* ``GET  /v1/agents``                      bundles this PBX has loaded (agent id + version) - heartbeat
* ``PUT  /v1/agents/{agent_id}``           load or replace a bundle (RAM only; 429 above the per-PBX quota)
* ``DELETE /v1/agents/{agent_id}``         forget a bundle
* ``GET  /v1/calls/{call_id}/result``      call result, retained until DELETE acknowledgement; ``?wait=2`` waits for an active call
* ``GET  /v1/usage?from=YYYY-MM-DD&to=YYYY-MM-DD``  this PBX's usage counters per UTC day and agent (billing)
* ``GET  /v1/voices``                      the voices this runtime speaks with and the default one
* ``GET  /v1/voices/{voice_id}/sample``    a sample of the voice as 8 kHz WAV (``?text=`` up to 300 characters)
* ``GET  /media``                        chan_websocket (subprotocol ``media``, Basic auth)
* ``GET  /ready``                          public readiness for a load balancer: 200 once bundles are taken over
                                           from the peer and greetings are warm, 503 while starting or draining

Operator API (``VA_ADMIN_TOKEN``), for blue-green deploys (``peer.py``, ``scripts/deploy.sh``):

* ``GET  /internal/handoff``               every bundle as the PBX sent it, for the next instance
* ``GET|DELETE /internal/calls/{tenant}/{call_id}/result``  a call result of this instance only (no forwarding)
* ``POST|DELETE /internal/drain``          stop / resume taking new calls; results stay readable

Many PBXs share one runtime. Every PBX is a tenant: its bundles, results and counters are keyed by its id,
it gets at most ``max_calls`` concurrent calls and ``max_agents`` bundles (``tenants.json`` or the
``VA_*_PER_TENANT`` defaults), and it never sees another tenant's numbers.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import logging
import ssl
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta

from aiohttp import web

from . import __version__
from .bundle import BundleError, parse_bundle
from .call import CallSession
from .config import VOICES, Settings
from .llm import ProviderRouter
from .metrics import Metrics
from .normalize import normalize
from .peer import Peer
from .prompt import MAX_TOKENS, prompt_tokens
from .protocol import SAMPLE_RATE, ProtocolError, call_params
from .stores import BundleStore, ResultStore
from .tenants import TenantRegistry, authenticate
from .usage import UsageMeter

log = logging.getLogger("voice_agent.server")
# A bundle within MAX_TOKENS is ~50 KB of JSON; the limit keeps JSON parsing of a bad bundle off the hot loop.
MAX_BUNDLE_BYTES = 1024 * 1024
MAX_RESULT_WAIT_S = 5.0
DRAIN_POLL_S = 0.2
MAX_USAGE_DAYS = 366
DEFAULT_USAGE_DAYS = 31
TAKEOVER_WARM_S = 120.0  # how long a new instance waits for the greetings of taken-over bundles
# What this version can do, for the PBX module (GET /health): features the module would otherwise detect by 404.
CAPABILITIES = ("result_ack", "usage", "voices", "voice_samples", "off_hours", "hotwords", "ready", "handoff")
SAMPLE_TEXT = "Здравствуйте! Вы позвонили в нашу компанию. Чем могу помочь?"
MAX_SAMPLE_CHARS = 300


@dataclass
class Runtime:
    settings: Settings
    tenants: TenantRegistry
    engines: object  # SpeechEngines; tests may pass any object with the same methods
    bundles: BundleStore
    results: ResultStore
    metrics: Metrics
    sessions: set = field(default_factory=set)  # active CallSessions, hung up on drain
    draining: bool = False
    # Greeting warm-ups trickle through one thread: after a restart every PBX pushes its bundles at once,
    # and a burst of synthesis must not compete with live calls for CPU.
    warm_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="warm"))
    usage: UsageMeter = field(default_factory=lambda: UsageMeter(None))  # billing counters (usage.py)
    boot_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # a new instance: the PBX pushes its bundles at once
    ready: bool = True  # False while a new instance takes bundles over from its peer (GET /ready)
    peer: Peer | None = None
    router: ProviderRouter | None = None  # which LLM provider calls ask first (failover), None: always the primary

    def tenant_call_limit(self, tenant: str) -> int:
        entry = self.tenants.get(tenant)
        limit = entry.max_calls if entry is not None and entry.max_calls is not None else self.settings.max_calls_per_tenant
        return min(limit, self.settings.max_calls) if limit else self.settings.max_calls

    def tenant_agent_limit(self, tenant: str) -> int:
        entry = self.tenants.get(tenant)
        if entry is not None and entry.max_agents is not None:
            return entry.max_agents
        return self.settings.max_agents_per_tenant

    def tenant_usage(self, tenant: str) -> dict:
        usage = self.metrics.tenant_snapshot(tenant)
        return {"id": tenant, "active_calls": usage["active_calls"], "max_calls": self.tenant_call_limit(tenant),
                "bundles": self.bundles.count(tenant), "max_agents": self.tenant_agent_limit(tenant),
                "pending_results": self.results.pending_count(tenant), "counters": usage["counters"]}

    def known_tenants(self) -> list[str]:
        seen = set(self.bundles.counts()) | set(self.results.pending_counts()) | set(self.metrics.tenant_active)
        seen |= set(self.metrics.snapshot()["tenants"])
        return sorted(seen)


RUNTIME = web.AppKey("runtime", Runtime)


def _unauthorized(rt: Runtime) -> web.HTTPUnauthorized:
    rt.metrics.inc("auth_failures")
    return web.HTTPUnauthorized(text='{"error":"unauthorized"}', content_type="application/json")


def _tenant(request: web.Request) -> str:
    rt = request.app[RUNTIME]
    tenant = authenticate(request.headers.get("Authorization", ""), rt.tenants, "bearer")
    if tenant is None:
        raise _unauthorized(rt)
    return tenant


def _require_admin(request: web.Request) -> None:
    if not _is_admin(request):
        raise _unauthorized(request.app[RUNTIME])


def _is_admin(request: web.Request) -> bool:
    token = request.app[RUNTIME].settings.admin_token
    header = request.headers.get("Authorization", "")
    if not token or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip().encode("utf-8", "replace"), token.encode("utf-8"))


def _llm_status(rt: Runtime) -> dict:
    if rt.router is not None:
        return rt.router.status()
    p = rt.settings.primary
    return {"active": p.name, "model": p.model} if p is not None else {}


async def health(request: web.Request) -> web.Response:
    rt = request.app[RUNTIME]
    status = "draining" if rt.draining else "ok"
    if _is_admin(request):
        snapshot = rt.metrics.snapshot()
        snapshot.pop("tenants", None)
        return web.json_response({"status": status, "version": __version__, "boot_id": rt.boot_id, "ready": rt.ready,
                                  "bundles": len(rt.bundles),
                                  "max_calls": rt.settings.max_calls, "pending_results": rt.results.pending_count(),
                                  "llm": _llm_status(rt),
                                  "tenants": {t: rt.tenant_usage(t) for t in rt.known_tenants()}, **snapshot})
    tenant = authenticate(request.headers.get("Authorization", ""), rt.tenants, "bearer")
    if tenant is None:
        return web.json_response({"status": status})
    return web.json_response({"status": status, "version": __version__, "boot_id": rt.boot_id,
                              "capabilities": list(CAPABILITIES), "tenant": rt.tenant_usage(tenant),
                              "stages": rt.metrics.snapshot()["stages"], "llm": _llm_status(rt)})


async def ready(request: web.Request) -> web.Response:
    rt = request.app[RUNTIME]
    status = "draining" if rt.draining else "ready" if rt.ready else "starting"
    return web.json_response({"status": status}, status=200 if status == "ready" else 503)


async def metrics(request: web.Request) -> web.Response:
    rt = request.app[RUNTIME]
    if not _is_admin(request):
        raise _unauthorized(rt)
    gauges = {"bundles": len(rt.bundles), "max_calls": rt.settings.max_calls,
              "pending_results": rt.results.pending_count(), "draining": int(rt.draining)}
    per_tenant = {"bundles": rt.bundles.counts(), "pending_results": rt.results.pending_counts()}
    usage = await asyncio.get_running_loop().run_in_executor(None, rt.usage.totals)
    return web.Response(text=rt.metrics.prometheus(gauges, per_tenant, usage), content_type="text/plain")


def _day(request: web.Request, key: str, default: date) -> date:
    raw = request.query.get(key, "")
    if not raw:
        return default
    try:
        if len(raw) != 10:  # fromisoformat also takes "20250101"; the API promises YYYY-MM-DD only
            raise ValueError(raw)
        return date.fromisoformat(raw)
    except ValueError:
        raise web.HTTPBadRequest(text=f'{{"error":"invalid_{key}","detail":"YYYY-MM-DD expected"}}',
                                 content_type="application/json") from None


async def usage(request: web.Request) -> web.Response:
    """Billing counters of the calling PBX for UTC days ``from``..``to`` (default: the last 31 days)."""
    tenant = _tenant(request)
    end = _day(request, "to", datetime.now(UTC).date())
    start = _day(request, "from", end - timedelta(days=DEFAULT_USAGE_DAYS - 1))
    if start > end or (end - start).days >= MAX_USAGE_DAYS:
        return web.json_response({"error": "invalid_range", "max_days": MAX_USAGE_DAYS}, status=400)
    meter = request.app[RUNTIME].usage
    report = await asyncio.get_running_loop().run_in_executor(None, meter.query, tenant, start, end)
    return web.json_response({"tenant": tenant, **report})


async def voices(request: web.Request) -> web.Response:
    _tenant(request)
    settings = request.app[RUNTIME].settings
    items = [{"id": v, "name": VOICES[v].name, "gender": VOICES[v].gender, "engine": VOICES[v].engine}
             for v in settings.voices]
    return web.json_response({"default": settings.default_voice, "voices": items})


def _wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return out.getvalue()


async def voice_sample(request: web.Request) -> web.Response:
    """How a voice sounds on the phone: the standard sample (cached) or the caller's text, e.g. a greeting.
    Synthesis goes through the warm-up thread, so previews never take CPU threads from live calls."""
    _tenant(request)
    rt = request.app[RUNTIME]
    voice = request.match_info["voice_id"]
    if voice not in rt.settings.voices:
        return web.json_response({"error": "unknown_voice"}, status=404)
    text = request.query.get("text", "").strip()
    if len(text) > MAX_SAMPLE_CHARS:
        return web.json_response({"error": "text_too_long", "limit": MAX_SAMPLE_CHARS}, status=400)
    loop = asyncio.get_running_loop()
    if text:
        pcm = await loop.run_in_executor(rt.warm_executor, rt.engines.synthesize, voice, normalize(text))
    else:
        pcm = await loop.run_in_executor(rt.warm_executor, rt.engines.phrase, voice, normalize(SAMPLE_TEXT))
    if not pcm:
        return web.json_response({"error": "nothing_to_say"}, status=422)
    return web.Response(body=_wav(pcm), content_type="audio/wav", headers={"Cache-Control": "no-store"})


async def list_agents(request: web.Request) -> web.Response:
    tenant = _tenant(request)
    items = [{"agent_id": s.bundle.agent_id, "version": s.bundle.version, "loaded_at": int(s.loaded_at)}
             for s in request.app[RUNTIME].bundles.list(tenant)]
    return web.json_response({"agents": items})


def _loaded_voice(rt: Runtime, bundle):
    """The bundle with a loaded voice: a voice of a newer module or one the operator did not load is replaced."""
    return bundle if bundle.voice in rt.settings.voices else replace(bundle, voice=rt.settings.default_voice)


def _log_warmup_failure(future: Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        log.warning("greeting warm-up failed: %s", future.exception())


async def put_agent(request: web.Request) -> web.Response:
    tenant = _tenant(request)
    rt = request.app[RUNTIME]
    agent_id = request.match_info["agent_id"]
    limit = rt.tenant_agent_limit(tenant)
    if not rt.bundles.has(tenant, agent_id) and rt.bundles.count(tenant) >= limit:
        rt.metrics.inc_tenant(tenant, "rejected_bundles")
        return web.json_response({"error": "too_many_agents", "limit": limit}, status=429)
    try:
        data = await request.json()
        bundle = parse_bundle(agent_id, data)
    except (ValueError, BundleError) as exc:
        return web.json_response({"error": "invalid_bundle", "detail": str(exc)[:300]}, status=422)
    tokens = prompt_tokens(bundle)
    if tokens > MAX_TOKENS:
        return web.json_response({"error": "prompt_too_large", "estimated_tokens": tokens, "limit": MAX_TOKENS},
                                 status=413)
    requested = bundle.voice
    bundle = _loaded_voice(rt, bundle)
    rt.bundles.put(tenant, bundle, data)
    log.info("bundle loaded", extra={"fields": {"tenant": tenant, "agent": agent_id, "version": bundle.version,
                                                "estimated_tokens": tokens, "voice": bundle.voice,
                                                "requested_voice": requested}})
    # warm the greeting so the first call does not wait for synthesis
    rt.warm_executor.submit(rt.engines.phrase, bundle.voice, normalize(bundle.greeting)).add_done_callback(
        _log_warmup_failure)
    return web.json_response({"agent_id": agent_id, "version": bundle.version, "estimated_tokens": tokens,
                              "voice": bundle.voice, "voice_fallback": bundle.voice != requested})


async def delete_agent(request: web.Request) -> web.Response:
    tenant = _tenant(request)
    deleted = request.app[RUNTIME].bundles.delete(tenant, request.match_info["agent_id"])
    return web.json_response({"deleted": deleted}, status=200 if deleted else 404)


def _wait(request: web.Request) -> float:
    try:
        return min(MAX_RESULT_WAIT_S, max(0.0, float(request.query.get("wait", "0"))))
    except ValueError:
        return 0.0


async def _local_result(rt: Runtime, tenant: str, call_id: str, wait: float) -> tuple[int, dict]:
    state, result = await rt.results.take(tenant, call_id, wait)
    if state == "ready":
        return 200, result
    if state == "active":
        return 202, {"status": "active"}
    return 404, {"error": "not_found"}


async def call_result(request: web.Request) -> web.Response:
    """A call result; one this instance does not know may belong to the peer (the instance before a deploy)."""
    tenant = _tenant(request)
    rt, call_id, wait = request.app[RUNTIME], request.match_info["call_id"], _wait(request)
    status, body = await _local_result(rt, tenant, call_id, wait)
    if status == 404 and rt.peer is not None:
        status, body = await rt.peer.result(tenant, call_id, wait) or (status, body)
    return web.json_response(body, status=status)


async def acknowledge_result(request: web.Request) -> web.Response:
    tenant = _tenant(request)
    rt, call_id = request.app[RUNTIME], request.match_info["call_id"]
    if rt.results.is_pending(tenant, call_id) or rt.peer is None:
        rt.results.acknowledge(tenant, call_id)
    else:
        await rt.peer.acknowledge(tenant, call_id)
    return web.Response(status=204)


async def internal_result(request: web.Request) -> web.Response:
    """A result of this instance only, for the peer (never forwarded again)."""
    _require_admin(request)
    rt, tenant, call_id = request.app[RUNTIME], request.match_info["tenant"], request.match_info["call_id"]
    if request.method == "DELETE":
        rt.results.acknowledge(tenant, call_id)
        return web.Response(status=204)
    status, body = await _local_result(rt, tenant, call_id, _wait(request))
    return web.json_response(body, status=status)


async def handoff(request: web.Request) -> web.Response:
    """Every bundle as the PBX sent it, for the instance that replaces this one."""
    _require_admin(request)
    rt = request.app[RUNTIME]
    items = [{"tenant": tenant, "agent_id": stored.bundle.agent_id, "data": stored.raw}
             for tenant, stored in rt.bundles.all() if stored.raw is not None]
    log.info("bundles handed over", extra={"fields": {"bundles": len(items)}})
    return web.json_response({"boot_id": rt.boot_id, "bundles": items})


async def drain_calls(request: web.Request) -> web.Response:
    """POST: take no new calls (503, the load balancer sends them to the other instance; /ready answers 503);
    active calls go on, results stay readable. DELETE: take calls again."""
    _require_admin(request)
    rt = request.app[RUNTIME]
    rt.draining = request.method == "POST"
    log.info("draining" if rt.draining else "draining cancelled",
             extra={"fields": {"active_calls": rt.metrics.active_calls, "pending_results": rt.results.pending_count()}})
    return web.json_response({"draining": rt.draining, "active_calls": rt.metrics.active_calls,
                              "pending_results": rt.results.pending_count()})


async def take_over(rt: Runtime) -> int:
    """Loads the peer's bundles (those a PBX has not pushed here meanwhile) and warms their greetings."""
    loop = asyncio.get_running_loop()
    taken, warming = 0, []
    for item in await rt.peer.bundles():
        tenant, agent_id, data = str(item.get("tenant", "")), str(item.get("agent_id", "")), item.get("data")
        if rt.tenants.get(tenant) is None or rt.bundles.has(tenant, agent_id):
            continue
        try:
            bundle = _loaded_voice(rt, parse_bundle(agent_id, data))
        except (BundleError, TypeError, ValueError) as exc:
            log.warning("handed-over bundle rejected: %s", exc)
            continue
        rt.bundles.put(tenant, bundle, data)
        taken += 1
        warming.append(loop.run_in_executor(rt.warm_executor, rt.engines.phrase, bundle.voice, normalize(bundle.greeting)))
    if warming:
        try:
            await asyncio.wait_for(asyncio.gather(*warming, return_exceptions=True), TAKEOVER_WARM_S)
        except TimeoutError:
            log.warning("greetings of handed-over bundles are still warming up")
    return taken


async def peer_ctx(app: web.Application):
    """With a peer: not ready until its bundles are taken over, so the load balancer sends no call too early."""
    rt = app[RUNTIME]
    if not rt.settings.peer_url:
        yield
        return
    rt.peer = rt.peer or Peer(rt.settings.peer_url, rt.settings.admin_token)
    rt.ready = False

    async def start() -> None:
        try:
            taken = await take_over(rt)
            log.info("ready", extra={"fields": {"bundles_taken_over": taken, "peer": rt.settings.peer_url}})
        except Exception:  # noqa: BLE001 - without the peer's bundles the PBXs push them again
            log.exception("taking bundles over failed")
        rt.ready = True
    task = asyncio.create_task(start())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    await rt.peer.close()


async def maintenance(app: web.Application):
    async def maintain():
        while True:
            await asyncio.sleep(min(30.0, app[RUNTIME].results.ttl_s / 2))
            app[RUNTIME].results.expire()
    task = asyncio.create_task(maintain())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    app[RUNTIME].warm_executor.shutdown(wait=False, cancel_futures=True)


async def flush_usage(app: web.Application):
    """Writes the usage counters to disk every ``VA_USAGE_FLUSH_SECONDS`` and once more at shutdown."""
    meter, loop = app[RUNTIME].usage, asyncio.get_running_loop()

    async def flush() -> None:
        try:
            await loop.run_in_executor(None, meter.flush)
        except Exception:  # noqa: BLE001 - the counters stay in RAM and go with the next flush
            log.exception("usage flush failed")

    async def periodic() -> None:
        while True:
            await asyncio.sleep(app[RUNTIME].settings.usage_flush_s)
            await flush()
    task = asyncio.create_task(periodic())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    await flush()  # cleanup runs after the server has shut down the call handlers, which recorded their calls


def _reject(rt: Runtime, tenant: str, reason: str) -> web.Response:
    rt.metrics.inc(f"rejected_{reason}")
    rt.metrics.inc_tenant(tenant, "rejected")
    return web.Response(status=503, text=reason, headers={"Retry-After": "30"})


async def media(request: web.Request) -> web.StreamResponse:
    rt = request.app[RUNTIME]
    tenant = authenticate(request.headers.get("Authorization", ""), rt.tenants, "basic")
    if tenant is None:
        rt.metrics.inc("auth_failures")
        return web.Response(status=401, headers={"WWW-Authenticate": 'Basic realm="voice-agent"'})
    try:
        params = call_params(request.path_qs)
    except ProtocolError as exc:
        return web.Response(status=400, text=str(exc))
    if rt.draining:
        return _reject(rt, tenant, "draining")
    if rt.metrics.active_calls >= rt.settings.max_calls:
        return _reject(rt, tenant, "capacity")
    if rt.metrics.tenant_active.get(tenant, 0) >= rt.tenant_call_limit(tenant):
        return _reject(rt, tenant, "tenant_capacity")
    rt.metrics.call_started(tenant)
    # Asterisk sends audio every 20 ms, so a ping only goes out on a silent link: a PBX that vanished
    # without closing the socket releases its call slot after ~1.5 heartbeats instead of VA_MAX_CALL_SECONDS.
    ws = web.WebSocketResponse(protocols=("media",), max_msg_size=1 << 20,
                               heartbeat=rt.settings.ws_heartbeat_s or None)
    session = None
    try:
        await ws.prepare(request)
        stored = rt.bundles.get(tenant, params.agent_id)
        reason = "bundle_missing" if stored is None else None
        session = CallSession(rt, ws, tenant, params, stored.bundle if stored else None,
                              stored.static_prompt if stored else "", reason, stored.terms if stored else ())
        rt.sessions.add(session)
        rt.metrics.inc("calls")
        rt.metrics.inc_tenant(tenant, "calls")
        log.info("call started", extra={"fields": {"call": session.id, "tenant": tenant, "agent": params.agent_id,
                                                   "unavailable": reason}})
        await session.run()
    finally:
        rt.sessions.discard(session)
        rt.metrics.call_ended(tenant)
    return ws


async def drain(runtime: Runtime, timeout_s: float) -> None:
    """Graceful stop: new calls get 503 (the PBX dials its fallback), active calls are hung up with a
    ``runtime_shutdown`` fallback result, and the PBXs get up to ``timeout_s`` to collect those results."""
    runtime.draining = True
    sessions = list(runtime.sessions)
    keys = [(s.tenant, s.params.call_id) for s in sessions]
    for session in sessions:
        session.request_shutdown()
    log.info("draining", extra={"fields": {"active_calls": len(sessions), "timeout_s": timeout_s}})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not runtime.sessions and not any(runtime.results.is_pending(t, c) for t, c in keys):
            break
        await asyncio.sleep(DRAIN_POLL_S)
    unclaimed = sum(runtime.results.is_pending(t, c) for t, c in keys)
    log.info("drained", extra={"fields": {"active_calls": len(runtime.sessions), "unclaimed_results": unclaimed}})


def build_app(runtime: Runtime) -> web.Application:
    app = web.Application(client_max_size=MAX_BUNDLE_BYTES)
    app[RUNTIME] = runtime
    app.cleanup_ctx.append(maintenance)
    app.cleanup_ctx.append(flush_usage)
    app.cleanup_ctx.append(peer_ctx)
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/v1/agents", list_agents)
    app.router.add_put(r"/v1/agents/{agent_id:[0-9]{1,16}}", put_agent)
    app.router.add_delete(r"/v1/agents/{agent_id:[0-9]{1,16}}", delete_agent)
    app.router.add_get(r"/v1/calls/{call_id:[A-Za-z0-9._-]{1,64}}/result", call_result)
    app.router.add_delete(r"/v1/calls/{call_id:[A-Za-z0-9._-]{1,64}}/result", acknowledge_result)
    app.router.add_get("/v1/usage", usage)
    app.router.add_get("/v1/voices", voices)
    app.router.add_get(r"/v1/voices/{voice_id:[A-Za-z0-9_.-]{1,64}}/sample", voice_sample)
    app.router.add_get("/media", media)
    app.router.add_get("/internal/handoff", handoff)
    internal_call = r"/internal/calls/{tenant:[A-Za-z0-9._-]{1,64}}/{call_id:[A-Za-z0-9._-]{1,64}}/result"
    app.router.add_get(internal_call, internal_result)
    app.router.add_delete(internal_call, internal_result)
    app.router.add_post("/internal/drain", drain_calls)
    app.router.add_delete("/internal/drain", drain_calls)
    return app


def ssl_context(settings: Settings) -> ssl.SSLContext | None:
    if not settings.tls_enabled:
        return None
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(settings.tls_cert, settings.tls_key)
    return ctx
