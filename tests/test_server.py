"""HTTP API and media WebSocket through a real aiohttp server on localhost."""

import asyncio
import base64
import json
import os

import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from conftest import bundle_data
from voice_agent_runtime.config import LlmProvider, Settings
from voice_agent_runtime.metrics import Metrics
from voice_agent_runtime.server import Runtime, build_app
from voice_agent_runtime.stores import BundleStore, ResultStore
from voice_agent_runtime.tenants import TenantRegistry, token_hash

TOKEN = "s3cret-token"
OTHER = "other-token"
ADMIN = "adm1n-token"


class SilentEngines:
    """Speech engines for protocol tests: silence in, a short tone out. No models are loaded."""

    class Vad:
        def prob(self, hop):
            return 0.0

    def new_vad(self):
        return self.Vad()

    def recognize(self, audio, terms=()):
        return ""

    def synthesize(self, voice, text):
        return (np.ones(800, dtype=np.int16) * 1000).tobytes()  # 100 ms

    def phrase(self, voice, text):
        return self.synthesize(voice, text)


def make_runtime(**settings):
    registry = TenantRegistry.from_mapping({"tenants": [
        {"id": "pbx-acme", "token_sha256": [token_hash(TOKEN)]},
        {"id": "pbx-other", "token_sha256": token_hash(OTHER)},
    ]})
    provider = LlmProvider("primary", "http://127.0.0.1:9/v1/chat/completions", "m", "k")
    settings.setdefault("admin_token", ADMIN)
    settings.setdefault("aec", False)  # synthetic test audio; the canceller has its own tests
    settings.setdefault("topic_filter", False)  # no side LLM requests; the filter has its own tests
    settings.setdefault("handoff_file", None)  # never the real /data; the handoff tests pass a file of their own
    return Runtime(Settings(primary=provider, **settings), registry, SilentEngines(), BundleStore(),
                   ResultStore(900), Metrics())


def bearer(tenant="pbx-acme", token=TOKEN):
    return {"Authorization": f"Bearer {tenant}.{token}"}


def admin():
    return {"Authorization": f"Bearer {ADMIN}"}


def basic(tenant="pbx-acme", token=TOKEN):
    return {"Authorization": "Basic " + base64.b64encode(f"{tenant}:{token}".encode()).decode()}


def run(coro_fn, **settings):
    async def main():
        runtime = make_runtime(**settings)
        async with TestClient(TestServer(build_app(runtime))) as client:
            await coro_fn(client, runtime)
    asyncio.run(main())


def test_auth_is_required_and_per_tenant():
    async def check(client, runtime):
        assert (await client.get("/v1/agents")).status == 401
        assert (await client.get("/v1/agents", headers=bearer(token="wrong"))).status == 401
        assert (await client.get("/v1/agents", headers=bearer("pbx-nobody"))).status == 401
        assert (await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())).status == 200
        other = await (await client.get("/v1/agents", headers=bearer("pbx-other", OTHER))).json()
        assert other == {"agents": []}  # tenants never see each other's agents
        mine = await (await client.get("/v1/agents", headers=bearer())).json()
        assert [(a["agent_id"], a["version"]) for a in mine["agents"]] == [("3", "v1")]
        health = await (await client.get("/health")).json()
        assert health == {"status": "ok"}
        health = await (await client.get("/health", headers=bearer())).json()
        assert health["tenant"]["bundles"] == 1 and health["tenant"]["id"] == "pbx-acme"
        assert health["llm"]["active"] == "primary"
        health = await (await client.get("/health", headers=admin())).json()
        assert health["bundles"] == 1 and health["tenants"]["pbx-acme"]["bundles"] == 1
        assert health["llm"]["active"] == "primary"
    run(check)


def test_bundle_validation_and_delete():
    async def check(client, runtime):
        resp = await client.put("/v1/agents/3", json=bundle_data(voice="bad voice/"), headers=bearer())
        assert resp.status == 422
        huge = bundle_data(knowledge=[{"topic": f"t{i}", "answer": "а" * 3000} for i in range(30)])
        assert (await client.put("/v1/agents/3", json=huge, headers=bearer())).status == 413
        assert (await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())).status == 200
        assert (await client.delete("/v1/agents/3", headers=bearer())).status == 200
        assert (await client.delete("/v1/agents/3", headers=bearer())).status == 404
    run(check)


class RecordingEngines(SilentEngines):
    def __init__(self):
        self.spoken = []

    def synthesize(self, voice, text):
        self.spoken.append((voice, text))
        return b"" if text == "" else super().synthesize(voice, text)


def test_voice_not_loaded_falls_back_to_the_default_voice():
    async def check(client, runtime):
        for voice in ("ru_RU-ruslan-medium", "silero-oksana"):  # unknown here / known but not loaded
            resp = await client.put("/v1/agents/3", json=bundle_data(voice=voice), headers=bearer())
            body = await resp.json()
            assert resp.status == 200 and body["voice"] == "ru_RU-denis-medium" and body["voice_fallback"] is True
            assert runtime.bundles.get("pbx-acme", "3").bundle.voice == "ru_RU-denis-medium"
        resp = await client.put("/v1/agents/3", json=bundle_data(voice="silero-roman"), headers=bearer())
        assert (await resp.json())["voice_fallback"] is True
        resp = await client.put("/v1/agents/3", json=bundle_data(voice="ru_RU-dmitri-medium"), headers=bearer())
        assert (await resp.json()) | {"estimated_tokens": 0} == {
            "agent_id": "3", "version": "v1", "estimated_tokens": 0, "voice": "ru_RU-dmitri-medium", "voice_fallback": False}
    run(check, voices=("ru_RU-dmitri-medium", "ru_RU-denis-medium"), default_voice="ru_RU-denis-medium")


def test_voices_are_listed_for_the_pbx_only():
    async def check(client, runtime):
        assert (await client.get("/v1/voices")).status == 401
        assert (await client.get("/v1/voices", headers=bearer(token="wrong"))).status == 401
        body = await (await client.get("/v1/voices", headers=bearer())).json()
        assert body["default"] == "silero-zinaida"
        assert body["voices"] == [
            {"id": "silero-zinaida", "name": "Зинаида", "gender": "female", "engine": "silero"},
            {"id": "ru_RU-dmitri-medium", "name": "Дмитрий", "gender": "male", "engine": "piper"},
        ]
    run(check, voices=("silero-zinaida", "ru_RU-dmitri-medium"), default_voice="silero-zinaida")


def test_voice_sample_is_a_wav_of_the_standard_phrase_or_the_given_text():
    import io
    import wave

    engines = RecordingEngines()

    async def check(client, runtime):
        runtime.engines = engines
        assert (await client.get("/v1/voices/silero-oksana/sample")).status == 401
        resp = await client.get("/v1/voices/silero-oksana/sample", headers=bearer())
        assert resp.status == 200 and resp.content_type == "audio/wav" and resp.headers["Cache-Control"] == "no-store"
        with wave.open(io.BytesIO(await resp.read())) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (8000, 1, 2, 800)
        resp = await client.get("/v1/voices/silero-oksana/sample", params={"text": "Работаем с 9:00 до 18:00"},
                                headers=bearer())
        assert resp.status == 200
        assert engines.spoken[-1] == ("silero-oksana", "Работаем с девяти до восемнадцати")  # normalized like a call
        assert len(engines.spoken) == 2 and engines.spoken[0][0] == "silero-oksana"
        resp = await client.get("/v1/voices/silero-oksana/sample", params={"text": "а" * 301}, headers=bearer())
        assert resp.status == 400 and (await resp.json())["limit"] == 300
        resp = await client.get("/v1/voices/silero-roman/sample", headers=bearer())  # not loaded
        assert resp.status == 404 and (await resp.json())["error"] == "unknown_voice"
        resp = await client.get("/v1/voices/silero-oksana/sample", params={"text": "«»"}, headers=bearer())
        assert resp.status == 422 and (await resp.json())["error"] == "nothing_to_say"
    run(check, voices=("silero-oksana", "ru_RU-dmitri-medium"))


def test_result_is_retried_until_acknowledged():
    async def check(client, runtime):
        runtime.results.start("pbx-acme", "call-1")
        resp = await client.get("/v1/calls/call-1/result", headers=bearer())
        assert resp.status == 202
        asyncio.get_running_loop().call_later(0.2, runtime.results.finish, "pbx-acme", "call-1",
                                              {"outcome": "transferred", "transfer": {"extension": "210"}})
        resp = await client.get("/v1/calls/call-1/result?wait=2", headers=bearer())
        assert resp.status == 200 and (await resp.json())["transfer"] == {"extension": "210"}
        assert (await client.get("/v1/calls/call-1/result", headers=bearer())).status == 200
        assert (await client.delete("/v1/calls/call-1/result", headers=bearer("pbx-other", OTHER))).status == 204
        assert (await client.get("/v1/calls/call-1/result", headers=bearer())).status == 200
        assert (await client.delete("/v1/calls/call-1/result", headers=bearer())).status == 204
        assert (await client.delete("/v1/calls/call-1/result", headers=bearer())).status == 204
        assert (await client.get("/v1/calls/call-1/result", headers=bearer())).status == 404
        # another tenant cannot read it
        runtime.results.finish("pbx-acme", "call-2", {"outcome": "completed"})
        assert (await client.get("/v1/calls/call-2/result", headers=bearer("pbx-other", OTHER))).status == 404
        assert (await client.get("/v1/calls/call-2/result", headers=bearer())).status == 200
    run(check)


async def _media_call(client, path, headers):
    ws = await client.ws_connect(path, headers=headers, protocols=("media",))
    await ws.send_str("MEDIA_START connection_id:c1 channel:WebSocket/x format:slin optimal_frame_size:320 ptime:20")
    frames, commands = [], []
    while True:
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        if msg.type == WSMsgType.BINARY:
            frames.append(msg.data)
        elif msg.type == WSMsgType.TEXT:
            commands.append(msg.data)
            if msg.data == "HANGUP":
                break
        else:
            break
    await ws.close()
    return ws, frames, commands


def test_media_requires_basic_auth():
    async def check(client, runtime):
        resp = await client.get("/media?callid=c1&agent=3", headers=basic(token="nope"))
        assert resp.status == 401
        resp = await client.get("/media?agent=3", headers=basic())
        assert resp.status == 400
    run(check)


def test_missing_bundle_plays_unavailable_phrase_and_hangs_up_with_fallback():
    async def check(client, runtime):
        runtime.ready_at -= 60  # long after the start: a missing bundle is not waited for
        ws, frames, commands = await _media_call(client, "/media?callerid=79991234567&callid=c-9&agent=3", basic())
        assert ws.protocol == "media"
        assert frames and all(len(f) == 320 for f in frames)
        assert commands == ["HANGUP"]
        resp = await client.get("/v1/calls/c-9/result?wait=1", headers=bearer())
        result = await resp.json()
        assert result["outcome"] == "fallback" and result["fallback_reason"] == "bundle_missing"
        assert result["caller"] == "79991234567" and result["turns"] == []
    run(check)


def test_call_right_after_the_start_waits_for_its_bundle():
    async def check(client, runtime):
        async def push_later():
            await asyncio.sleep(0.6)  # the PBX notices the new instance and pushes its bundles
            assert (await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())).status == 200
        pusher = asyncio.create_task(push_later())
        ws = await client.ws_connect("/media?callerid=79991234567&callid=c-w&agent=3", headers=basic(),
                                     protocols=("media",))
        await ws.send_str("MEDIA_START connection_id:c1 format:slin optimal_frame_size:320 ptime:20")
        greeting = [await asyncio.wait_for(ws.receive(), 5) for _ in range(3)]
        assert all(m.type == WSMsgType.BINARY for m in greeting)
        await ws.close()
        await pusher
        result = await (await client.get("/v1/calls/c-w/result?wait=2", headers=bearer())).json()
        assert result["outcome"] != "fallback" and result.get("fallback_reason") != "bundle_missing"
    run(check)


def test_capacity_limit_rejects_excess_connection():
    async def check(client, runtime):
        await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())
        runtime.metrics.active_calls = 1  # one call already running
        response = await client.get("/media?callid=c-10&agent=3", headers=basic())
        assert response.status == 503
        assert runtime.metrics.active_calls == 1
        assert not runtime.results.is_active("pbx-acme", "c-10")
    run(check, max_calls=1)


def test_greeting_is_played_and_caller_hangup_is_recorded():
    async def check(client, runtime):
        await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())
        ws = await client.ws_connect("/media?callerid=79991234567&callid=c-11&agent=3", headers=basic(),
                                     protocols=("media",))
        await ws.send_str("MEDIA_START connection_id:c1 format:slin optimal_frame_size:320 ptime:20")
        greeting = [await asyncio.wait_for(ws.receive(), 5) for _ in range(3)]
        assert all(m.type == WSMsgType.BINARY and len(m.data) == 320 for m in greeting)
        await ws.send_bytes(b"\0" * 640)  # caller audio (silence)
        await ws.close()  # the caller hangs up
        result = await (await client.get("/v1/calls/c-11/result?wait=2", headers=bearer())).json()
        assert result["outcome"] == "caller_hangup" and result["agent_id"] == "3"
    run(check)


@pytest.mark.parametrize("path", ["/metrics"])
def test_metrics_endpoint(path):
    async def check(client, runtime):
        runtime.metrics.observe("latency_ms", 615)
        text = await (await client.get(path, headers=admin())).text()
        assert 'voice_agent_latency_milliseconds{quantile="0.5"} 615.0' in text
    run(check)


def test_results_expire_without_readers():
    async def main():
        runtime = make_runtime()
        runtime.results = ResultStore(0.04)
        async with TestClient(TestServer(build_app(runtime))):
            runtime.results.finish("pbx-acme", "expired", {"caller": "7999"})
            await asyncio.sleep(0.15)
            assert not runtime.results._results
    asyncio.run(main())


def run_php(tmp_path, script, prepare=None, **settings):
    """Runs ``script`` (PHP, the module's RuntimeClient loaded) against this runtime over TLS; returns stdout.
    The script gets the module test bootstrap as $argv[1] and the runtime URL as $argv[2]."""
    import ssl
    import subprocess
    from pathlib import Path

    module_dir = Path(os.environ.get(
        "VOICE_AGENT_MODULE_DIR", Path(__file__).resolve().parents[2] / "Extensions" / "ModuleVoiceAgent"))
    bootstrap = module_dir / "Tests" / "bootstrap.php"
    if not bootstrap.is_file():
        pytest.skip("ModuleVoiceAgent checkout is needed for the PHP client integration test")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost"],
                   check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    output = {}

    async def main():
        runtime = make_runtime(**settings)
        if prepare:
            prepare(runtime)
        server = TestServer(build_app(runtime))
        await server.start_server(ssl=context)
        async with TestClient(server) as client:
            process = await asyncio.create_subprocess_exec(
                "php", "-r", script, str(bootstrap), str(client.make_url("")),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await process.communicate()
            assert process.returncode == 0, err.decode()
            output["out"] = out.decode()
    asyncio.run(main())
    return output["out"]


def test_php_client_retries_result_then_acknowledges(tmp_path):
    script = r"""require $argv[1];
$c = new \Modules\ModuleVoiceAgent\Lib\Services\RuntimeClient($argv[2], 'pbx-acme', 's3cret-token', false);
echo json_encode([$c->fetchResult('php-1', 0)['status'], $c->fetchResult('php-1', 0)['status'],
    $c->acknowledgeResult('php-1')['status'], $c->fetchResult('php-1', 0)['status']]);"""
    out = run_php(tmp_path, script,
                  lambda runtime: runtime.results.finish("pbx-acme", "php-1", {"outcome": "callback"}))
    assert out == "[200,200,204,404]"


def test_php_module_lists_voices_and_plays_samples(tmp_path):
    script = r"""require $argv[1];
use Modules\ModuleVoiceAgent\Lib\Services\{RuntimeClient, VoiceService};
$c = new RuntimeClient($argv[2], 'pbx-acme', 's3cret-token', false);
$catalog = VoiceService::catalog($c);
$sample = VoiceService::sample($c, 'silero-oksana', 'Здравствуйте, компания «Ромашка»!');
$wav = base64_decode(substr($sample['audio'], strlen('data:audio/wav;base64,')));
$raw = $c->voiceSample('silero-oksana');
echo json_encode([
    'checked' => $catalog['checked'], 'default' => $catalog['runtime_default'],
    'available' => array_column($catalog['voices'], 'available', 'id'),
    'sample' => [$sample['available'], substr($wav, 0, 4), substr($wav, 8, 4), strlen($wav)],
    'raw' => [$raw['status'], strlen($raw['audio']), $raw['body']],
    'missing' => VoiceService::sample($c, 'silero-roman', '')['reason'],
    'silent' => VoiceService::sample($c, 'silero-oksana', '«»')['reason'],
], JSON_UNESCAPED_UNICODE);"""
    out = json.loads(run_php(tmp_path, script, lambda runtime: setattr(runtime, "engines", RecordingEngines()),
                             voices=("silero-oksana", "ru_RU-dmitri-medium")))
    assert out["checked"] is True and out["default"] == "ru_RU-dmitri-medium"
    assert out["available"] == {"ru_RU-dmitri-medium": True, "ru_RU-denis-medium": False, "silero-ekaterina": False,
                                "silero-oksana": True, "silero-zinaida": False, "silero-roman": False,
                                "silero-eduard": False}
    assert out["sample"] == [True, "RIFF", "WAVE", 44 + 1600]  # 100 ms of the test tone
    assert out["raw"] == [200, 44 + 1600, []]
    assert out["missing"] == "voice_unavailable" and out["silent"] == "nothing_to_say"
