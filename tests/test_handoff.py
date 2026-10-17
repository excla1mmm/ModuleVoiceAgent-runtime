"""Blue-green deploy: readiness, draining, bundles taken over from the peer, results answered by the peer."""

import asyncio
import json
import socket
import time
import urllib.error
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from conftest import bundle_data
from test_server import ADMIN, RecordingEngines, admin, basic, bearer, make_runtime, run
from voice_agent_runtime import cli
from voice_agent_runtime.config import ConfigError, Settings
from voice_agent_runtime.peer import Peer
from voice_agent_runtime.server import build_app, take_over
from voice_agent_runtime.tenants import TenantRegistry, token_hash


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def until_ready(client, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await client.get("/ready")
        if resp.status == 200:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("never ready")


def pair(check, blue_settings=None, green_settings=None, prepare_blue=None):
    """Runs blue (the old instance) and green (the new one, peer = blue) on localhost."""
    async def main():
        blue = make_runtime(**(blue_settings or {}))
        if prepare_blue:
            prepare_blue(blue)
        async with TestClient(TestServer(build_app(blue))) as blue_client:
            green = make_runtime(peer_url=str(blue_client.make_url("")).rstrip("/"), **(green_settings or {}))
            green.engines = RecordingEngines()
            async with TestClient(TestServer(build_app(green))) as green_client:
                await until_ready(green_client)
                await check(blue_client, blue, green_client, green)
    asyncio.run(main())


def test_ready_health_and_drain():
    async def check(client, runtime):
        assert (await client.get("/ready")).status == 200
        body = await (await client.get("/health", headers=bearer())).json()
        assert len(body["boot_id"]) == 32 and body["boot_id"] == runtime.boot_id and body["version"] == "0.2.0"
        assert {"voices", "usage", "result_ack", "handoff", "ready"} <= set(body["capabilities"])
        assert "boot_id" not in await (await client.get("/health")).json()  # anonymous: status only
        assert (await client.post("/internal/drain", headers=bearer())).status == 401  # a PBX cannot drain
        assert (await client.post("/internal/drain")).status == 401
        runtime.results.finish("pbx-acme", "c-1", {"outcome": "transferred"})
        resp = await client.post("/internal/drain", headers=admin())
        assert await resp.json() == {"draining": True, "active_calls": 0, "pending_results": 1}
        resp = await client.get("/ready")
        assert resp.status == 503 and (await resp.json())["status"] == "draining"
        await client.put("/v1/agents/3", json=bundle_data(), headers=bearer())
        assert (await client.get("/media?callid=c-2&agent=3", headers=basic())).status == 503  # no new calls
        assert (await client.get("/v1/calls/c-1/result", headers=bearer())).status == 200  # results still served
        assert (await (await client.get("/health")).json())["status"] == "draining"
        assert (await (await client.delete("/internal/drain", headers=admin())).json())["draining"] is False
        assert (await client.get("/ready")).status == 200
    run(check)
    assert make_runtime().boot_id != make_runtime().boot_id


def test_new_instance_takes_bundles_over_before_it_is_ready():
    async def check(blue_client, blue, green_client, green):
        for agent in ("3", "4"):
            assert green.bundles.get("pbx-acme", agent) is not None
        assert green.bundles.get("pbx-acme", "3").bundle.voice == "ru_RU-denis-medium"  # not loaded here: default
        assert green.bundles.get("pbx-other", "5").bundle.greeting == "Другая АТС"
        assert green.bundles.get("pbx-acme", "3").raw == bundle_data(voice="silero-oksana", extension="7003")
        greetings = {text for _, text in green.engines.spoken}
        assert "Здравствуйте! Компания Телеком Сервис, чем могу помочь?" in greetings  # warmed before ready
        assert (await green_client.get("/v1/agents", headers=bearer())).status == 200
        listed = await (await green_client.get("/v1/agents", headers=bearer())).json()
        assert sorted(a["agent_id"] for a in listed["agents"]) == ["3", "4"]  # the PBX sees nothing to push
        resp = await blue_client.get("/internal/handoff", headers=bearer())
        assert resp.status == 401  # a PBX token never reads other tenants' bundles

    def prepare(blue):
        from voice_agent_runtime.bundle import parse_bundle

        for tenant, agent, data in (("pbx-acme", "3", bundle_data(voice="silero-oksana", extension="7003")),
                                    ("pbx-acme", "4", bundle_data(extension="7004")),
                                    ("pbx-other", "5", bundle_data(extension="7005", greeting="Другая АТС"))):
            blue.bundles.put(tenant, parse_bundle(agent, data), data)
    pair(check, green_settings={"voices": ("ru_RU-dmitri-medium", "ru_RU-denis-medium"),
                                "default_voice": "ru_RU-denis-medium"}, prepare_blue=prepare)


def test_take_over_keeps_newer_pushes_and_skips_unknown_tenants_and_bad_bundles():
    async def check(blue_client, blue, green_client, green):
        from voice_agent_runtime.bundle import parse_bundle
        blue.bundles.put("pbx-acme", parse_bundle("3", bundle_data(version="v1")), bundle_data(version="v1"))
        blue.bundles.put("pbx-other", parse_bundle("5", bundle_data()), bundle_data())
        blue.bundles.put("pbx-acme", parse_bundle("6", bundle_data()), {"schema": 99})  # damaged in transit
        green.bundles = type(green.bundles)()
        green.bundles.put("pbx-acme", parse_bundle("3", bundle_data(version="v2")), bundle_data(version="v2"))
        green.tenants = TenantRegistry.from_mapping({"tenants": [{"id": "pbx-acme", "token_sha256": token_hash("x")}]})
        assert await take_over(green) == 0
        assert green.bundles.get("pbx-acme", "3").bundle.version == "v2"  # the PBX pushed it here meanwhile
        assert green.bundles.get("pbx-other", "5") is None and green.bundles.get("pbx-acme", "6") is None
    pair(check)


def test_results_of_calls_the_peer_served_are_answered_and_acknowledged_there():
    async def check(blue_client, blue, green_client, green):
        blue.results.finish("pbx-acme", "old-1", {"outcome": "callback"})
        blue.results.start("pbx-acme", "old-2")
        resp = await green_client.get("/v1/calls/old-1/result", headers=bearer())
        assert resp.status == 200 and (await resp.json()) == {"outcome": "callback"}
        assert (await green_client.get("/v1/calls/old-2/result", headers=bearer())).status == 202
        assert (await green_client.get("/v1/calls/old-1/result", headers=bearer(tenant="pbx-other", token="other-token"))).status == 404
        assert (await green_client.delete("/v1/calls/old-1/result", headers=bearer())).status == 204
        assert not blue.results.is_pending("pbx-acme", "old-1")
        assert (await green_client.get("/v1/calls/old-1/result", headers=bearer())).status == 404
        green.results.finish("pbx-acme", "new-1", {"outcome": "transferred"})
        assert (await green_client.delete("/v1/calls/new-1/result", headers=bearer())).status == 204
        assert not green.results.is_pending("pbx-acme", "new-1")
        assert (await blue_client.get("/internal/calls/pbx-acme/old-2/result", headers=bearer())).status == 401
        assert (await blue_client.get("/v1/calls/old-1/result", headers=bearer())).status == 404  # no loop back
    pair(check)


def test_without_a_running_peer_the_instance_is_ready_at_once_and_answers_404():
    async def check(client, runtime):
        await until_ready(client, timeout=3)
        started = time.monotonic()
        assert (await client.get("/v1/calls/x/result?wait=1", headers=bearer())).status == 404
        assert time.monotonic() - started < 2.5
        assert (await client.delete("/v1/calls/x/result", headers=bearer())).status == 204
    run(check, peer_url=f"http://127.0.0.1:{free_port()}")


def test_peer_needs_the_operator_token():
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_PEER_URL": "http://runtime-blue:8443"})
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_PEER_URL": "runtime-blue:8443", "VA_ADMIN_TOKEN": "t"})
    s = Settings.from_env({"LLM_API_KEY": "k", "VA_PEER_URL": "http://runtime-blue:8443/", "VA_ADMIN_TOKEN": "t"})
    assert s.peer_url == "http://runtime-blue:8443" and Settings.from_env({"LLM_API_KEY": "k"}).peer_url == ""


def test_peer_client_does_not_send_pbx_credentials():
    peer = Peer("http://x/", ADMIN)
    assert peer.url == "http://x" and peer._headers == {"Authorization": f"Bearer {ADMIN}"}
    assert peer._call_path("pbx-acme", "a/b") == "http://x/internal/calls/pbx-acme/a%2Fb/result"


def test_cli_drain_waits_for_calls_then_results(capsys):
    answers = iter([{"active_calls": 2, "pending_results": 1}, {"active_calls": 1, "pending_results": 2},
                    {"active_calls": 0, "pending_results": 1}, {"active_calls": 0, "pending_results": 0}])
    calls = []

    def fake(url, method="GET", token=""):
        calls.append((url.rsplit("/", 1)[-1], method, token))
        return next(answers)
    env = {"LLM_API_KEY": "k", "VA_ADMIN_TOKEN": "adm"}
    with patch.dict("os.environ", env, clear=True), patch.object(cli, "_request", fake), patch.object(cli.time, "sleep"):
        assert cli.main(["drain"]) == 0
    assert calls[0] == ("drain", "POST", "adm") and all(c == ("health", "GET", "adm") for c in calls[1:])
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"active_calls": 0, "pending_results": 0}


def test_cli_drain_gives_up_after_the_limits():
    clock = iter(range(0, 10000, 10))
    stuck = lambda url, method="GET", token="": {"active_calls": 1, "pending_results": 0}  # noqa: E731
    env = {"LLM_API_KEY": "k", "VA_ADMIN_TOKEN": "adm"}
    with patch.dict("os.environ", env, clear=True), patch.object(cli, "_request", stuck), \
            patch.object(cli.time, "sleep"), patch.object(cli.time, "monotonic", lambda: next(clock)):
        assert cli.main(["drain", "--wait", "30"]) == 3
    with patch.dict("os.environ", {"LLM_API_KEY": "k"}, clear=True):
        assert cli.main(["drain"]) == 2  # no operator token


def test_cli_ready_check():
    def unavailable(url, method="GET", token=""):
        raise urllib.error.HTTPError(url, 503, "starting", {}, None)
    with patch.object(cli, "_request", unavailable):
        assert cli.main(["healthcheck", "--ready"]) == 1
    with patch.object(cli, "_request", lambda url, method="GET", token="": {"status": "ready"}):
        assert cli.main(["healthcheck", "--ready"]) == 0


def test_cli_keeps_the_server_drain_for_sigterm():
    from voice_agent_runtime import server
    assert cli.drain is server.drain  # the CLI command is drain_calls: run_server must still drain the server


def test_cli_status(capsys):
    health = {"active_calls": 1, "pending_results": 2, "ready": True, "status": "ok", "version": "0.2.0",
              "boot_id": "b", "bundles": 3, "tenants": {"secret": 1}}
    with patch.dict("os.environ", {"LLM_API_KEY": "k", "VA_ADMIN_TOKEN": "adm"}, clear=True), \
            patch.object(cli, "_request", lambda url, method="GET", token="": health):
        assert cli.main(["status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {k: v for k, v in health.items() if k != "tenants"}


def test_php_module_notices_a_new_instance(tmp_path):
    from test_server import run_php

    boot = {}
    script = r"""require $argv[1];
use Modules\ModuleVoiceAgent\Lib\Services\{RuntimeClient, SyncService};
$c = new RuntimeClient($argv[2], 'pbx-acme', 's3cret-token', false);
$h = $c->health()['body'];
echo json_encode([$h['boot_id'], in_array('handoff', $h['capabilities'], true),
    SyncService::runtimeChanged($c, $h['boot_id']), SyncService::runtimeChanged($c, 'previous-instance')]);"""
    out = json.loads(run_php(tmp_path, script, lambda runtime: boot.setdefault("id", runtime.boot_id)))
    assert out == [boot["id"], True, False, True]
