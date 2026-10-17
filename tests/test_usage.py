"""Usage metering: aggregate counters per day/tenant/agent, SQLite persistence, /v1/usage and /metrics."""

import asyncio
import sqlite3
from datetime import UTC, date, datetime

from aiohttp.test_utils import TestClient, TestServer

from test_server import OTHER, _media_call, admin, basic, bearer, make_runtime
from voice_agent_runtime.server import build_app
from voice_agent_runtime.usage import COUNTERS, UsageMeter

CALL = {"calls": 1, "connected_seconds": 30.5, "agent_speech_seconds": 12.25, "caller_turns": 3, "llm_requests": 4,
        "llm_input_tokens": 6000, "llm_output_tokens": 80, "llm_estimated_requests": 1, "tts_characters": 140,
        "fallbacks": 0}


def test_counters_add_up_and_survive_a_restart(tmp_path):
    path = tmp_path / "data" / "usage.sqlite3"
    meter = UsageMeter(path)
    meter.record("2026-10-01", "pbx-a", "3", CALL)
    meter.flush()
    meter.record("2026-10-01", "pbx-a", "3", CALL)  # still pending: reads include it
    meter.record("2026-10-02", "pbx-a", "4", {"calls": 1, "fallbacks": 1})
    meter.record("2026-10-01", "pbx-b", "3", {"calls": 7})
    report = meter.query("pbx-a", date(2026, 10, 1), date(2026, 10, 31))
    assert report["total"]["calls"] == 3 and report["total"]["connected_seconds"] == 61.0
    assert report["agents"]["3"]["llm_input_tokens"] == 12000 and report["agents"]["4"]["fallbacks"] == 1
    assert [(d["date"], d["agent_id"]) for d in report["days"]] == [("2026-10-01", "3"), ("2026-10-02", "4")]
    assert meter.query("pbx-a", date(2026, 10, 2), date(2026, 10, 2))["total"]["calls"] == 1
    meter.flush()
    reopened = UsageMeter(path)
    assert reopened.totals() == meter.totals()
    assert reopened.totals("pbx-b") == {"pbx-b": {**dict.fromkeys(COUNTERS, 0), "calls": 7}}


def test_only_aggregates_are_stored(tmp_path):
    path = tmp_path / "usage.sqlite3"
    meter = UsageMeter(path)
    meter.record("2026-10-01", "pbx-a", "3", CALL)
    meter.flush()
    with sqlite3.connect(path) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(usage)")]
    assert columns == ["day", "tenant", "agent", *COUNTERS]


def test_unknown_counter_is_rejected():
    try:
        UsageMeter(None).record("2026-10-01", "pbx-a", "3", {"caller_number": 1})
    except ValueError:
        return
    raise AssertionError("unknown counter accepted")


def run(check, tmp_path):
    async def main():
        runtime = make_runtime()
        runtime.usage = UsageMeter(tmp_path / "usage.sqlite3")
        async with TestClient(TestServer(build_app(runtime))) as client:
            await check(client, runtime)
    asyncio.run(main())


def test_call_is_metered_and_reported_to_its_tenant_only(tmp_path):
    async def check(client, runtime):
        await _media_call(client, "/media?callerid=79991234567&callid=u-1&agent=3", basic())  # no bundle: fallback
        await client.get("/v1/calls/u-1/result?wait=1", headers=bearer())
        await asyncio.sleep(0.05)
        today = datetime.now(UTC).date().isoformat()
        assert (await client.get("/v1/usage")).status == 401
        report = await (await client.get(f"/v1/usage?from={today}&to={today}", headers=bearer())).json()
        total = report["total"]
        assert report["tenant"] == "pbx-acme" and total["calls"] == 1 and total["fallbacks"] == 1
        assert total["agent_speech_seconds"] > 0 and total["tts_characters"] > 0 and total["llm_requests"] == 0
        assert report["days"][0]["agent_id"] == "3" and "79991234567" not in str(report)
        other = await (await client.get("/v1/usage", headers=bearer("pbx-other", OTHER))).json()
        assert other["total"]["calls"] == 0 and other["days"] == []
        text = await (await client.get("/metrics", headers=admin())).text()
        assert 'voice_agent_usage_calls_total{tenant="pbx-acme"} 1' in text
        assert 'voice_agent_usage_fallbacks_total{tenant="pbx-acme"} 1' in text
    run(check, tmp_path)


def test_usage_dates_are_validated(tmp_path):
    async def check(client, runtime):
        for query in ("from=2026-13-01", "to=20261001", "from=2026-10-02&to=2026-10-01", "from=2024-01-01&to=2026-01-01"):
            assert (await client.get(f"/v1/usage?{query}", headers=bearer())).status == 400, query
        resp = await client.get("/v1/usage?from=2026-10-01&to=2026-10-31", headers=bearer())
        assert resp.status == 200 and (await resp.json())["from"] == "2026-10-01"
    run(check, tmp_path)
