"""LLM streaming against a local OpenAI-compatible HTTP server (real sockets, scripted answers)."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from voice_agent_runtime.config import LlmProvider
from voice_agent_runtime.llm import LlmError, LlmSession, ProviderRouter, parse_sse


def sse(*chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def text_chunk(s):
    return {"choices": [{"delta": {"content": s}}]}


TOOL_STREAM = sse(
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "transfer_call", "arguments": '{"ext'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ension":"210"}'}}]}}]},
    {"usage": {"prompt_tokens": 2000, "completion_tokens": 20}, "choices": []},
)


def test_parse_sse_text_and_tool():
    assert list(parse_sse(io.BufferedReader(io.BytesIO(sse(text_chunk("Да, "), text_chunk("работаем.")))))) == [
        ("text", "Да, "), ("text", "работаем.")]
    events = list(parse_sse(io.BufferedReader(io.BytesIO(TOOL_STREAM))))
    assert events == [("usage", {"prompt_tokens": 2000, "completion_tokens": 20}),
                      ("tool", ("transfer_call", '{"extension":"210"}'))]


def test_parse_sse_error_event():
    with pytest.raises(LlmError):
        list(parse_sse(io.BufferedReader(io.BytesIO(sse({"error": {"message": "overloaded"}})))))


class Scripted(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    delay = 0.0
    status = 200
    body = sse(text_chunk("Ответ."))
    check_body: bytes | None = None  # the answer to topic filter requests (None: the same as the others)
    requests: list = []

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.headers.get("Authorization"), payload))
        time.sleep(self.delay)
        is_check = str((payload.get("messages") or [{}])[0].get("content")).startswith("Ты фильтр")
        body = self.check_body if is_check and self.check_body is not None else self.body
        self.send_response(self.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def server(**attrs):
    handler = type("H", (Scripted,), {**attrs, "requests": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler


def provider(srv, name):
    return LlmProvider(name, f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", "model-" + name, "key-" + name,
                       {"reasoning_effort": "low"}, timeout=5)


def collect(session, cancelled=None):
    return list(session.stream([{"role": "user", "content": "привет"}], [], cancelled or threading.Event()))


def test_request_body_and_reused_connection():
    srv, handler = server()
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        assert collect(session) == [("text", "Ответ.")]
        assert collect(session) == [("text", "Ответ.")]
        auth, body = handler.requests[0]
        assert auth == "Bearer key-primary"
        assert body["stream"] is True and body["tool_choice"] == "auto" and body["temperature"] == 0.2
        assert body["reasoning_effort"] == "low" and body["model"] == "model-primary"
        assert session.last_provider == "primary"
    finally:
        srv.shutdown()


def test_hedge_to_secondary_when_primary_is_slow():
    slow, _ = server(delay=1.5, body=sse(text_chunk("медленный")))
    fast, _ = server(body=sse(text_chunk("быстрый")))
    try:
        reasons = []
        session = LlmSession(provider(slow, "primary"), provider(fast, "secondary"), 300, 0.2, 400)
        started = time.monotonic()
        events = list(session.stream([], [], threading.Event(), on_hedge=reasons.append))
        assert events == [("text", "быстрый")] and session.last_provider == "secondary"
        assert reasons == ["slow"] and time.monotonic() - started < 1.2
    finally:
        slow.shutdown()
        fast.shutdown()


def test_failover_on_error():
    broken, _ = server(status=500, body=b"boom")
    ok, _ = server(body=sse(text_chunk("работает")))
    try:
        reasons = []
        session = LlmSession(provider(broken, "primary"), provider(ok, "secondary"), 5000, 0.2, 400)
        assert list(session.stream([], [], threading.Event(), on_hedge=reasons.append)) == [("text", "работает")]
        assert reasons == ["error"]
    finally:
        broken.shutdown()
        ok.shutdown()


def test_all_providers_failing_raises():
    broken, _ = server(status=503, body=b"down")
    try:
        with pytest.raises(LlmError, match="HTTP 503"):
            collect(LlmSession(provider(broken, "primary"), None, 600, 0.2, 400))
    finally:
        broken.shutdown()


class Sequenced(Scripted):
    """Answers the n-th request with ``bodies[n]`` (the last one repeats)."""
    bodies: list = []

    def do_POST(self):
        handler = type(self)
        handler.body = handler.bodies[min(len(handler.requests), len(handler.bodies) - 1)]
        super().do_POST()


def sequenced(*bodies):
    handler = type("H", (Sequenced,), {"bodies": list(bodies), "requests": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, handler


HARMONY_ERROR = sse({"error": {"message": "unexpected tokens remaining in message header", "code": 500}})


def test_a_failed_turn_is_asked_once_more_before_giving_up():
    srv, handler = sequenced(HARMONY_ERROR, sse(text_chunk("Со второго раза.")))
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        assert collect(session) == [("text", "Со второго раза.")]
        assert len(handler.requests) == 2 and session.last_provider == "primary"
    finally:
        srv.shutdown()


def test_a_second_failure_or_a_failure_after_speech_is_not_retried():
    srv, handler = sequenced(HARMONY_ERROR)
    try:
        with pytest.raises(LlmError, match="unexpected tokens"):
            collect(LlmSession(provider(srv, "primary"), None, 600, 0.2, 400))
        assert len(handler.requests) == 2
    finally:
        srv.shutdown()
    half = sse(text_chunk("Начало ответа, ")).removesuffix(b"data: [DONE]\n\n") + HARMONY_ERROR
    srv, handler = sequenced(half, sse(text_chunk("повтор")))
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        events = []
        with pytest.raises(LlmError):
            for event in session.stream([], [], threading.Event()):
                events.append(event)
        assert events == [("text", "Начало ответа, ")] and len(handler.requests) == 1
    finally:
        srv.shutdown()


def test_cancelled_turn_drops_the_connection_and_the_next_request_works():
    srv, handler = server(delay=0.5)
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        cancelled = threading.Event()
        threading.Timer(0.1, cancelled.set).start()
        assert collect(session, cancelled) == []
        handler.delay = 0.0
        assert collect(session) == [("text", "Ответ.")]
    finally:
        srv.shutdown()


def test_usage_from_stream_and_estimate_without_it():
    reported = sse(text_chunk("Да."), {"choices": [], "usage": {"prompt_tokens": 1800, "completion_tokens": 7}})
    srv, handler = server(body=reported)
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        assert collect(session) == [("text", "Да.")]
        assert handler.requests[0][1]["stream_options"] == {"include_usage": True}
        assert (session.usage.requests, session.usage.input_tokens, session.usage.output_tokens,
                session.usage.estimated_requests) == (1, 1800, 7, 0)
        handler.body = sse(text_chunk("Ответ из шестнадцати симв"))  # no usage chunk: estimated
        collect(session)
        assert session.usage.requests == 2 and session.usage.estimated_requests == 1
        assert session.usage.output_tokens == 7 + 8 and session.usage.input_tokens > 1800
    finally:
        srv.shutdown()


class RejectsStreamOptions(Scripted):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.headers.get("Authorization"), payload))
        body, status = (b'{"error":"unknown field stream_options"}', 400) if "stream_options" in payload else (
            self.body, 200)
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_provider_rejecting_stream_options_is_asked_again_without_them():
    handler = type("R", (RejectsStreamOptions,), {"requests": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        prov = LlmProvider("primary", f"http://127.0.0.1:{srv.server_port}/v1/rejects", "model-r", "k", timeout=5)
        session = LlmSession(prov, None, 600, 0.2, 400)
        assert collect(session) == [("text", "Ответ.")]
        assert collect(session) == [("text", "Ответ.")]
        assert ["stream_options" in body for _, body in handler.requests] == [True, False, False]
        assert session.usage.requests == 2 and session.usage.estimated_requests == 2
    finally:
        srv.shutdown()


def test_stream_usage_can_be_turned_off():
    srv, handler = server()
    try:
        prov = LlmProvider("primary", f"http://127.0.0.1:{srv.server_port}/v1/off", "m", "k", stream_usage=False)
        collect(LlmSession(prov, None, 600, 0.2, 400))
        assert "stream_options" not in handler.requests[0][1]
    finally:
        srv.shutdown()


def router_pair(primary_attrs=None, secondary_attrs=None, **router_kw):
    """A primary and a secondary scripted server, a router over them and a session that uses it."""
    p_srv, p_handler = server(**{"body": sse(text_chunk("основной")), **(primary_attrs or {})})
    s_srv, s_handler = server(**{"body": sse(text_chunk("резерв")), **(secondary_attrs or {})})
    kw = {"failures": 3, "window_s": 60.0, "probe_s": 3600.0, "probes": 3, "probe_gap_s": 0.0, **router_kw}
    router = ProviderRouter(provider(p_srv, "primary"), provider(s_srv, "secondary"), 0.3, **kw)
    session = LlmSession(provider(p_srv, "primary"), provider(s_srv, "secondary"), 300, 0.2, 400, router=router)
    return session, router, (p_srv, p_handler), (s_srv, s_handler)


def shutdown(router, *servers):
    router.close()
    for srv, _ in servers:
        srv.shutdown()


def test_router_fails_over_after_slow_turns_and_comes_back_after_probes():
    session, router, primary, secondary = router_pair({"delay": 1.0})
    try:
        for _ in range(2):
            assert collect(session) == [("text", "резерв")]  # each slow turn is answered by the hedge
        assert router.active == 0
        assert collect(session) == [("text", "резерв")]
        assert router.active == 1 and router.status()["active"] == "secondary" and router.status()["next_probe"]
        asked = len(primary[1].requests)
        assert collect(session) == [("text", "резерв")]
        assert len(primary[1].requests) == asked  # the secondary answered in time: the primary is not asked
        primary[1].delay = 0.0  # the primary recovers
        assert router.probe_now() is True
        assert len(primary[1].requests) == asked + 3 and router.active == 0 and router.switches == 2
        assert collect(session) == [("text", "основной")]
    finally:
        shutdown(router, primary, secondary)


def test_router_counts_errors_and_a_good_turn_resets_the_count():
    session, router, primary, secondary = router_pair({"status": 500, "body": b"boom"})
    try:
        collect(session), collect(session)
        primary[1].status, primary[1].body = 200, sse(text_chunk("основной"))
        assert collect(session) == [("text", "основной")]  # a good turn: the failures before do not add up
        primary[1].status, primary[1].body = 500, b"boom"
        collect(session), collect(session)
        assert router.active == 0
        assert collect(session) == [("text", "резерв")]
        assert router.active == 1
    finally:
        shutdown(router, primary, secondary)


def test_router_failures_outside_the_window_do_not_count():
    session, router, primary, secondary = router_pair({"status": 500, "body": b"boom"}, window_s=0.3)
    try:
        collect(session), collect(session)
        time.sleep(0.4)
        collect(session)
        assert router.active == 0
    finally:
        shutdown(router, primary, secondary)


def test_router_failed_probe_keeps_the_secondary():
    session, router, primary, secondary = router_pair({"status": 503, "body": b"down"})
    try:
        for _ in range(3):
            collect(session)
        assert router.active == 1
        assert router.probe_now() is False and router.active == 1
        assert len(primary[1].requests) == 4  # 3 turns and the first probe; a failed probe ends the series
    finally:
        shutdown(router, primary, secondary)


def test_router_hands_back_to_the_primary_when_the_secondary_fails_too():
    session, router, primary, secondary = router_pair({"delay": 1.0})
    try:
        for _ in range(3):
            collect(session)
        assert router.active == 1
        primary[1].delay, secondary[1].delay = 0.0, 1.0
        for _ in range(3):
            assert collect(session) == [("text", "основной")]
        assert router.active == 0 and router.status()["next_probe"] is None
    finally:
        shutdown(router, primary, secondary)


def test_router_probes_by_itself_and_side_checks_follow_the_active_provider():
    session, router, primary, secondary = router_pair({"status": 500, "body": b"boom"}, probe_s=0.3)
    try:
        for _ in range(3):
            collect(session)
        assert router.active == 1
        check = session.check("Ты фильтр.", "x y z", 50)
        check.poll(5.0)
        assert check.text == "резерв"  # the topic filter asks the provider turns ask first
        primary[1].status, primary[1].body = 200, sse(text_chunk("да"))
        deadline = time.monotonic() + 5
        while router.active != 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert router.active == 0
    finally:
        shutdown(router, primary, secondary)


def test_no_router_without_a_secondary():
    srv, _ = server()
    try:
        p = provider(srv, "primary")
        session = LlmSession(p, None, 600, 0.2, 400, router=ProviderRouter(p, p, 0.3))
        assert session.router is None and collect(session) == [("text", "Ответ.")]
    finally:
        srv.shutdown()
