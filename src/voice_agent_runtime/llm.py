"""OpenAI-compatible chat completions with streaming and tools, one warm connection per provider per call.

* Each call keeps its own HTTP connection(s) open between turns (TLS setup is not on the critical path).
* A turn cancelled mid-stream leaves an unread response on the connection; such a connection is closed,
  otherwise the next request on it fails (``ResponseNotReady``). http.client reconnects on the next request.
* Hedging: when a secondary provider is configured, it is asked as well if the primary gives no first
  token within ``hedge_ms`` or fails; the first provider to answer wins and the other request is dropped.
* Failover (``ProviderRouter``, one per process): after a few such slow or failed turns in a row the other
  provider is asked first by every call; the primary is probed now and then and gets the turns back once
  it answers in time again.
* Retry: a turn whose request failed before anything was yielded (e.g. a provider 500 on a malformed model
  message) is asked once more of the first provider instead of handing the caller to a person at once.
* Usage: requests ask for token usage in the stream (``stream_options.include_usage``). A provider that
  rejects the option (HTTP 400/422 naming it) is asked again without it and remembered for the process.
  Every request the provider accepted is counted, including dropped speculative and hedged ones; without
  reported usage the tokens are estimated (characters / 3.2) and the request is counted as estimated.
* Checks: a short side request (the topic filter, ``check``) goes to the provider asked first over a connection
  of its own, in parallel with the turn's main stream; its tokens count in the call's usage.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import queue
import ssl
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import LlmProvider

POLL_S = 0.05
CHARS_PER_TOKEN = 3.2  # the same estimate as prompt.py
WARM_CONNECT_TIMEOUT_S = 3.0
log = logging.getLogger(__name__)
_NO_STREAM_USAGE: set[tuple[str, str]] = set()  # (url, model) of providers that reject stream_options
_ssl_lock = threading.Lock()
_ssl_context: ssl.SSLContext | None = None


def default_ssl_context() -> ssl.SSLContext:
    """One CA store for the process: loading it per call cost 10-30 ms on the event loop."""
    global _ssl_context
    with _ssl_lock:
        if _ssl_context is None:
            _ssl_context = ssl.create_default_context()
        return _ssl_context


@dataclass
class LlmUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_requests: int = 0


def _estimate(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


class LlmError(RuntimeError):
    pass


class ProviderConnection:
    """A persistent connection to one provider. Used by one request at a time."""

    def __init__(self, provider: LlmProvider):
        self.provider = provider
        self.url = urlsplit(provider.url)
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()  # a warm-up in progress and the first request share self._conn

    def _new_connection(self, timeout: float) -> http.client.HTTPConnection:
        if self.url.scheme == "https":
            return http.client.HTTPSConnection(self.url.netloc, context=default_ssl_context(), timeout=timeout)
        return http.client.HTTPConnection(self.url.netloc, timeout=timeout)

    def _connection(self) -> http.client.HTTPConnection:
        with self._lock:
            if self._conn is None:
                self._conn = self._new_connection(self.provider.timeout)
            return self._conn

    def warm(self) -> None:
        """Opens the connection ahead of the first request (TLS handshake off the first turn). Bounded by a
        short connect timeout so that a dead provider does not hold the lock, and the first request, for the
        whole request timeout."""
        with self._lock:
            if self._conn is not None:
                return
            conn = self._new_connection(min(WARM_CONNECT_TIMEOUT_S, self.provider.timeout))
            try:
                conn.connect()
            except OSError:
                conn.close()
                return
            conn.timeout = self.provider.timeout
            if conn.sock is not None:
                conn.sock.settimeout(self.provider.timeout)
            self._conn = conn

    def body(self, payload: dict) -> bytes:
        p = self.provider
        if p.stream_usage and (p.url, p.model) not in _NO_STREAM_USAGE:
            payload = {"stream_options": {"include_usage": True}, **payload}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def open(self, payload: dict) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, int]:
        """Sends the request; returns the connection, the streaming response and the request size in chars."""
        headers = {"Authorization": f"Bearer {self.provider.api_key}", "Content-Type": "application/json",
                   "Accept": "text/event-stream"}
        path = self.url.path or "/"
        stale_retry = usage_retry = True
        while True:
            body = self.body(payload)
            conn = self._connection()
            try:
                conn.request("POST", path, body, headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException):
                self.release(conn)  # a stale keep-alive connection: reconnect once
                if not stale_retry:
                    raise
                stale_retry = False
                continue
            if resp.status == 200:
                return conn, resp, len(body.decode("utf-8"))
            detail = resp.read(300).decode("utf-8", "replace")
            self.release(conn)
            if usage_retry and resp.status in (400, 422) and "stream_options" in detail and b"stream_options" in body:
                usage_retry = False
                _NO_STREAM_USAGE.add((self.provider.url, self.provider.model))
                log.warning("%s rejects stream_options: token usage will be estimated (set %s_STREAM_USAGE=false)",
                            self.provider.name, "LLM" if self.provider.name == "primary" else "LLM2")
                continue
            raise LlmError(f"{self.provider.name} HTTP {resp.status}: {detail}")

    def release(self, conn: http.client.HTTPConnection) -> None:
        """Closes a connection that may still carry an unread response; the next request opens a new one."""
        with self._lock:
            if self._conn is conn:
                self._conn = None
        conn.close()

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def parse_sse(resp) -> Iterator[tuple[str, object]]:
    """Yields ("text", str), ("usage", dict) and finally ("tool", (name, args)) for the first tool call."""
    tools: dict[int, list[str]] = {}
    buf = b""
    while part := resp.read1(65536):
        buf += part
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise LlmError(f"stream error: {str(chunk['error'])[:200]}")
            if chunk.get("usage"):
                yield "usage", chunk["usage"]
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                slot = tools.setdefault(int(tc.get("index", 0) or 0), ["", ""])
                slot[0] = fn.get("name") or slot[0]
                slot[1] += fn.get("arguments") or ""
            if delta.get("content"):
                yield "text", delta["content"]
    if tools:
        name, args = tools[min(tools)]
        if name:
            yield "tool", (name, args)


class _Attempt(threading.Thread):
    def __init__(self, index: int, conn: ProviderConnection, payload: dict, out: queue.Queue):
        super().__init__(daemon=True, name=f"llm-{conn.provider.name}")
        self.index, self.conn, self.payload, self.out = index, conn, payload, out
        self.cancelled = threading.Event()
        self.finished = False
        self.http: http.client.HTTPConnection | None = None
        self.request_chars = 0  # > 0 once the provider accepted the request
        self.output_chars = 0
        self.usage: dict | None = None

    def run(self) -> None:
        try:
            self.http, resp, self.request_chars = self.conn.open(self.payload)
            if self.cancelled.is_set():
                return
            for kind, payload in parse_sse(resp):
                if kind == "usage":
                    self.usage = payload  # type: ignore[assignment]
                elif kind == "text":
                    self.output_chars += len(payload)  # type: ignore[arg-type]
                elif kind == "tool":
                    self.output_chars += sum(map(len, payload))  # type: ignore[arg-type]
                if self.cancelled.is_set():
                    return
                self.out.put((self.index, kind, payload))
            self.finished = True
            self.out.put((self.index, "done", None))
        except Exception as exc:  # noqa: BLE001 - reported to the consumer
            if not self.cancelled.is_set():
                self.out.put((self.index, "error", exc))
        finally:
            if not self.finished and self.http is not None:
                self.conn.release(self.http)  # unread response or broken connection

    def cancel(self) -> None:
        self.cancelled.set()
        if self.http is not None:
            self.conn.release(self.http)


class Check:
    """A short side request in flight; ``result`` gives its text, or None on an error or a timeout."""

    def __init__(self, session: LlmSession, attempt: _Attempt, out: queue.Queue):
        self.session, self.attempt, self.out = session, attempt, out
        self.text: str | None = None
        self.settled = False
        self.started = time.monotonic()
        self.elapsed_ms = 0

    def poll(self, timeout: float = 0.0) -> bool:
        """Waits up to ``timeout`` for the answer; True once it is settled (answered, failed or given up)."""
        deadline = time.monotonic() + timeout
        parts: list[str] = []
        while not self.settled:
            try:
                _, kind, payload = self.out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return False
            if kind == "text":
                parts.append(payload)
            elif kind == "done":
                self.text = "".join(parts)
                self._settle()
            elif kind == "error":
                log.info("check failed: %s", payload)
                self._settle()
        return True

    def give_up(self) -> None:
        if not self.settled:
            self.attempt.cancel()
            self._settle()

    def _settle(self) -> None:
        self.settled = True
        self.elapsed_ms = round((time.monotonic() - self.started) * 1000)
        self.session._account(self.attempt)


class ProviderRouter:
    """Which provider a turn asks first, shared by all calls of the process.

    The provider asked first fails a turn when it gives no first token within the hedge time or returns an
    error before answering. ``failures`` such turns within ``window_s`` (with no good turn in between) make
    the other provider the one asked first, for every call. While the secondary is asked first, the primary
    is probed every ``probe_s``: ``probes`` short requests in a row, each answered within the hedge time,
    bring it back. A secondary that fails the same way hands the turns back to the primary at once.
    """

    PROBE_TEXT = "Ответь одним словом: да."
    PROBE_MAX_TOKENS = 32

    def __init__(self, primary: LlmProvider, secondary: LlmProvider, limit_s: float, failures: int = 3,
                 window_s: float = 300.0, probe_s: float = 28800.0, probes: int = 3, probe_gap_s: float = 5.0):
        self.providers = (primary, secondary)
        self.limit_s = limit_s if limit_s > 0 else primary.timeout
        self.failures, self.window_s = failures, window_s
        self.probe_s, self.probes, self.probe_gap_s = probe_s, probes, probe_gap_s
        self.active = 0
        self.since = time.time()
        self.switches = 0
        self.next_probe: float | None = None  # wall time of the next probe of the primary
        self._failed: list[float] = []  # monotonic times of the active provider's failures in a row
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prober: threading.Thread | None = None

    def order(self) -> tuple[int, int]:
        active = self.active
        return active, 1 - active

    def report(self, index: int, ok: bool) -> None:
        """The outcome of a turn for the provider it asked first."""
        with self._lock:
            if index != self.active:
                return  # asked first under the previous order: says nothing about the current one
            if ok:
                self._failed.clear()
                return
            now = time.monotonic()
            self._failed = [t for t in self._failed if now - t <= self.window_s] + [now]
            if len(self._failed) >= self.failures:
                self._switch(1 - self.active, f"{len(self._failed)} slow or failed turns in a row")

    def _switch(self, index: int, reason: str) -> None:
        old = self.providers[self.active]
        self.active, self._failed, self.since = index, [], time.time()
        self.switches += 1
        new = self.providers[index]
        log.warning("llm provider switched", extra={"fields": {"from": old.name, "to": new.name, "model": new.model,
                                                                "reason": reason}})
        if index == 0:
            self.next_probe = None
        else:
            self.next_probe = time.time() + self.probe_s
            if self._prober is None or not self._prober.is_alive():
                self._prober = threading.Thread(target=self._probe_loop, daemon=True, name="llm-probe")
                self._prober.start()

    def _probe_loop(self) -> None:
        while not self._stop.wait(self.probe_s):
            if self.active == 0 or self.probe_now():
                return
            with self._lock:
                self.next_probe = time.time() + self.probe_s

    def probe_now(self) -> bool:
        """Probes the primary; True (and the primary asked first again) when every probe answered in time."""
        for i in range(self.probes):
            if i and self._stop.wait(self.probe_gap_s):
                return False
            if not self._probe_once(self.providers[0]):
                log.info("llm probe failed", extra={"fields": {"provider": self.providers[0].name}})
                return False
        with self._lock:
            if self.active != 0:
                self._switch(0, f"{self.probes} probes answered in time")
        return True

    def _probe_once(self, provider: LlmProvider) -> bool:
        conn = ProviderConnection(provider)
        payload = {"model": provider.model, "stream": True, "max_tokens": self.PROBE_MAX_TOKENS, "temperature": 0,
                   "messages": [{"role": "user", "content": self.PROBE_TEXT}], **provider.extra}
        out: queue.Queue = queue.Queue()
        attempt = _Attempt(0, conn, payload, out)
        attempt.start()
        try:
            _, kind, _ = out.get(timeout=self.limit_s)
            return kind != "error"
        except queue.Empty:
            return False
        finally:
            attempt.cancel()
            conn.close()

    def status(self) -> dict:
        p = self.providers[self.active]
        iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None  # noqa: E731
        return {"active": p.name, "model": p.model, "since": iso(self.since), "switches": self.switches,
                "next_probe": iso(self.next_probe)}

    def close(self) -> None:
        self._stop.set()


class LlmSession:
    """The LLM side of one call."""

    def __init__(self, primary: LlmProvider, secondary: LlmProvider | None, hedge_ms: int,
                 temperature: float, max_tokens: int, router: ProviderRouter | None = None):
        self.conns = [ProviderConnection(primary)] + ([ProviderConnection(secondary)] if secondary else [])
        # side requests never queue behind the main stream: a connection of their own per provider
        self.check_conns = [ProviderConnection(primary)] + ([ProviderConnection(secondary)] if secondary else [])
        self.router = router if secondary else None
        self.hedge_s = hedge_ms / 1000
        self.temperature, self.max_tokens = temperature, max_tokens
        self.last_provider = ""
        self.usage = LlmUsage()

    def _order(self) -> tuple[int, int]:
        return self.router.order() if self.router is not None else (0, 1)

    @property
    def check_conn(self) -> ProviderConnection:
        """Side requests go to the provider turns ask first."""
        return self.check_conns[self._order()[0]]

    def warm(self, checks: bool = False) -> None:
        for conn in self.conns + ([self.check_conn] if checks else []):
            conn.warm()

    def close(self) -> None:
        for conn in self.conns + self.check_conns:
            conn.close()

    def check(self, system: str, text: str, max_tokens: int) -> Check:
        """Starts a short request (temperature 0, no tools) and returns at once."""
        p = self.check_conn.provider
        payload = {"model": p.model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}], **p.extra}
        out: queue.Queue = queue.Queue()
        attempt = _Attempt(0, self.check_conn, payload, out)
        attempt.start()
        return Check(self, attempt, out)

    def _payload(self, conn: ProviderConnection, messages: list[dict], tools: list[dict]) -> dict:
        p = conn.provider
        return {"model": p.model, "stream": True, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "tools": tools, "tool_choice": "auto", "messages": messages, **p.extra}

    def _account(self, attempt: _Attempt) -> None:
        """Adds what one request cost; called when the turn is done with it (finished or dropped)."""
        if not attempt.request_chars:
            return  # refused or never sent: nothing was consumed
        usage = self.usage
        usage.requests += 1
        reported = attempt.usage or {}
        prompt = reported.get("prompt_tokens", reported.get("input_tokens"))
        completion = reported.get("completion_tokens", reported.get("output_tokens"))
        if isinstance(prompt, int) and isinstance(completion, int):
            usage.input_tokens += prompt
            usage.output_tokens += completion
        else:
            usage.estimated_requests += 1
            usage.input_tokens += _estimate(attempt.request_chars)
            usage.output_tokens += _estimate(attempt.output_chars)

    def stream(self, messages: list[dict], tools: list[dict], cancelled: threading.Event,
               on_hedge=None) -> Iterator[tuple[str, object]]:
        """Yields ("text", str) and ("tool", (name, args)) from the winning provider."""
        out: queue.Queue = queue.Queue()
        attempts: list[_Attempt] = []
        order = self._order()  # provider indexes: the one asked first, then the hedge
        first_conn = self.conns[order[0]]
        reported = False

        def start(i: int) -> None:  # i: 0 = the provider asked first, 1 = the other; the attempt's index is
            conn = self.conns[order[i]]  # its place in ``attempts``
            attempt = _Attempt(len(attempts), conn, self._payload(conn, messages, tools), out)
            attempts.append(attempt)
            attempt.start()

        def report(ok: bool) -> None:  # once per turn: how the provider asked first did
            nonlocal reported
            if self.router is not None and not reported:
                reported = True
                self.router.report(order[0], ok)

        start(0)
        started_at, winner, errors = time.monotonic(), None, []
        yielded = retried = False
        try:
            while True:
                if cancelled.is_set():
                    return
                if (winner is None and len(attempts) == 1 and len(self.conns) > 1
                        and time.monotonic() - started_at >= self.hedge_s):
                    start(1)
                    if on_hedge:
                        on_hedge("slow")
                    if self.hedge_s > 0:  # with no hedge time both are always asked: nobody is "slow"
                        report(False)
                try:
                    index, kind, payload = out.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                if winner is not None and index != winner:
                    continue
                if kind == "error":
                    if not yielded and attempts[index].conn is first_conn:
                        report(False)
                    errors.append(payload)
                    if winner is not None and yielded:
                        raise LlmError(str(payload)) from payload
                    if winner is not None or (len(errors) >= len(attempts) and not
                                              (len(attempts) == 1 and len(self.conns) > 1)):
                        if retried:
                            raise LlmError("; ".join(str(e) for e in errors))
                        log.info("llm retry after: %s", payload)
                        retried, winner = True, None
                        start(0)
                    elif len(attempts) == 1 and len(self.conns) > 1:
                        start(1)
                        if on_hedge:
                            on_hedge("error")
                    continue
                if winner is None:
                    winner = index
                    self.last_provider = attempts[index].conn.provider.name
                    if attempts[index].conn is first_conn:
                        report(True)
                    for other in attempts:
                        if other.index != index:
                            other.cancel()
                if kind == "done":
                    return
                if kind == "usage":
                    continue
                yielded = True
                yield kind, payload
        finally:
            for attempt in attempts:
                if not attempt.finished:
                    attempt.cancel()
                self._account(attempt)
