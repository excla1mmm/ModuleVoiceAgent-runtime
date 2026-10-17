"""OpenAI-compatible chat completions with streaming and tools, one warm connection per provider per call.

* Each call keeps its own HTTP connection(s) open between turns (TLS setup is not on the critical path).
* A turn cancelled mid-stream leaves an unread response on the connection; such a connection is closed,
  otherwise the next request on it fails (``ResponseNotReady``). http.client reconnects on the next request.
* Hedging: when a secondary provider is configured, it is asked as well if the primary gives no first
  token within ``hedge_ms`` or fails; the first provider to answer wins and the other request is dropped.
* Retry: a turn whose request failed before anything was yielded (e.g. a provider 500 on a malformed model
  message) is asked once more of the first provider instead of handing the caller to a person at once.
* Usage: requests ask for token usage in the stream (``stream_options.include_usage``). A provider that
  rejects the option (HTTP 400/422 naming it) is asked again without it and remembered for the process.
  Every request the provider accepted is counted, including dropped speculative and hedged ones; without
  reported usage the tokens are estimated (characters / 3.2) and the request is counted as estimated.
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


class LlmSession:
    """The LLM side of one call."""

    def __init__(self, primary: LlmProvider, secondary: LlmProvider | None, hedge_ms: int,
                 temperature: float, max_tokens: int):
        self.conns = [ProviderConnection(primary)] + ([ProviderConnection(secondary)] if secondary else [])
        self.hedge_s = hedge_ms / 1000
        self.temperature, self.max_tokens = temperature, max_tokens
        self.last_provider = ""
        self.usage = LlmUsage()

    def warm(self) -> None:
        for conn in self.conns:
            conn.warm()

    def close(self) -> None:
        for conn in self.conns:
            conn.close()

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

        def start(i: int) -> None:  # i: provider; the attempt's index is its place in ``attempts``
            attempt = _Attempt(len(attempts), self.conns[i], self._payload(self.conns[i], messages, tools), out)
            attempts.append(attempt)
            attempt.start()

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
                try:
                    index, kind, payload = out.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                if winner is not None and index != winner:
                    continue
                if kind == "error":
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
