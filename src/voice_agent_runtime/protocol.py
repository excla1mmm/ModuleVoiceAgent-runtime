"""Asterisk chan_websocket media protocol (Asterisk 22, ``Dial(WebSocket/<connection>/c(slin)v(...))``).

Asterisk opens the WebSocket (subprotocol ``media``) and sends control messages as text, by default
``MEDIA_START connection_id:... channel:... channel_id:... format:slin optimal_frame_size:320 ptime:20``
(the JSON control format is accepted as well). Binary frames carry signed linear 16-bit little-endian
audio at 8 kHz (``slin``) or 16 kHz (``slin16``, dialled for wideband callers) in both directions; the
application paces its frames at 20 ms. Commands sent back as text:
``FLUSH_MEDIA`` drops audio queued in Asterisk (barge-in), ``HANGUP`` ends the WebSocket leg so the
dialplan continues after Dial(). Variables from ``v(...)`` arrive as the query string of the request path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

SAMPLE_RATE = 8000  # of the audio processing (echo canceller, VAD, recognition); slin16 input is brought down to it
FRAME_BYTES = 320  # 20 ms of slin at 8 kHz
FRAME_SECONDS = 0.02
FORMAT_RATES = {"slin": 8000, "slin16": 16000}  # media format -> sample rate on the wire
CMD_FLUSH_MEDIA = "FLUSH_MEDIA"
CMD_HANGUP = "HANGUP"
EVENT_MEDIA_START = "MEDIA_START"
SUPPORTED_FORMATS = set(FORMAT_RATES)

_CALLER = re.compile(r"[^0-9+]")
_CALL_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_AGENT = re.compile(r"^[0-9]{1,16}$")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ControlMessage:
    name: str
    params: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MediaStart:
    connection_id: str
    channel: str
    format: str
    optimal_frame_size: int
    ptime: int

    @classmethod
    def from_control(cls, msg: ControlMessage) -> MediaStart:
        fmt = msg.params.get("format", "")
        if fmt not in SUPPORTED_FORMATS:
            raise ProtocolError(f"unsupported media format {fmt!r}, dial with c(slin) or c(slin16)")
        default_frame = int(FORMAT_RATES[fmt] * FRAME_SECONDS) * 2
        try:
            frame = int(msg.params.get("optimal_frame_size", default_frame))
            ptime = int(msg.params.get("ptime", 20))
        except ValueError as exc:
            raise ProtocolError("invalid MEDIA_START frame parameters") from exc
        return cls(msg.params.get("connection_id", ""), msg.params.get("channel", ""), fmt,
                   frame if frame > 0 else default_frame, ptime if ptime > 0 else 20)

    @property
    def rate(self) -> int:
        return FORMAT_RATES[self.format]


@dataclass(frozen=True)
class CallParams:
    caller: str
    call_id: str
    agent_id: str


def parse_control(text: str) -> ControlMessage:
    """Parses a text control message in the plain-text or the JSON control format."""
    text = text.strip()
    if not text:
        raise ProtocolError("empty control message")
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid JSON control message") from exc
        name = str(data.pop("event", data.pop("command", "")))
        if not name:
            raise ProtocolError("JSON control message without event")
        return ControlMessage(name, {str(k): str(v) for k, v in data.items()})
    name, _, rest = text.partition(" ")
    params: dict[str, str] = {}
    for token in rest.split():
        key, sep, value = token.partition(":")
        if sep:
            params[key] = value
    return ControlMessage(name, params)


def call_params(path: str) -> CallParams:
    """Reads callerid, callid and agent from the request path (``/media?callerid=...&callid=...&agent=...``)."""
    query = parse_qs(urlsplit(path).query)

    def first(key: str) -> str:
        return query.get(key, [""])[0].strip()

    call_id, agent = first("callid"), first("agent")
    if not _CALL_ID.match(call_id):
        raise ProtocolError("callid is missing or invalid")
    if not _AGENT.match(agent):
        raise ProtocolError("agent is missing or invalid")
    return CallParams(caller=_CALLER.sub("", first("callerid"))[:32], call_id=call_id, agent_id=agent)


def frames(pcm: bytes, size: int = FRAME_BYTES) -> list[bytes]:
    """Splits PCM into fixed-size frames, padding the last one with silence."""
    out = [pcm[i:i + size] for i in range(0, len(pcm), size)]
    if out and len(out[-1]) < size:
        out[-1] = out[-1].ljust(size, b"\0")
    return out
