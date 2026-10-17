"""The agent hearing itself: a caller on a speakerphone without echo cancellation sends the agent's voice back
a few hundred milliseconds later (~330 ms, as loud as -5 dB of the agent). Recognized, it looked like the caller
said "Здравствуй" / "Я" / "Спасибо" over the agent: the agent cut itself off and answered its own words.

Two layers:

* ``EchoCanceller``: WebRTC AEC3 (livekit's AudioProcessingModule, Apache-2.0) on the caller audio with the
  agent audio as the reference, before VAD and recognition. On recorded calls it took 8-34 dB off the echo and
  under 1 dB off the caller; what is left of the echo is quiet but can still be "recognized" as "да" / "угу",
  mostly in the first second, before the filter has adapted. So a phrase the canceller took more than
  ``ECHO_REMOVED_DB`` off is echo as well.
* ``EchoGuard``: remembers what the agent said lately; caller speech that started while the agent was audible
  and consists only of the agent's recent words is echo. "да" / "нет" alone are never echo by text (a word of 4
  letters or more is required), and speech that started after the agent fell silent is never checked.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

import numpy as np

WINDOW_S = 15.0  # agent phrases older than this are forgotten (long replies are synthesized ahead of playback)
MAX_WORDS = 8  # echo picked up by a microphone is a fragment, not a long sentence
WORD = re.compile(r"[а-яa-z0-9]+")
ECHO_REMOVED_DB = 6.0  # the canceller took at least this much off a phrase: it was the agent's echo
AEC_RATE = 8000
AEC_FRAME_BYTES = AEC_RATE // 100 * 2  # the module works on 10 ms frames of 16-bit mono
MAX_RENDER_FRAMES = 100  # agent audio waiting for caller audio (1 s); older frames are dropped


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower().replace("ё", "е"))


def _same(caller: str, agent: str) -> bool:
    """The same word up to its ending ("здравствуй" / "здравствуйте")."""
    if caller == agent:
        return True
    n = min(5, len(caller), len(agent))
    return n >= 4 and caller[:n] == agent[:n]


class EchoGuard:
    def __init__(self, window_s: float = WINDOW_S):
        self.window_s = window_s
        self._said: deque[tuple[float, list[str]]] = deque(maxlen=50)
        self._lock = threading.Lock()  # phrases are added from the call's worker thread

    def said(self, text: str, now: float | None = None) -> None:
        words = _words(text)
        if words:
            with self._lock:
                self._said.append((time.monotonic() if now is None else now, words))

    def is_echo(self, text: str, now: float | None = None) -> bool:
        words = _words(text)
        if not words or len(words) > MAX_WORDS or not any(len(w) >= 4 for w in words):
            return False
        since = (time.monotonic() if now is None else now) - self.window_s
        with self._lock:
            agent = {w for at, said in self._said if at >= since for w in said}
        return all(any(_same(w, a) for a in agent) for w in words)


def level_db(pcm: np.ndarray) -> float:
    """RMS level of float samples in dBFS (-120 for silence)."""
    if not len(pcm):
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))) + 1e-6))


class EchoCanceller:
    """Per call: the agent audio as it is sent goes in with ``render``; the caller audio goes through
    ``capture``, which returns the caller audio and its echo-cancelled version, frame-aligned (both lag the
    input by less than 10 ms). Each caller frame is matched with the next agent frame (silence when the agent
    is quiet): both streams run at real time, so the agent audio stays aligned with the caller audio.
    """

    def __init__(self):
        from livekit import rtc  # optional: without it the call runs without echo cancellation

        self._rtc = rtc
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=False,
                                             high_pass_filter=True, auto_gain_control=False)
        self.render_frames: deque[bytes] = deque(maxlen=MAX_RENDER_FRAMES)
        self._render_tail = b""
        self._capture_tail = b""
        self._silence = bytes(AEC_FRAME_BYTES)
        self._samples = AEC_FRAME_BYTES // 2

    def _frame(self, data: bytes):
        return self._rtc.AudioFrame(data, AEC_RATE, 1, self._samples)

    def render(self, pcm: bytes) -> None:
        data = self._render_tail + pcm
        cut = len(data) - len(data) % AEC_FRAME_BYTES
        self.render_frames.extend(data[i:i + AEC_FRAME_BYTES] for i in range(0, cut, AEC_FRAME_BYTES))
        self._render_tail = data[cut:]

    def capture(self, pcm: bytes) -> tuple[bytes, bytes]:
        data = self._capture_tail + pcm
        cut = len(data) - len(data) % AEC_FRAME_BYTES
        self._capture_tail = data[cut:]
        raw, clean = data[:cut], bytearray()
        for i in range(0, cut, AEC_FRAME_BYTES):
            reference = self.render_frames.popleft() if self.render_frames else self._silence
            self.apm.process_reverse_stream(self._frame(reference))
            frame = self._frame(raw[i:i + AEC_FRAME_BYTES])
            self.apm.process_stream(frame)
            clean += bytes(frame.data)
        return raw, bytes(clean)


def new_canceller() -> EchoCanceller | None:
    try:
        return EchoCanceller()
    except Exception:  # noqa: BLE001 - no livekit (tests, an old image): run without echo cancellation
        return None
