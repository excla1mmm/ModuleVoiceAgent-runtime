"""Turn-taking on streaming VAD probabilities (Silero, 256-sample hops at 8 kHz = 32 ms).

Smart-turn models were unreliable on Russian telephony; the end of a turn is plain silence: 600 ms, or
300 ms when the speculative transcript ends with "?". After 120 ms of silence a speculative turn starts
(STT -> LLM -> first-clause TTS) and is discarded if the caller resumes. About four voiced hops (~130 ms)
of caller speech interrupt the agent and cancel a scheduled hangup. A pause of ``chunk_pause_ms`` inside the
turn (CHUNK) lets the call commit the speech before it to incremental recognition (0 disables it).

The detector is pure: it turns probabilities into events, the call session acts on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Event(Enum):
    SPECULATE = "speculate"  # start a speculative turn on audio from ``segment_start``
    CANCEL = "cancel"  # the caller resumed: drop the speculative turn
    TURN_END = "turn_end"  # the turn is over: play the speculative answer
    BARGE_IN = "barge_in"  # the caller talks over the agent: stop playback
    CALLER_SPEECH = "caller_speech"  # sustained caller speech (cancels a pending hangup)
    CHUNK = "chunk"  # an internal pause: the speech so far can be recognized as a finished piece


@dataclass(frozen=True)
class Emitted:
    event: Event
    segment_start: float = 0.0  # seconds of call audio where the turn's speech starts (SPECULATE, CHUNK)


class TurnDetector:
    PRE_ROLL_S = 0.2  # audio kept before the first voiced hop

    def __init__(self, hop_s: float, speculate_ms: int = 120, turn_end_ms: int = 600, question_end_ms: int = 300,
                 barge_in_frames: int = 4, threshold: float = 0.5, chunk_pause_ms: int = 0):
        self.hop_s = hop_s
        self.speculate_ms, self.turn_end_ms, self.question_end_ms = speculate_ms, turn_end_ms, question_end_ms
        self.chunk_pause_ms = chunk_pause_ms
        self.barge_in_frames, self.threshold = barge_in_frames, threshold
        self.pos = 0.0  # seconds of caller audio processed
        self.voiced_run = 0
        self.speech_start: float | None = None
        self.last_voice: float | None = None
        self.speculating = False
        self.chunked = False  # CHUNK already emitted for the current pause
        self.turn_ended = False
        self.question = False  # the speculative transcript ends with "?"

    def set_transcript(self, text: str) -> None:
        """The speculative turn reports its transcript; a question shortens the end-of-turn pause."""
        self.question = text.rstrip().endswith("?")

    def reset_turn(self) -> None:
        """The answer to the ended turn has started: the next caller speech starts a new turn."""
        self.speech_start = self.last_voice = None
        self.speculating = self.turn_ended = self.question = False

    def push(self, prob: float, agent_speaking: bool) -> list[Emitted]:
        self.pos += self.hop_s
        events: list[Emitted] = []
        if prob >= self.threshold:
            self.voiced_run += 1
            if self.voiced_run >= self.barge_in_frames:
                if agent_speaking:
                    events.append(Emitted(Event.BARGE_IN))
                if self.voiced_run == self.barge_in_frames:
                    events.append(Emitted(Event.CALLER_SPEECH))
            if self.speech_start is None:
                self.speech_start = self.pos - self.hop_s
            self.last_voice = self.pos
            self.chunked = False
            if self.speculating and not self.turn_ended:
                self.speculating, self.question = False, False  # caller kept talking
                events.append(Emitted(Event.CANCEL))
            return events
        self.voiced_run = 0
        if self.last_voice is None or self.turn_ended:
            return events
        silence_ms = (self.pos - self.last_voice) * 1000
        start = max(0.0, (self.speech_start or 0.0) - self.PRE_ROLL_S)
        if not self.speculating and silence_ms >= self.speculate_ms - 1e-6:
            self.speculating = True
            events.append(Emitted(Event.SPECULATE, start))
        if self.chunk_pause_ms and not self.chunked and silence_ms >= self.chunk_pause_ms - 1e-6:
            self.chunked = True
            events.append(Emitted(Event.CHUNK, start))
        if self.speculating and (silence_ms >= self.turn_end_ms - 1e-6
                                 or (self.question and silence_ms >= self.question_end_ms - 1e-6)):
            self.turn_ended = True
            events.append(Emitted(Event.TURN_END))
        return events
