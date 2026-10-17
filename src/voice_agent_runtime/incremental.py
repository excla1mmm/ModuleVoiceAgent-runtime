"""Incremental recognition of one caller turn: pieces cut at internal pauses, the tail at the end of the turn.

GigaAM is not streaming and its time grows with the audio (~340 ms for a 2-4 s phrase, 500-700 ms for
5-10 s), so long questions used to wait for the whole utterance to be recognized after the caller stopped.
Every speculative turn (120 ms of silence) recognizes the audio from the last committed boundary to "now"
as the turn's tail. When the pause reaches ``chunk_pause_ms`` (CHUNK) that tail is committed as a finished
piece: the boundary moves to its end and later speculations recognize only the speech after it. Each piece of
audio is recognized once (plus the occasional tail of a speculation the caller talked over, as before), so the
CPU cost does not grow with the length of the turn, and at the end of the turn only the last phrase is
left to recognize. Pieces shorter than ``min_piece_s`` are not committed (too little context for the model);
their audio stays in the next tail.

All methods run on the event loop; ``recognize(start, end)`` submits the audio range to the call's worker
thread and returns a future with the text.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

MIN_TAIL_S = 0.1  # shorter tails (silence after a committed piece) are not worth a model run


@dataclass
class Piece:
    start: float
    end: float
    text: Future


class Utterance:
    def __init__(self, start: float, recognize: Callable[[float, float], Future], min_piece_s: float):
        self.start = start  # seconds of call audio where the turn's speech starts (with pre-roll)
        self.boundary = start  # audio before it belongs to committed pieces
        self.recognize = recognize
        self.min_piece_s = min_piece_s
        self.committed: list[Piece] = []
        self.tail: Piece | None = None  # recognized by the current speculation, not committed

    def speculate(self, now: float) -> list[Future]:
        """A speculative turn starts: recognize the tail; the turn's text is the pieces plus the tail."""
        self._drop_tail()
        if now - self.boundary >= MIN_TAIL_S:
            self.tail = Piece(self.boundary, now, self.recognize(self.boundary, now))
        return [p.text for p in self.committed] + ([self.tail.text] if self.tail else [])

    def cut(self, now: float) -> None:
        """An internal pause: commit the speculation's tail, or recognize the speech so far as a piece."""
        if self.tail is not None:
            if self.tail.end - self.tail.start >= self.min_piece_s:
                self._commit(self.tail)
                self.tail = None
        elif now - self.boundary >= self.min_piece_s:
            self._commit(Piece(self.boundary, now, self.recognize(self.boundary, now)))

    def resume(self) -> None:
        """The caller resumed before the pause was long enough: the speculative tail is obsolete."""
        self._drop_tail()

    def close(self) -> None:
        """The turn is over (answered or abandoned): drop work nobody will read."""
        self._drop_tail()
        for piece in self.committed:
            piece.text.cancel()  # a no-op for pieces already recognized

    def _commit(self, piece: Piece) -> None:
        self.committed.append(piece)
        self.boundary = piece.end

    def _drop_tail(self) -> None:
        if self.tail is not None:
            self.tail.text.cancel()  # only a queued recognition can be cancelled; a running one finishes
            self.tail = None


def join_pieces(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip())
