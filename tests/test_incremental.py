"""Incremental recognition on synthetic VAD sequences: what is recognized when, and that nothing is recognized twice."""

from concurrent.futures import Future

import pytest

from voice_agent_runtime.incremental import Utterance, join_pieces
from voice_agent_runtime.turns import Event, TurnDetector

HOP_S = 256 / 8000  # 32 ms


def hops(ms):
    return round(ms / (HOP_S * 1000))


def voice(ms):
    return [0.9] * hops(ms)


def pause(ms):
    return [0.1] * hops(ms)


class Recorder:
    """Stands in for the call worker: records the audio ranges and answers with their bounds."""

    def __init__(self):
        self.ranges = []

    def __call__(self, start, end):
        self.ranges.append((round(start, 3), round(end, 3)))
        future = Future()
        future.set_result(f"[{start:.2f}-{end:.2f}]")
        return future


def drive(probs, chunk_pause_ms=160, speculate_ms=120, min_piece_s=1.0):
    """Mirrors CallSession.on_event: returns the recorder and the pieces of the last speculation."""
    detector = TurnDetector(HOP_S, speculate_ms=speculate_ms, chunk_pause_ms=chunk_pause_ms)
    recorder, utterance, pieces, events = Recorder(), None, None, []
    for prob in probs:
        for emitted in detector.push(prob, agent_speaking=False):
            events.append(emitted.event)
            if emitted.event in (Event.SPECULATE, Event.CHUNK) and (
                    utterance is None or utterance.start != emitted.segment_start):
                utterance = Utterance(emitted.segment_start, recorder, min_piece_s)
            if emitted.event is Event.SPECULATE:
                pieces = utterance.speculate(detector.pos)
            elif emitted.event is Event.CHUNK:
                utterance.cut(detector.pos)
            elif emitted.event is Event.CANCEL:
                utterance.resume()
    return recorder, pieces, events, utterance


def covered(ranges):
    return sum(end - start for start, end in ranges)


LONG_TURN = pause(300) + voice(2500) + pause(250) + voice(3000) + pause(200) + voice(1500) + pause(700)


def test_chunk_event_once_per_pause_and_off_by_default():
    detector = TurnDetector(HOP_S, chunk_pause_ms=160)
    events = [e.event for p in voice(500) + pause(700) for e in detector.push(p, False)]
    assert events.count(Event.CHUNK) == 1
    assert events.index(Event.SPECULATE) < events.index(Event.CHUNK) < events.index(Event.TURN_END)
    detector = TurnDetector(HOP_S)
    assert Event.CHUNK not in [e.event for p in voice(500) + pause(400) for e in detector.push(p, False)]


def test_long_turn_is_recognized_piece_by_piece_without_overlap():
    recorder, pieces, events, utterance = drive(LONG_TURN)
    assert events[-1] is Event.TURN_END
    assert len(utterance.committed) == 3 and len(pieces) == 3
    # the speculations of the turn: each recognizes only the speech after the previous pause
    assert len(recorder.ranges) == 3
    for (_, end), (start, _) in zip(recorder.ranges, recorder.ranges[1:], strict=False):
        assert start == end  # contiguous, never overlapping
    first, last = recorder.ranges[0][0], recorder.ranges[-1][1]
    assert covered(recorder.ranges) == pytest.approx(last - first)  # every second recognized once
    # at the end of the turn only the last phrase (1.5 s + pauses) is left
    assert recorder.ranges[-1][1] - recorder.ranges[-1][0] < 1.8
    assert join_pieces([p.result() for p in pieces]).count("[") == 3


def test_disabled_chunks_recognize_the_whole_turn_each_time():
    recorder, pieces, _, _ = drive(LONG_TURN, chunk_pause_ms=0)
    assert len(pieces) == 1
    assert {start for start, _ in recorder.ranges} == {recorder.ranges[0][0]}  # always from the turn start
    assert covered(recorder.ranges) > covered(drive(LONG_TURN)[0].ranges) * 1.5


def test_short_pause_drops_the_tail_and_keeps_the_boundary():
    # 130 ms: a speculation starts but the pause is too short to cut the turn there
    recorder, pieces, events, utterance = drive(pause(300) + voice(2000) + pause(130) + voice(1500) + pause(700))
    assert Event.CANCEL in events and not utterance.committed[:-1]
    starts = [start for start, _ in recorder.ranges]
    assert starts[0] == starts[1]  # the second speculation still starts at the beginning of the turn
    assert len(pieces) == 1


def test_short_piece_is_not_committed():
    recorder, pieces, _, utterance = drive(pause(300) + voice(500) + pause(200) + voice(1500) + pause(700))
    assert len(pieces) == 1 and recorder.ranges[0][0] == recorder.ranges[-1][0]
    assert len(utterance.committed) == 1  # only the final tail, after the last pause


def test_chunk_before_speculation_submits_its_own_piece():
    recorder, pieces, events, utterance = drive(
        pause(300) + voice(2000) + pause(200) + voice(1500) + pause(700), chunk_pause_ms=96, speculate_ms=160)
    assert events.index(Event.CHUNK) < events.index(Event.SPECULATE)
    assert len(pieces) == 2 and utterance.committed[0].end == pytest.approx(recorder.ranges[1][0], abs=1e-3)


def test_resume_cancels_a_queued_tail_and_close_cancels_queued_pieces():
    queued = []

    def recognize(start, end):
        queued.append(Future())
        return queued[-1]

    utterance = Utterance(0.0, recognize, 1.0)
    utterance.speculate(0.5)
    utterance.resume()
    assert queued[0].cancelled() and utterance.tail is None
    utterance.speculate(1.5)
    utterance.cut(1.6)
    utterance.close()
    assert queued[1].cancelled()


def test_join_pieces_skips_empty_texts():
    assert join_pieces(["Здравствуйте.", "", "  соедините с бухгалтерией "]) == "Здравствуйте. соедините с бухгалтерией"
