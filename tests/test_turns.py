from voice_agent_runtime.turns import Event, TurnDetector

HOP_S = 256 / 8000  # 32 ms


def run(detector, probs, speaking=False):
    out = []
    for i, p in enumerate(probs):
        for e in detector.push(p, speaking):
            out.append((i, e.event, round(e.segment_start, 3)))
    return out


def hops(ms):
    return round(ms / (HOP_S * 1000))


def test_speculate_then_turn_end_after_600ms():
    d = TurnDetector(HOP_S)
    events = run(d, [0.0] * 10 + [0.9] * 20 + [0.1] * 30)
    names = [e for _, e, _ in events]
    assert names == [Event.CALLER_SPEECH, Event.SPECULATE, Event.TURN_END]
    speculate_at = events[1][0] - 29  # hops of silence after the last voiced hop
    turn_end_at = events[2][0] - 29
    assert speculate_at * HOP_S * 1000 >= 120 and (speculate_at - 1) * HOP_S * 1000 < 120
    assert turn_end_at * HOP_S * 1000 >= 600 and (turn_end_at - 1) * HOP_S * 1000 < 600
    # the segment starts 200 ms before the first voiced hop
    assert events[1][2] == round(10 * HOP_S - 0.2, 3)


def test_question_ends_the_turn_after_300ms():
    d = TurnDetector(HOP_S)
    run(d, [0.9] * 20 + [0.1] * 4)  # speculation starts at 128 ms
    assert d.speculating
    d.set_transcript("Вы работаете в субботу?")
    events = run(d, [0.1] * 10)
    end = [i for i, e, _ in events if e is Event.TURN_END][0]
    assert (4 + end + 1) * HOP_S * 1000 >= 300
    assert (4 + end + 1) * HOP_S * 1000 < 600


def test_caller_resumes_cancels_speculation():
    d = TurnDetector(HOP_S)
    events = run(d, [0.9] * 10 + [0.1] * 5 + [0.9] * 3 + [0.1] * 30)
    names = [e for _, e, _ in events]
    assert names[:3] == [Event.CALLER_SPEECH, Event.SPECULATE, Event.CANCEL]
    assert names[3:] == [Event.SPECULATE, Event.TURN_END]
    # the second speculation still includes the speech from the start of the turn
    assert events[3][2] == 0.0


def test_barge_in_needs_four_voiced_hops_while_speaking():
    d = TurnDetector(HOP_S)
    events = run(d, [0.9, 0.9, 0.9], speaking=True)
    assert events == []
    events = run(d, [0.9], speaking=True)
    assert [e for _, e, _ in events] == [Event.BARGE_IN, Event.CALLER_SPEECH]


def test_short_noise_does_not_barge_in():
    d = TurnDetector(HOP_S)
    assert all(e is not Event.BARGE_IN for _, e, _ in run(d, [0.9, 0.9, 0.1, 0.9, 0.9, 0.1], speaking=True))


def test_no_events_after_turn_end_until_reset():
    d = TurnDetector(HOP_S)
    run(d, [0.9] * 10 + [0.1] * 25)
    assert d.turn_ended
    assert run(d, [0.1] * 40) == []
    d.reset_turn()
    names = [e for _, e, _ in run(d, [0.9] * 5 + [0.1] * 25)]
    assert names == [Event.CALLER_SPEECH, Event.SPECULATE, Event.TURN_END]
