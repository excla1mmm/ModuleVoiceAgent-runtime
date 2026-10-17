import numpy as np
import pytest

from voice_agent_runtime.echo import EchoGuard


def test_the_agent_own_recent_words_are_echo():
    guard = EchoGuard()
    guard.said("Здравствуйте. Я - Маша, голосовой помощник компании мик+о.", now=100)
    assert guard.is_echo("Здравствуй.", now=101)
    assert guard.is_echo("голосовой помощник", now=101)
    assert not guard.is_echo("Здравствуйте, мне нужен бухгалтер", now=101)


def test_short_answers_old_phrases_and_long_speech_are_not_echo():
    guard = EchoGuard(window_s=15)
    guard.said("Да, нет, конечно. Перезвонить вам на номер, с которого вы звоните, удобно ли завтра?", now=100)
    assert not guard.is_echo("Да.", now=101)
    assert not guard.is_echo("нет", now=101)
    assert not guard.is_echo("Перезвонить вам на номер, с которого вы звоните, удобно ли завтра", now=101)
    assert guard.is_echo("Перезвонить", now=110)
    assert not guard.is_echo("Перезвонить", now=120)
    assert not guard.is_echo("", now=101)


def _speechlike(seconds, seed, rate=8000):
    """Band-limited noise with a syllable envelope: enough like speech for the canceller."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    noise = np.convolve(rng.standard_normal(n), np.ones(4) / 4, mode="same")
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / rate)
    return (noise * envelope * 6000).astype(np.int16)


def test_the_canceller_removes_the_agent_echo_but_keeps_the_caller():
    pytest.importorskip("livekit.rtc")
    from voice_agent_runtime.echo import EchoCanceller, level_db
    canceller = EchoCanceller()
    agent = _speechlike(6, 1)
    delay = int(0.33 * 8000)  # as on a real call
    echo = np.zeros_like(agent)
    echo[delay:] = (agent[:-delay] * 0.6).astype(np.int16)
    caller = np.zeros_like(agent)
    caller[4 * 8000:] = _speechlike(2, 2)
    mic = (echo.astype(np.int32) + caller).clip(-32768, 32767).astype(np.int16)
    raws, cleans = [], []
    for i in range(0, len(mic), 160):  # 20 ms frames both ways, as on the media socket
        canceller.render(agent[i:i + 160].tobytes())
        raw, clean = canceller.capture(mic[i:i + 160].tobytes())
        raws.append(raw)
        cleans.append(clean)
    raw = np.frombuffer(b"".join(raws), dtype=np.int16).astype(np.float32) / 32768
    clean = np.frombuffer(b"".join(cleans), dtype=np.int16).astype(np.float32) / 32768
    echo_only, double_talk = slice(2 * 8000, 4 * 8000), slice(4 * 8000 + delay, 6 * 8000)
    assert level_db(raw[echo_only]) - level_db(clean[echo_only]) >= 6  # adapted: echo mostly removed
    caller_level = level_db(caller[4 * 8000 + delay:].astype(np.float32) / 32768)
    assert level_db(clean[double_talk]) > caller_level - 6  # the caller over the echo is kept
