"""Checks with the real models (GigaAM v3, Silero VAD, Silero TTS): skipped unless they are already in the
models cache (the Silero TTS checks also need ``silero/v5_cis_base.pt`` there and torch installed).

Speech comes from Piper (Linux, the service image) or macOS ``say`` (Piper's macOS wheel lacks espeak data).
Set ``VA_MODELS_DIR`` to the cache (default ``~/.cache/va-models``); nothing is downloaded here.
"""

import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import bundle_data
from voice_agent_runtime.bundle import parse_bundle
from voice_agent_runtime.config import SUPPORTED_VOICES, VOICES, LlmProvider, Settings
from voice_agent_runtime.hotwords import abbreviation_fixes, bundle_terms, correct
from voice_agent_runtime.incremental import Utterance, join_pieces
from voice_agent_runtime.normalize import normalize
from voice_agent_runtime.speech import SILERO_FILE
from voice_agent_runtime.turns import Event, TurnDetector

MODELS = Path(os.environ.get("VA_MODELS_DIR", Path.home() / ".cache/va-models"))
HUB = MODELS / "huggingface" / "hub"
NEEDED = ("models--istupakov--gigaam-v3-onnx", "models--istupakov--silero-vad-onnx", "models--rhasspy--piper-voices")
pytestmark = pytest.mark.skipif(not all((HUB / n).is_dir() for n in NEEDED), reason=f"speech models not in {HUB}")
SILERO = (MODELS / SILERO_FILE).is_file()
SILERO_VOICES = [v for v in SUPPORTED_VOICES if VOICES[v].engine == "silero"]


@pytest.fixture(scope="module")
def engines():
    pytest.importorskip("onnx_asr")
    from voice_agent_runtime.speech import SpeechEngines

    if SILERO:
        pytest.importorskip("torch")
    voices = tuple(v for v in SUPPORTED_VOICES if SILERO or VOICES[v].engine != "silero")
    return SpeechEngines(Settings(primary=LlmProvider("p", "https://x", "m", "k"), models_dir=MODELS, voices=voices))


@pytest.fixture(scope="module")
def speak(engines, tmp_path_factory):
    folder = tmp_path_factory.mktemp("speech")

    def say(text: str) -> np.ndarray:
        if sys.platform != "darwin":
            pcm = engines.synthesize("ru_RU-dmitri-medium", text)
            return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        if not shutil.which("say") or not shutil.which("afconvert"):
            pytest.skip("no speech synthesizer")
        aiff, wav = folder / "s.aiff", folder / "s.wav"
        subprocess.run(["say", "-v", "Milena", "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@8000", "-c", "1", str(aiff), str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768

    return say


PHRASES = [
    "Здравствуйте, соедините меня с бухгалтерией, пожалуйста.",
    "Позовите Алексееву Наталью, помощника директора.",
    "Сколько стоит тариф Старт и можно ли подключить его на этой неделе?",
]


def test_hotwords_do_not_change_normal_speech(engines, speak):
    bundle = parse_bundle("3", bundle_data(hotwords=["АТС", "МикоПБХ"]))
    terms = bundle_terms(bundle)
    fixes = abbreviation_fixes(terms)
    for text in PHRASES:
        audio = speak(text)
        assert correct(engines.recognize(audio, terms), fixes) == engines.recognize(audio)


def test_incremental_recognition_leaves_only_the_tail_after_the_turn(engines, speak):
    silence = np.zeros(int(0.4 * 8000), dtype=np.float32)
    audio = np.concatenate([silence, speak("Здравствуйте, я хотел бы узнать про ваши тарифы для офиса."), silence,
                            speak("И ещё подскажите, вы работаете в субботу?"), np.zeros(8000, dtype=np.float32)])
    detector, vad = TurnDetector(256 / 8000, chunk_pause_ms=160), engines.new_vad()
    timings: list[tuple[float, float, float]] = []

    def recognize(start, end):
        from concurrent.futures import Future

        started = time.perf_counter()
        text = engines.recognize(audio[int(start * 8000):int(end * 8000)])
        timings.append((start, end, time.perf_counter() - started))
        future = Future()
        future.set_result(text)
        return future

    utterance = pieces = speech_end = None
    for i in range(0, len(audio) - 256, 256):
        for emitted in detector.push(vad.prob(audio[i:i + 256]), agent_speaking=False):
            if emitted.event in (Event.SPECULATE, Event.CHUNK) and (
                    utterance is None or utterance.start != emitted.segment_start):
                utterance = Utterance(emitted.segment_start, recognize, 1.0)
            if emitted.event is Event.SPECULATE:
                pieces, speech_end = utterance.speculate(detector.pos), detector.pos
            elif emitted.event is Event.CHUNK:
                utterance.cut(detector.pos)
            elif emitted.event is Event.CANCEL:
                utterance.resume()
        if detector.turn_ended:
            break
    assert detector.turn_ended and len(pieces) >= 2
    tail_s = timings[-1][2]
    started = time.perf_counter()
    whole = engines.recognize(audio[int(utterance.start * 8000):int(speech_end * 8000)])
    whole_s = time.perf_counter() - started
    text = join_pieces([p.result() for p in pieces])
    print(f"\nincremental: tail {tail_s * 1000:.0f} ms vs whole turn {whole_s * 1000:.0f} ms; {text!r} / {whole!r}")
    assert tail_s < whole_s * 0.8
    assert text.rstrip().endswith("?")
    words = lambda s: {w.strip(".,?!").lower() for w in s.split()}  # noqa: E731
    assert len(words(text) & words(whole)) >= 0.8 * len(words(whole))


def words_of(text: str) -> list[str]:
    return [w.strip(".,?!-—").lower().replace("ё", "е") for w in text.split() if w.strip(".,?!-—")]


@pytest.mark.skipif(not SILERO, reason=f"Silero TTS model not in {MODELS / SILERO_FILE}")
@pytest.mark.parametrize("voice", SILERO_VOICES)
def test_silero_voice_is_understood_by_the_recognizer(engines, voice):
    text = normalize("Соединяю вас с отделом продаж. Мы работаем с 9:00 до 18:00, тариф стоит 2 900 рублей в месяц.")
    started = time.perf_counter()
    pcm = engines.synthesize(voice, text)
    took = time.perf_counter() - started
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    heard = engines.recognize(audio)
    print(f"\n{voice}: {len(audio) / 8000:.1f} s of speech in {took * 1000:.0f} ms; {heard!r}")
    assert len(audio) > 8000 * 3 and np.abs(audio).max() > 0.05
    expected = words_of(text)
    assert len(set(words_of(normalize(heard))) & set(expected)) >= 0.8 * len(set(expected))


@pytest.mark.skipif(not SILERO, reason=f"Silero TTS model not in {MODELS / SILERO_FILE}")
def test_silero_skips_what_it_cannot_say(engines):
    assert engines.synthesize("silero-oksana", "...") == b""
    assert len(engines.synthesize("silero-oksana", "Hello. Да.")) > 0  # the Latin sentence is dropped, not an error
    many = " ".join(["Здравствуйте, это длинное приветствие компании."] * 40)  # ~1900 characters, several pieces
    assert len(engines.synthesize("silero-roman", many)) > 2 * 8000 * 60
