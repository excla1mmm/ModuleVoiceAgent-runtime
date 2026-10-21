import numpy as np

from voice_agent_runtime.speech import Downsampler, to_pcm


def tone(freq, seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)


def level(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)[200:]  # past the filter's start
    return float(np.sqrt(np.mean(x ** 2)))


def test_slin16_is_brought_down_to_8_khz_the_same_in_frames_as_at_once():
    audio = (tone(440, 1.0, 16000) + tone(1000, 1.0, 16000) // 2).tobytes()
    whole = Downsampler(16000).push(audio)
    down = Downsampler(16000)
    framed = b"".join(down.push(audio[i:i + 642]) for i in range(0, len(audio), 642))  # odd sample counts too
    assert len(whole) == len(audio) // 2 and framed == whole


def test_the_downsampler_keeps_speech_and_removes_what_8_khz_cannot_carry():
    assert level(Downsampler(16000).push(tone(1000, 0.5, 16000).tobytes())) > 6000  # 10000 * 0.707
    assert level(Downsampler(16000).push(tone(6000, 0.5, 16000).tobytes())) < 300  # would alias to 2 kHz


def test_synthesized_audio_is_resampled_to_the_call_rate():
    audio = np.sin(np.linspace(0, 200, 24000)).astype(np.float32)  # 1 s at 24 kHz
    assert len(to_pcm(audio, 24000, 16000)) == 16000 * 2
    assert len(to_pcm(audio, 24000)) == 8000 * 2
    assert len(to_pcm(audio[:8000], 8000)) == 8000 * 2
