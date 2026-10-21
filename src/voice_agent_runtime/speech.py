"""Speech models: Silero VAD, GigaAM v3 e2e-rnnt (onnx-asr), Piper voices and Silero TTS v5_cis_base.

Models are downloaded (Hugging Face; Silero TTS from models.silero.ai, checked by SHA-256) into ``VA_MODELS_DIR``
on first start (or ahead of time with ``voice-agent-runtime download-models``) and loaded once per process.
Only the voices in ``VA_VOICES`` are loaded; torch is imported only when a Silero voice is among them. Every ONNX session runs with one
intra-op thread: under load the calls themselves are the parallelism (~2 concurrent calls per vCPU).
Recognition can be biased towards an agent's terms (``biasing.py``); compiled term sets are cached per
distinct list.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import urllib.request
from collections import OrderedDict
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from . import biasing
from .config import SUPPORTED_VOICES, VOICES, Settings
from .normalize import STRESS, unstressed
from .protocol import SAMPLE_RATE

HOP = 256  # Silero VAD at 8 kHz: 32 ms hops ...
CONTEXT = 32  # ... with 4 ms of context from the previous hop
VAD_REPO, VAD_FILE = "istupakov/silero-vad-onnx", "silero_vad.onnx"
ASR_MODEL = "gigaam-v3-e2e-rnnt"
PIPER_REPO = "rhasspy/piper-voices"
# Silero TTS v5_cis_base (MIT): one model with every speaker; stress marks come from silero-stress (MIT).
SILERO_URL = "https://models.silero.ai/models/tts/ru/v5_cis_base.pt"
SILERO_SHA256 = "ba41b18f6a707ad93605a162998865e7c087153d2e010a26dd02229dab0e672a"
SILERO_FILE = "silero/v5_cis_base.pt"
SILERO_MAX_CHARS = 800  # per apply_tts call: the model takes ~1000 symbols, stress marks included
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
SPEAKABLE = re.compile(r"[А-Яа-яЁё]")
BIAS_CACHE_SIZE = 64
log = logging.getLogger(__name__)


def voice_files(voice: str) -> tuple[str, str]:
    _, name_quality = voice.split("_", 1)  # "ru_RU-dmitri-medium" -> "RU-dmitri-medium"
    _, speaker, quality = name_quality.split("-")
    base = f"ru/ru_RU/{speaker}/{quality}/{voice}.onnx"
    return base, base + ".json"


def silero_chunks(text: str, limit: int = SILERO_MAX_CHARS) -> list[str]:
    """The text in pieces for Silero: whole sentences packed up to ``limit`` characters, a longer sentence cut
    at a space. "+" marks stress for Silero: a hand-set mark before a vowel ("з+амок") stays, any other "+"
    becomes a space; a piece without Cyrillic
    letters is dropped (Silero speaks no Latin and rejects text with nothing it can say)."""
    pieces: list[str] = []
    current = ""
    text = STRESS.sub("\0", text).replace("+", " ").replace("\0", "+")
    for sentence in SENTENCE_END.split(text):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces += [current] if current else []
            current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    pieces.append(current)
    return [p for p in pieces if SPEAKABLE.search(p)]


SILERO_RATES = (8000, 24000, 48000)  # what Silero synthesizes; 16 kHz is made from 24 kHz


def to_pcm(audio: np.ndarray, rate: int, out_rate: int = SAMPLE_RATE) -> bytes:
    audio = np.asarray(audio)
    if audio.dtype == np.int16 or np.abs(audio).max(initial=0) > 1.5:
        audio = audio.astype(np.float32) / 32768
    if rate != out_rate:
        g = gcd(out_rate, rate)
        audio = resample_poly(audio.astype(np.float32), out_rate // g, rate // g)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class Downsampler:
    """slin16 -> 8 kHz for one stream, continuous across frames: the low-pass keeps its state between them."""

    def __init__(self, rate_in: int):
        self.factor = rate_in // SAMPLE_RATE
        self.taps = firwin(63, 0.9 * SAMPLE_RATE / 2, fs=rate_in)
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0  # index of the first sample to keep in the next chunk

    def push(self, data: bytes) -> bytes:
        x = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        y, self.zi = lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.phase::self.factor]
        self.phase = (self.phase - len(x)) % self.factor
        return np.clip(np.round(out), -32768, 32767).astype(np.int16).tobytes()


class StreamVad:
    """Streaming Silero VAD for one call."""

    def __init__(self, session):
        self.session = session
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.sr = np.array([SAMPLE_RATE], dtype=np.int64)

    def prob(self, hop: np.ndarray) -> float:
        x = np.concatenate([self.context, hop])[None, :]
        out, self.state = self.session.run(["output", "stateN"], {"input": x, "state": self.state, "sr": self.sr})
        self.context = hop[-CONTEXT:]
        return float(out[0][0])


def use_models_dir(models_dir: Path) -> None:
    """Points the Hugging Face cache (used by onnx-asr as well) at the models volume."""
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_silero(models_dir: Path, url: str = SILERO_URL, sha256: str = SILERO_SHA256) -> Path:
    """The Silero TTS package in the models volume; downloaded (and checked) when missing or damaged."""
    path = models_dir / SILERO_FILE
    if path.is_file() and _sha256(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:  # noqa: S310 - fixed https URL
        while block := response.read(1 << 20):
            out.write(block)
    if _sha256(partial) != sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: checksum mismatch")
    partial.replace(path)
    return path


def fetch_files(models_dir: Path, voices: tuple[str, ...] = SUPPORTED_VOICES) -> dict[str, Path]:
    """Downloads (or finds in the cache) the VAD model and the models of ``voices``."""
    use_models_dir(models_dir)
    from huggingface_hub import hf_hub_download

    paths = {"vad": Path(hf_hub_download(VAD_REPO, VAD_FILE))}
    for voice in voices:
        if VOICES[voice].engine != "piper":
            continue
        model, config = voice_files(voice)
        paths[voice] = Path(hf_hub_download(PIPER_REPO, model))
        hf_hub_download(PIPER_REPO, config)
    if any(VOICES[v].engine == "silero" for v in voices):
        paths["silero"] = fetch_silero(models_dir)
    return paths


def download_models(models_dir: Path, voices: tuple[str, ...] = SUPPORTED_VOICES) -> None:
    """Fetches every model ahead of the first start (``voice-agent-runtime download-models``)."""
    fetch_files(models_dir, voices)
    import onnx_asr

    onnx_asr.load_model(ASR_MODEL)  # downloads the STT model into the same cache


class SileroTts:
    """Silero TTS v5_cis_base with silero-stress. One lock: the model is shared by every speaker and call."""

    def __init__(self, path: Path):
        import torch
        from silero_stress import load_accentor

        torch.set_num_threads(1)
        self.torch = torch
        self.model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self.accentor = load_accentor()
        self.lock = threading.Lock()

    def synthesize(self, speaker: str, text: str, rate: int = SAMPLE_RATE) -> bytes:
        model_rate = rate if rate in SILERO_RATES else 24000
        chunks = []
        for piece in silero_chunks(text):
            with self.lock, self.torch.inference_mode():
                try:
                    audio = self.model.apply_tts(text=self.accentor(piece), speaker=speaker, sample_rate=model_rate)
                except ValueError:  # nothing the model can say in this piece
                    continue
            chunks.append(audio.numpy())
        if not chunks:
            return b""
        return to_pcm(np.concatenate(chunks), model_rate, rate)


class SpeechEngines:
    def __init__(self, settings: Settings):
        import onnx_asr
        import onnxruntime as ort
        from piper import PiperVoice

        def options(threads: int) -> ort.SessionOptions:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            return so

        paths = fetch_files(settings.models_dir, settings.voices)
        self.vad_session = ort.InferenceSession(str(paths["vad"]), sess_options=options(1),
                                                providers=["CPUExecutionProvider"])
        self.asr = onnx_asr.load_model(ASR_MODEL, sess_options=options(settings.asr_threads),
                                       providers=["CPUExecutionProvider"])
        biasing.install(self.asr.asr)
        self.hotword_boost = settings.hotword_boost
        self._biases: OrderedDict[tuple[str, ...], biasing.PhraseBias | None] = OrderedDict()
        self._bias_lock = threading.Lock()
        self.voices = settings.voices
        self.default_voice = settings.default_voice
        self.piper = {v: PiperVoice.load(str(paths[v])) for v in self.voices if VOICES[v].engine == "piper"}
        self.silero = SileroTts(paths["silero"]) if "silero" in paths else None
        self._espeak = threading.Lock()  # the espeak-ng phonemizer inside Piper is not thread-safe
        # Greetings and service phrases of every loaded agent: bounded by bytes, not entries, so that a
        # runtime serving many PBXs does not evict greetings between calls and re-synthesize them at call start.
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = settings.phrase_cache_mb * 1024 * 1024
        self._cache_lock = threading.Lock()
        self.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up
        if self.silero is not None:
            first = next(v for v in self.voices if VOICES[v].engine == "silero")
            self.synthesize(first, "Здравствуйте.")  # the first Silero call is several times slower
        log.info("speech models loaded: %s, voices %s", ASR_MODEL, ", ".join(self.voices))

    def new_vad(self) -> StreamVad:
        return StreamVad(self.vad_session)

    def recognize(self, audio: np.ndarray, terms: tuple[str, ...] = ()) -> str:
        bias = self.phrase_bias(terms)
        if bias is None:
            return str(self.asr.recognize(audio, sample_rate=SAMPLE_RATE)).strip()
        return str(self.asr.recognize(audio, sample_rate=SAMPLE_RATE, phrase_bias=bias)).strip()

    def phrase_bias(self, terms: tuple[str, ...]) -> biasing.PhraseBias | None:
        """The compiled bias for a term list (cached; compiling ~300 terms takes tens of milliseconds)."""
        if not terms or self.hotword_boost <= 0:
            return None
        with self._bias_lock:
            if terms in self._biases:
                self._biases.move_to_end(terms)
                return self._biases[terms]
        asr = self.asr.asr
        bias = biasing.compile_terms(list(terms), asr._vocab, asr._blank_idx, self.hotword_boost)
        with self._bias_lock:
            self._biases[terms] = bias
            while len(self._biases) > BIAS_CACHE_SIZE:
                self._biases.popitem(last=False)
        return bias

    def synthesize(self, voice_id: str, text: str, rate: int = SAMPLE_RATE) -> bytes:
        """PCM of ``text`` at ``rate`` (8 kHz, or 16 kHz for slin16 calls); a voice that is not loaded speaks with
        the default voice."""
        voice_id = voice_id if voice_id in self.voices else self.default_voice
        if VOICES[voice_id].engine == "silero":
            return self.silero.synthesize(VOICES[voice_id].speaker, text, rate)
        voice = self.piper[voice_id]
        with self._espeak:
            sentences = voice.phonemize(unstressed(text))
        chunks = []
        for phonemes in sentences:
            audio = voice.phoneme_ids_to_audio(voice.phonemes_to_ids(phonemes))
            chunks.append(audio[0] if isinstance(audio, tuple) else audio)
        if not chunks:
            return b""
        return to_pcm(np.concatenate(chunks), voice.config.sample_rate, rate)

    def phrase(self, voice_id: str, text: str, rate: int = SAMPLE_RATE) -> bytes:
        """Synthesized audio of a fixed phrase (greeting, transfer, goodbye ...), cached per voice and rate."""
        key = (voice_id, text, rate)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        pcm = self.synthesize(voice_id, text, rate)
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = pcm
                self._cache_bytes += len(pcm)
            while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= len(old)
        return pcm

    def cache_stats(self) -> dict[str, int]:
        with self._cache_lock:
            return {"entries": len(self._cache), "bytes": self._cache_bytes, "limit_bytes": self._cache_limit}
