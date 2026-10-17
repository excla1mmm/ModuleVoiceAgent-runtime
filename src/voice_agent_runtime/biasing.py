"""Phrase boosting in GigaAM's greedy transducer decoder (onnx-asr), after the LocalSTTWorker implementation.

Shallow fusion at decode time: while the decoder walks a term, the next token of the term gets a bonus on
top of the model's own score, growing with how far into the term it is; when the match breaks the progress
is lost. Guards that keep normal speech intact (tuned on real calls in LocalSTTWorker):

* the bonus starts at the third letter of a term (the first letters are shared by too many words);
* it is never negative and capped per token, so only tokens within the cap of the model's best can win;
* it changes which token is written, never whether one is: when the model picks blank, that stands.

The automaton is built over characters, so any segmentation the model picks can complete a term; matching
ignores case and the model keeps choosing the casing. Terms start at a word boundary.

Unlike LocalSTTWorker the bias is not stored on the shared model: calls recognize concurrently, so the
compiled bias travels with each ``recognize(..., phrase_bias=bias)`` call through onnx-asr's ``**kwargs``
to the decoder. A compiled ``PhraseBias`` is read-only apart from its transition cache.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import numpy as np

FREE_LETTERS = 2
LETTER_SCORE = 2.0
MAX_TOKEN_BOOST = 4.0


class PhraseTree:
    """Aho-Corasick automaton over characters with NeMo-style score deltas."""

    def __init__(self, patterns: list[str]) -> None:
        self.arcs: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.score: list[float] = [0.0]
        self.end: list[bool] = [False]
        for pattern in sorted(set(patterns)):
            node = 0
            letters = 0
            for ch in pattern:
                letters += ch != " "
                child = self.arcs[node].get(ch)
                if child is None:
                    gain = LETTER_SCORE if ch != " " and letters > FREE_LETTERS else 0.0
                    child = len(self.score)
                    self.arcs.append({})
                    self.fail.append(0)
                    self.score.append(self.score[node] + gain)
                    self.end.append(False)
                    self.arcs[node][ch] = child
                node = child
            self.end[node] = True
        queue = list(self.arcs[0].values())
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for ch, child in self.arcs[node].items():
                fallback = self.fail[node]
                while fallback and ch not in self.arcs[fallback]:
                    fallback = self.fail[fallback]
                suffix = self.arcs[fallback].get(ch, 0)
                self.fail[child] = suffix if suffix != child else 0
                queue.append(child)
        self.start = self.arcs[0].get(" ", 0)  # an utterance starts at a word boundary

    def step(self, node: int, text: str) -> tuple[int, float]:
        """Feed one token's text: (next node, score delta)."""
        delta = 0.0
        for ch in text:
            while ch not in self.arcs[node] and node:
                fallback = self.fail[node]
                if not self.end[node]:
                    delta += self.score[fallback] - self.score[node]
                node = fallback
            child = self.arcs[node].get(ch)
            if child is None:
                continue
            delta += self.score[child] - self.score[node]
            node = child
        return node, delta


class PhraseBias:
    """Terms compiled against one model's vocabulary."""

    def __init__(self, tree: PhraseTree, vocab: dict[int, str], alpha: float, terms: int):
        self.tree, self.vocab, self.alpha, self.terms = tree, vocab, alpha, terms
        self._cache: dict[tuple[int, int], tuple[int, float]] = {}

    def transition(self, node: int, token: int) -> tuple[int, float]:
        key = (node, token)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = self.tree.step(node, self.vocab.get(token, ""))
        return hit

    def bonus(self, node: int, token: int) -> float:
        # Never negative: greedy decoding keeps one hypothesis, a penalty for leaving a term could only make it
        # drop or swap the real next token. Leaving a term just ends the bonus.
        return min(max(self.alpha * self.transition(node, token)[1], 0.0), MAX_TOKEN_BOOST)

    def select(self, scores: np.ndarray, node: int, blank: int) -> int:
        """Best token after the bonus; blank chosen by the model stands."""
        best = int(scores.argmax())
        if best == blank:
            return best
        floor = scores[best] - MAX_TOKEN_BOOST
        choice, choice_score = best, -np.inf
        for token in np.flatnonzero(scores >= floor):
            token = int(token)
            score = scores[token] + (0.0 if token == blank else self.bonus(node, token))
            if score > choice_score:
                choice, choice_score = token, score
        return choice


def compile_terms(terms: list[str], vocab: dict[int, str], blank: int, alpha: float) -> PhraseBias | None:
    """Keeps the terms the vocabulary can spell (ignoring case); None when none is left or alpha is 0."""
    if alpha <= 0:
        return None
    pieces = {text.casefold() for token, text in vocab.items() if token != blank and text and not text.startswith("<")}
    patterns = [" " + term.casefold() for term in terms if _spellable(" " + term.casefold(), pieces)]
    if not patterns:
        return None
    return PhraseBias(PhraseTree(patterns), {t: text.casefold() for t, text in vocab.items()}, alpha, len(patterns))


def _spellable(text: str, pieces: set[str]) -> bool:
    longest = max(map(len, pieces), default=0)
    reachable = [True] + [False] * len(text)
    for end in range(1, len(text) + 1):
        reachable[end] = any(reachable[start] and text[start:end] in pieces
                             for start in range(max(0, end - longest), end))
    return reachable[-1]


def _transducer_decoding(asr: Any, bias: PhraseBias, encoder_out: np.ndarray,
                         encoder_out_lens: np.ndarray) -> Iterator:
    """onnx-asr 0.12.0 (pinned) greedy transducer decoding; only the token choice differs."""
    blank = asr._blank_idx
    if asr.use_low_precision:
        encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])
    for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
        prev_state = asr._create_state()
        tokens: list[int] = []
        timestamps: list[int] = []
        node = bias.tree.start
        t = 0
        emitted_tokens = 0
        while t < encodings_len:
            logits, step, state = asr._decode(tokens, prev_state, encodings[t])
            token = bias.select(logits, node, blank)  # stock: logits.argmax()
            if token != blank:
                node = bias.transition(node, token)[0]
                prev_state = state
                tokens.append(token)
                timestamps.append(t)
                emitted_tokens += 1
            if step > 0:
                t += step
                emitted_tokens = 0
            elif token == blank or emitted_tokens == asr._max_tokens_per_step:
                t += 1
                emitted_tokens = 0
        yield tokens, timestamps, None


def install(asr: Any) -> None:
    """Routes the model's decoding through the bias passed as ``phrase_bias`` (stock decoding without it)."""
    from onnx_asr.asr import _AsrWithTransducerDecoding

    if not isinstance(asr, _AsrWithTransducerDecoding):
        raise TypeError(f"phrase boosting supports transducer models, not {type(asr).__name__}")
    stock = asr._decoding

    def decoding(self: Any, encoder_out: np.ndarray, encoder_out_lens: np.ndarray, /, **kwargs: object) -> Iterator:
        bias = kwargs.pop("phrase_bias", None)
        if bias is None:
            return stock(encoder_out, encoder_out_lens, **kwargs)
        return _transducer_decoding(self, bias, encoder_out, encoder_out_lens)  # type: ignore[arg-type]

    asr._decoding = types.MethodType(decoding, asr)
