"""One phone call on the media WebSocket.

Pipeline (ported from the prototype that was tested on real calls): greeting from the phrase cache; Silero
VAD per 32 ms hop; after 120 ms of silence a speculative turn (GigaAM -> LLM stream with the conversation
history -> clause-split TTS -> audio queue) runs in the call's worker thread; the turn ends after 600 ms of
silence (300 ms after a question) and its audio is played, paced at 20 ms per frame. Caller speech over the
agent interrupts it (FLUSH_MEDIA); caller speech after a transfer/goodbye phrase cancels the hangup.
The caller audio goes through an echo canceller (the agent audio as the reference) before VAD and recognition;
caller speech over the agent that the canceller took most of, or that is only the agent's own recent words
(echo from a speakerphone, ``echo.py``), neither interrupts it nor gets an answer; an interruption is confirmed by recognition first (at most
``BARGE_IN_CONFIRM_S``, after which it interrupts as before).
Long turns are recognized incrementally (``incremental.py``): speech before an internal pause is recognized
while the caller goes on, so the end of the turn waits only for the last phrase. Recognition is biased
towards the agent's terms (``hotwords.py``). When the call ends its usage counters go to the usage meter.

Lessons kept from real calls:
* the end-of-audio sentinel is put even when a turn fails (a lost sentinel froze calls);
* a hangup scheduled after a goodbye or transfer phrase is cancelled with a token check, not task.cancel()
  (waiting for playback swallowed CancelledError and the call dropped anyway);
* tool calls go to the history as assistant ``tool_calls`` + ``tool`` messages (plain-text markers made
  gpt-oss imitate JSON).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np

from .bundle import AgentBundle
from .echo import ECHO_REMOVED_DB, EchoGuard, level_db, new_canceller
from .guards import (ANYTHING_ELSE, PHRASES, TOOL_RESULTS, CallContext, ReplyFilter, ToolDecision, decide,
                     last_question, options_phrase)
from .hotwords import abbreviation_fixes, correct
from .incremental import Utterance, join_pieces
from .llm import LlmError, LlmSession
from .normalize import normalize
from .prompt import system_prompt, tools
from .protocol import (CMD_FLUSH_MEDIA, CMD_HANGUP, EVENT_MEDIA_START, FRAME_SECONDS, SAMPLE_RATE, CallParams,
                       MediaStart, ProtocolError, frames, parse_control)
from .turns import Event, TurnDetector

log = logging.getLogger("voice_agent.call")
HOP = 256
HOP_BYTES = HOP * 2
KEEP_AUDIO_S = 30
ECHO_TAIL_S = 0.5  # caller speech this soon after the agent fell silent may still be its echo (~0.33 s seen)
BARGE_IN_CONFIRM_S = 0.8  # the longest an interruption waits for recognition
BARGE_IN_AUDIO_S = 0.6  # caller speech collected before it is recognized (130 ms is too short to recognize)


class Turn:
    """One speculative answer: STT -> LLM -> guards -> TTS clauses, run in the call's worker thread."""

    def __init__(self, call: CallSession, pieces: list[Future], deliver, echo_possible: bool = False,
                 echo_removed: bool = False):
        self.call = call
        self.echo_possible = echo_possible  # the caller started speaking while the agent was audible
        self.echo_removed = echo_removed  # ... and the echo canceller took most of that speech off
        self.pieces = pieces  # texts of the turn's audio pieces, recognized in order by the call's worker
        self.deliver = deliver  # thread-safe: puts PCM (or None = end of audio) into the call's audio queue
        self.cancelled = threading.Event()
        self.t_start = time.perf_counter()
        self.stt_text = ""
        self.reply = ""
        self.decision = ToolDecision(None)
        self.context = dataclasses.replace(call.context)
        self.transfer_target: str | None = None
        self.timings: dict[str, int] = {}
        self.provider = ""
        self.failed = False
        self.spoken_chars = 0
        self.future: Future | None = None

    def run(self) -> None:
        try:
            self._run()
        except Exception:  # noqa: BLE001 - the call goes on; the turn is logged as failed
            self.failed = True
            log.exception("turn failed", extra={"call": self.call.id})
        finally:
            self.deliver(None)  # the answer task must never wait forever

    def _ms(self) -> int:
        return round((time.perf_counter() - self.t_start) * 1000)

    def _run(self) -> None:
        call = self.call
        try:
            self.stt_text = join_pieces([piece.result() for piece in self.pieces])
        except CancelledError:  # a piece was dropped: the caller resumed and this turn is obsolete
            return
        self.timings["stt_ms"] = self._ms()
        call.report_transcript(self, self.stt_text)
        if not self.stt_text or self.cancelled.is_set():
            return
        if self.echo_removed or (self.echo_possible and call.echo.is_echo(self.stt_text)):
            log.info("echo ignored", extra={"call": call.id})
            call.metrics.inc("echo_ignored")
            return
        messages = [{"role": "system", "content": call.system}, *call.history,
                    {"role": "user", "content": self.stt_text}]
        reply_filter, tool, args = ReplyFilter(), None, ""
        try:
            for kind, payload in call.llm.stream(messages, call.tools, self.cancelled, on_hedge=call.on_hedge):
                if kind == "tool":
                    tool, args = payload
                    continue
                for clause in reply_filter.feed(payload):
                    self._say(clause)
        except LlmError as exc:
            log.warning("llm failed: %s", exc, extra={"call": call.id})
            call.metrics.inc("llm_errors")
            self._fail_over()
            return
        if self.cancelled.is_set():
            return
        self.provider = call.llm.last_provider
        for clause in reply_filter.finish():
            self._say(clause)
        decision = decide(tool, args, self.stt_text, reply_filter.reply, self.context)
        if decision.list_options:
            decision = dataclasses.replace(decision, phrase=options_phrase([q.name for q in call.bundle.queues]))
        if decision.tool == "transfer_call":
            self.transfer_target = call.bundle.resolve_transfer(str(decision.args.get("extension", "")))
            if self.transfer_target is None:  # the model named a number that is not in the directory
                decision = ToolDecision("transfer_call", decision.args, phrase=PHRASES["transfer_denied"],
                                        tool_result="Такого номера нет в справочнике, перевод не выполнен.")
        if decision.phrase:
            self._say(decision.phrase, cached=decision.phrase_cached)
        self.decision = decision
        self.reply = " ".join(p for p in (reply_filter.reply, decision.phrase) if p).strip()

    def _fail_over(self) -> None:
        """Both LLM providers failed: apologize and hand the caller over to a person."""
        self.decision = ToolDecision("fallback")
        self.reply = PHRASES["unavailable"]
        self._say(self.reply, cached=True)

    def _say(self, text: str, cached: bool = False) -> None:
        text = text.strip()
        if not text or self.cancelled.is_set():
            return
        if "llm_first_ms" not in self.timings:
            self.timings["llm_first_ms"] = self._ms() - self.timings.get("stt_ms", 0)
        engines, voice = self.call.engines, self.call.bundle.voice
        pcm = engines.phrase(voice, text) if cached else engines.synthesize(voice, normalize(text))
        self.spoken_chars += len(text)
        self.call.echo.said(text)
        self.timings.setdefault("first_audio_ms", self._ms())
        self.deliver(pcm)


class CallSession:
    def __init__(self, runtime, ws, tenant: str, params: CallParams, bundle: AgentBundle | None,
                 static_prompt: str, unavailable_reason: str | None, terms: tuple[str, ...] = ()):
        self.runtime = runtime
        self.settings = runtime.settings
        self.engines = runtime.engines
        self.metrics = runtime.metrics
        self.ws = ws
        self.tenant, self.params = tenant, params
        self.bundle = bundle
        self.unavailable_reason = unavailable_reason if bundle is not None else (unavailable_reason or "bundle_missing")
        self.id = uuid.uuid4().hex[:12]  # runtime-internal id for technical logs (no phone numbers there)
        self.loop = asyncio.get_running_loop()
        self.system = system_prompt(static_prompt, params.caller) if bundle else ""
        self.tools = tools(bundle) if bundle else []
        self.context = CallContext(caller=params.caller, callbacks_enabled=bool(bundle and bundle.callbacks_enabled))
        self.history: list[dict] = []
        self.detector = TurnDetector(HOP / SAMPLE_RATE, self.settings.speculate_ms, self.settings.turn_end_ms,
                                     self.settings.question_end_ms, self.settings.barge_in_frames,
                                     self.settings.vad_threshold,
                                     self.settings.stt_chunk_pause_ms if self.settings.stt_chunks else 0)
        self.utterance: Utterance | None = None
        self.echo = EchoGuard()
        self.aec = new_canceller() if self.settings.aec else None
        self.raw_pcm = bytearray()  # caller audio before echo cancellation, aligned with self.pcm
        self.rawbuf = b""
        self.agent_audio_pos = -1.0  # caller-audio position (s) at which the agent was last audible
        self.barge_check: asyncio.Task | None = None  # the pending check of an interruption
        self.barge_start = -1.0  # start of the phrase it checks
        self.barge_next_pos = 0.0  # a phrase found "not an interruption" is checked again from here
        self.terms = terms if self.settings.hotwords else ()
        self.term_fixes = abbreviation_fixes(self.terms)
        self.vad = self.engines.new_vad()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"call-{self.id}")
        self.llm = LlmSession(self.settings.primary, self.settings.secondary, self.settings.hedge_ms,
                              self.settings.temperature, self.settings.max_tokens)
        self.pcm = bytearray()  # caller audio, last KEEP_AUDIO_S..2*KEEP_AUDIO_S seconds
        self.pcm_base = 0.0  # seconds of caller audio dropped from the front of self.pcm
        self.inbuf = b""
        self.turn: Turn | None = None
        self.audio_q: asyncio.Queue | None = None
        self.playback: asyncio.Task | None = None
        self.pending_hangup: object | None = None
        self.frame_bytes = 320
        self.xon = asyncio.Event()
        self.xon.set()
        self.started = time.time()
        self.media_started = False
        self.finished = False
        self.callback: dict | None = None
        self.transfer: dict | None = None
        self.fallback_reason: str | None = None
        self.goodbye = False
        self.turn_log: list[dict] = []
        self.speech_end_wall: float | None = None
        self.last_voice_wall: float | None = None
        self.tasks: set[asyncio.Task] = set()
        self.played_bytes = 0  # agent audio sent to the caller (usage)
        self.spoken_chars = 0  # characters of the phrases played to the caller (usage)

    # --- events from the WebSocket -----------------------------------------------------------------

    async def run(self) -> None:
        from aiohttp import WSMsgType

        self.runtime.results.start(self.tenant, self.params.call_id)
        deadline = self.loop.call_later(self.settings.max_call_s, lambda: self._spawn(self._max_duration()))
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    await self.on_text(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    if self.media_started and self.unavailable_reason is None:
                        self.on_media(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            deadline.cancel()
            await self.close()

    async def on_text(self, text: str) -> None:
        try:
            msg = parse_control(text)
        except ProtocolError as exc:
            log.warning("bad control message: %s", exc, extra={"call": self.id})
            return
        if msg.name == EVENT_MEDIA_START and not self.media_started:
            try:
                start = MediaStart.from_control(msg)
            except ProtocolError as exc:
                log.error("cannot start media: %s", exc, extra={"call": self.id})
                await self.hangup(error=str(exc))
                return
            self.media_started = True
            self.frame_bytes = start.optimal_frame_size if start.optimal_frame_size % 2 == 0 else 320
            await self.start()
        elif msg.name == "MEDIA_XOFF":
            self.xon.clear()
        elif msg.name == "MEDIA_XON":
            self.xon.set()

    async def start(self) -> None:
        if self.unavailable_reason is not None:
            voice = self.bundle.voice if self.bundle else self.settings.default_voice
            pcm = await self.loop.run_in_executor(self.worker, self.engines.phrase, voice, PHRASES["unavailable"])
            self.spoken_chars += len(PHRASES["unavailable"])
            self.fallback_reason = self.unavailable_reason
            await self.play([pcm])
            self._spawn(self._hangup_after_playback(0.3))
            return
        greeting = await self.loop.run_in_executor(self.worker, self.engines.phrase, self.bundle.voice,
                                                   normalize(self.bundle.greeting))
        self.spoken_chars += len(self.bundle.greeting)
        self.echo.said(self.bundle.greeting)
        self.history.append({"role": "assistant", "content": self.bundle.greeting})
        await self.play([greeting])

    def on_media(self, data: bytes) -> None:
        raw, clean = self.aec.capture(data) if self.aec is not None else (data, data)
        self.inbuf += clean
        self.rawbuf += raw
        while len(self.inbuf) >= HOP_BYTES:
            hop_bytes, self.inbuf = self.inbuf[:HOP_BYTES], self.inbuf[HOP_BYTES:]
            raw_hop, self.rawbuf = self.rawbuf[:HOP_BYTES], self.rawbuf[HOP_BYTES:]
            self._keep_audio(hop_bytes, raw_hop)
            hop = np.frombuffer(hop_bytes, dtype=np.int16).astype(np.float32) / 32768
            if self.speaking:
                self.agent_audio_pos = self.detector.pos
            prob = self.vad.prob(hop)
            if prob >= self.settings.vad_threshold:
                self.last_voice_wall = time.perf_counter()
            for emitted in self.detector.push(prob, agent_speaking=self.speaking):
                self.on_event(emitted.event, emitted.segment_start)

    def _keep_audio(self, hop_bytes: bytes, raw_hop: bytes | None = None) -> None:
        self.pcm += hop_bytes
        self.raw_pcm += hop_bytes if raw_hop is None else raw_hop
        limit = 2 * KEEP_AUDIO_S * SAMPLE_RATE * 2
        if len(self.pcm) > limit:
            drop = len(self.pcm) - KEEP_AUDIO_S * SAMPLE_RATE * 2
            del self.pcm[:drop]
            del self.raw_pcm[:drop]
            self.pcm_base += drop / 2 / SAMPLE_RATE

    def _segment(self, start_s: float, end_s: float, raw: bool = False) -> np.ndarray:
        offset = max(0, int((start_s - self.pcm_base) * SAMPLE_RATE)) * 2
        end = max(offset, int((end_s - self.pcm_base) * SAMPLE_RATE) * 2)
        pcm = self.raw_pcm if raw else self.pcm
        return np.frombuffer(bytes(pcm[offset:end]), dtype=np.int16).astype(np.float32) / 32768

    def _echo_removed(self, start_s: float, end_s: float) -> bool:
        """The echo canceller took most of this caller audio off: it was the agent's echo."""
        if self.aec is None:
            return False
        removed = level_db(self._segment(start_s, end_s, raw=True)) - level_db(self._segment(start_s, end_s))
        return removed >= ECHO_REMOVED_DB

    def _recognize_range(self, start_s: float, end_s: float) -> Future:
        """Submits caller audio start..end (seconds of the call) to the worker; the future gives the text."""
        return self.worker.submit(self._recognize, self._segment(start_s, end_s))

    def _recognize(self, audio: np.ndarray) -> str:
        return correct(self.engines.recognize(audio, self.terms), self.term_fixes)

    def _utterance(self, segment_start: float) -> Utterance:
        """The incremental transcript of the caller turn that starts at ``segment_start``."""
        if self.utterance is None or self.utterance.start != segment_start:
            if self.utterance is not None:
                self.utterance.close()
            self.utterance = Utterance(segment_start, self._recognize_range, self.settings.stt_chunk_min_ms / 1000)
        return self.utterance

    def on_event(self, event: Event, segment_start: float) -> None:
        if event is Event.SPECULATE:
            pieces = self._utterance(segment_start).speculate(self.detector.pos)
            self.audio_q = asyncio.Queue()
            q = self.audio_q
            echo_possible = segment_start <= self.agent_audio_pos + ECHO_TAIL_S
            turn = Turn(self, pieces, lambda pcm: self.loop.call_soon_threadsafe(q.put_nowait, pcm),
                        echo_possible=echo_possible,
                        echo_removed=echo_possible and self._echo_removed(segment_start, self.detector.pos))
            self.turn = turn
            turn.future = self.worker.submit(turn.run)
        elif event is Event.CHUNK:
            utterance = self._utterance(segment_start)
            committed = len(utterance.committed)
            utterance.cut(self.detector.pos)
            if len(utterance.committed) > committed:
                self.metrics.inc("stt_pieces")
        elif event is Event.CANCEL:
            self.cancel_turn()
            if self.utterance is not None:
                self.utterance.resume()
        elif event is Event.TURN_END:
            if self.turn is not None and self.audio_q is not None:
                self.speech_end_wall = self.last_voice_wall
                self._spawn(self.answer(self.turn, self.audio_q))
        elif event is Event.BARGE_IN:
            # One check at a time, on the current phrase only; after "not an interruption" the phrase is checked
            # again once more of it has been heard. BARGE_IN repeats on every voiced hop over the agent.
            start = max(0.0, (self.detector.speech_start or self.detector.pos) - TurnDetector.PRE_ROLL_S)
            busy = self.barge_check is not None and not self.barge_check.done()
            if not busy and (start != self.barge_start or self.detector.pos >= self.barge_next_pos):
                self.barge_start = start
                self.barge_check = self._spawn(self.confirm_barge_in(start, self.turn, self.playback))
        elif event is Event.CALLER_SPEECH:
            if self.detector.turn_ended:
                self.cancel_turn()
                self.detector.reset_turn()
            if self.pending_hangup is not None:
                log.info("hangup cancelled, caller keeps talking", extra={"call": self.id})
            self.pending_hangup = None  # "нет, подождите": keep talking instead of dropping the call
            self.transfer = None
            self.goodbye = False

    async def confirm_barge_in(self, segment_start: float, turn: Turn | None, playback: asyncio.Task | None) -> None:
        """Interrupts the agent unless the caller's speech so far is silence to recognition or the agent's echo."""
        waited = 0.0
        while (self.detector.pos < segment_start + BARGE_IN_AUDIO_S and self.detector.voiced_run
               and waited < BARGE_IN_AUDIO_S + 0.4 and not self.finished):
            await asyncio.sleep(0.03)  # let more of the phrase arrive (it ends early if the caller stops)
            waited += 0.03
        end = self.detector.pos
        if self._echo_removed(segment_start, end):
            self.metrics.inc("barge_ins_ignored")
            self.barge_next_pos = end + BARGE_IN_AUDIO_S
            return
        future = self._recognize_range(segment_start, end)
        try:
            text = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), BARGE_IN_CONFIRM_S)
        except TimeoutError:
            text = None  # recognition is busy with the agent's own reply: interrupt as before
        except Exception:  # noqa: BLE001 - a failed recognition must not keep the agent talking over the caller
            text = None
        if text is not None and (not text.strip() or self.echo.is_echo(text)):
            self.metrics.inc("barge_ins_ignored")
            self.barge_next_pos = end + BARGE_IN_AUDIO_S
            return
        self.metrics.inc("barge_ins")
        if turn is not None and self.turn is turn:  # not the turn started meanwhile for the interruption
            self.cancel_turn()
        if self.playback is playback:  # not the answer to the interruption, if it is already playing
            await self.stop_playback(flush=True)

    def report_transcript(self, turn: Turn, text: str) -> None:
        """Called from the worker thread: a question shortens the end-of-turn pause."""
        def apply() -> None:
            if turn is self.turn:
                self.detector.set_transcript(text)
        self.loop.call_soon_threadsafe(apply)

    def on_hedge(self, reason: str) -> None:
        self.metrics.inc(f"llm_hedged_{reason}")

    # --- answering -----------------------------------------------------------------------------------

    def cancel_turn(self) -> None:
        if self.turn is not None:
            self.turn.cancelled.set()
            if self.turn.future is not None:
                self.turn.future.cancel()
            self.turn.deliver(None)  # a queued, cancelled worker cannot send its end-of-audio sentinel
            self.turn = None

    async def answer(self, turn: Turn, audio_q: asyncio.Queue) -> None:
        played = False
        try:
            first = await audio_q.get()
            if turn.cancelled.is_set() or self.finished or self.turn is not turn:
                return
            sent_wall = time.perf_counter()
            self.detector.reset_turn()
            if first is None:
                await asyncio.wrap_future(turn.future)
                return
            self.history.append({"role": "user", "content": turn.stt_text})
            played = True
            await self.play_queue(first, audio_q)
            await asyncio.wrap_future(turn.future)
            if not turn.cancelled.is_set() and not self.finished and self.turn is turn:
                self._commit(turn, sent_wall)
        finally:
            if played:
                self.spoken_chars += turn.spoken_chars  # as far as the turn got before it ended or was cut off
            if self.turn is turn:
                self.turn = None

    def _commit(self, turn: Turn, sent_wall: float) -> None:
        decision = turn.decision
        self.context.callback_wanted = turn.context.callback_wanted
        self.context.asked_anything_else = bool(ANYTHING_ELSE.search(last_question(turn.reply)))
        self.context.caller_said = (*self.context.caller_said, turn.stt_text)[-40:]
        if decision.tool and decision.tool != "fallback":
            call_id = f"call_{len(self.history)}"
            self.history += [
                {"role": "assistant", "content": turn.reply or None, "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": decision.tool, "arguments": json.dumps(decision.args, ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": call_id,
                 "content": decision.tool_result or TOOL_RESULTS.get(decision.tool, "Готово")},
            ]
        else:
            self.history.append({"role": "assistant", "content": turn.reply})
        if decision.callback:
            self.callback = decision.callback  # one request per call: a later call of the tool updates it
        latency = round((sent_wall - (self.speech_end_wall or sent_wall)) * 1000)
        entry = {"at": round(time.time() - self.started, 1), "caller": turn.stt_text, "agent": turn.reply,
                 "tool": decision.tool, "tool_args": decision.args or None, "latency_ms": latency,
                 "llm_provider": turn.provider, **turn.timings}
        self.turn_log.append(entry)
        self.metrics.inc("turns")
        self.metrics.inc_tenant(self.tenant, "turns")
        self.metrics.observe("latency_ms", latency)
        for stage in ("stt_ms", "llm_first_ms", "first_audio_ms"):
            if stage in turn.timings:
                self.metrics.observe(stage, turn.timings[stage])
        log_fields = {"call": self.id, "tenant": self.tenant, "agent": self.params.agent_id, "latency_ms": latency,
                      "tool": decision.tool, "provider": turn.provider, **turn.timings}
        if self.settings.log_transcripts:
            log_fields.update(caller_text=turn.stt_text, agent_text=turn.reply)
        log.info("turn", extra={"fields": log_fields})
        if decision.tool == "transfer_call" and turn.transfer_target:
            transfer = {"extension": turn.transfer_target, "requested": str(decision.args.get("extension", "")),
                        "reason": str(decision.args.get("reason", ""))[:300]}
            self._schedule_hangup(self.bundle.transfer_grace_s, transfer=transfer)
        elif decision.tool == "end_call":
            self._schedule_hangup(self.bundle.goodbye_grace_s, goodbye=True)
        elif decision.tool == "fallback":
            self._schedule_hangup(0.3, fallback="llm_unavailable")

    def _schedule_hangup(self, grace: float, transfer: dict | None = None, goodbye: bool = False,
                         fallback: str | None = None) -> None:
        token = object()
        self.pending_hangup = token
        self._spawn(self._hangup_later(token, grace, transfer, goodbye, fallback))

    async def _hangup_later(self, token: object, grace: float, transfer: dict | None, goodbye: bool,
                            fallback: str | None) -> None:
        """Hangs up after the goodbye/transfer phrase and a pause, unless the caller spoke meanwhile."""
        await self.wait_playback()
        await asyncio.sleep(grace)
        if self.pending_hangup is not token:  # caller speech reset it: the line stays open
            return
        self.pending_hangup = None
        self.transfer, self.goodbye = transfer, goodbye
        if fallback:
            self.fallback_reason = fallback
        await self.hangup()

    async def _hangup_after_playback(self, grace: float) -> None:
        await self.wait_playback()
        await asyncio.sleep(grace)
        await self.hangup()

    async def _max_duration(self) -> None:
        log.warning("maximum call duration reached", extra={"call": self.id})
        self.fallback_reason = self.fallback_reason or "max_duration"
        await self.hangup()

    def request_shutdown(self) -> None:
        """The runtime is stopping: end the leg with a fallback result so the PBX routes the caller to a person
        and can collect the result while the runtime drains."""
        if self.finished:
            return
        self.fallback_reason = self.fallback_reason or "runtime_shutdown"
        self._spawn(self.hangup())

    async def hangup(self, error: str | None = None) -> None:
        """Stores the result first (the PBX asks for it as soon as Dial() returns), then ends the leg."""
        if error:
            self.fallback_reason = self.fallback_reason or "error"
        self.finish(ended_by="agent")
        try:
            await self.ws.send_str(CMD_HANGUP)
        except (ConnectionError, RuntimeError):
            pass

    # --- playback ------------------------------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self.playback is not None and not self.playback.done()

    async def play(self, chunks: list[bytes]) -> None:
        await self.stop_playback()
        self.playback = self.loop.create_task(self._play(chunks, None))

    async def play_queue(self, first: bytes, audio_q: asyncio.Queue) -> None:
        await self.stop_playback()
        self.playback = self.loop.create_task(self._play([first], audio_q))

    async def _play(self, chunks: list[bytes], audio_q: asyncio.Queue | None) -> None:
        t0, n = self.loop.time(), 0

        async def send(pcm: bytes) -> None:
            nonlocal t0, n
            for frame in frames(pcm, self.frame_bytes):
                if not self.xon.is_set():
                    await self.xon.wait()
                    t0 = self.loop.time() - n * FRAME_SECONDS
                delay = t0 + n * FRAME_SECONDS - self.loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.ws.send_bytes(frame)
                if self.aec is not None:
                    self.aec.render(frame)  # the reference: what the caller's echo will be made of
                self.played_bytes += len(frame)
                n += 1

        for pcm in chunks:
            await send(pcm)
        if audio_q is not None:
            while (pcm := await audio_q.get()) is not None:
                await send(pcm)

    async def stop_playback(self, flush: bool = False) -> None:
        if self.playback is not None and not self.playback.done():
            self.playback.cancel()
            await asyncio.wait({self.playback})
            if flush:
                try:
                    await self.ws.send_str(CMD_FLUSH_MEDIA)
                except (ConnectionError, RuntimeError):
                    pass

    async def wait_playback(self) -> None:
        if self.playback is not None:
            await asyncio.wait({self.playback})  # never raises, whatever happened to the playback

    def _spawn(self, coro) -> asyncio.Task:
        task = self.loop.create_task(coro)
        self.tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self.tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.error("call task failed", exc_info=t.exception(), extra={"call": self.id})
        task.add_done_callback(done)
        return task

    # --- end of call ---------------------------------------------------------------------------------

    def finish(self, ended_by: str) -> None:
        if self.finished:
            return
        self.finished = True
        if self.transfer:
            outcome = "transferred"
        elif self.fallback_reason:
            outcome = "fallback"
        elif self.callback:
            outcome = "callback"
        elif self.goodbye:
            outcome = "completed"
        else:
            outcome = "caller_hangup" if ended_by == "caller" else "completed"
        result = {
            "call_id": self.params.call_id, "agent_id": self.params.agent_id, "caller": self.params.caller,
            "started_at": int(self.started), "ended_at": int(time.time()), "outcome": outcome,
            "transfer": self.transfer, "callback": self.callback, "fallback_reason": self.fallback_reason,
            "turns": self.turn_log,
        }
        evicted = self.runtime.results.finish(self.tenant, self.params.call_id, result)
        if evicted:
            self.metrics.inc("results_evicted", evicted)
            self.metrics.inc_tenant(self.tenant, "results_evicted", evicted)
        self.metrics.inc(f"outcome_{outcome}")
        self.metrics.inc_tenant(self.tenant, f"outcome_{outcome}")
        self._record_usage(outcome)
        log.info("call finished", extra={"fields": {"call": self.id, "tenant": self.tenant,
                                                    "agent": self.params.agent_id, "outcome": outcome,
                                                    "turns": len(self.turn_log),
                                                    "duration_s": round(time.time() - self.started)}})

    def _record_usage(self, outcome: str) -> None:
        """Aggregate counters only: no numbers, ids or texts leave the call."""
        llm = self.llm.usage
        self.runtime.usage.record(
            datetime.fromtimestamp(self.started, UTC).date().isoformat(), self.tenant, self.params.agent_id, {
                "calls": 1, "connected_seconds": time.time() - self.started,
                "agent_speech_seconds": self.played_bytes / 2 / SAMPLE_RATE, "caller_turns": len(self.turn_log),
                "llm_requests": llm.requests, "llm_input_tokens": llm.input_tokens,
                "llm_output_tokens": llm.output_tokens, "llm_estimated_requests": llm.estimated_requests,
                "tts_characters": self.spoken_chars, "fallbacks": int(outcome == "fallback")})

    async def close(self) -> None:
        self.finish(ended_by="caller")
        await self.stop_playback()
        self.cancel_turn()
        if self.utterance is not None:
            self.utterance.close()
        for task in list(self.tasks):
            task.cancel()
        self.worker.submit(self.llm.close)
        self.worker.shutdown(wait=False)
