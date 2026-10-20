import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from threading import Event as ThreadEvent

import pytest

from conftest import bundle_data
from test_server import make_runtime
from voice_agent_runtime.bundle import parse_bundle
from voice_agent_runtime.call import CallSession, Turn
from voice_agent_runtime.guards import ToolDecision
from voice_agent_runtime.protocol import CallParams
from voice_agent_runtime.turns import Event


class Socket:
    def __init__(self):
        self.commands = []
        self.audio = asyncio.Event()

    def __aiter__(self):
        return self.messages()

    async def send_str(self, text):
        self.commands.append(text)

    async def send_bytes(self, pcm):
        self.audio.set()


def session():
    bundle = parse_bundle("3", bundle_data(transfer_grace_seconds=0))
    return CallSession(make_runtime(), Socket(), "pbx-acme", CallParams("7999", "c1", "3"), bundle, "", None)


def heard(call, text):
    """Recognition of the caller's audio gives ``text`` (an interruption is confirmed by it)."""
    def recognize(start, end):
        future = Future()
        future.set_result(text)
        return future
    call._recognize_range = recognize


def streaming_turn(call):
    q = asyncio.Queue()
    turn = Turn(call, [], q.put_nowait)
    turn.stt_text = "Соедините с директором"
    turn.reply = "Соединяю"
    turn.decision = ToolDecision("transfer_call", {"extension": "100"})
    turn.transfer_target = "101"
    turn.future = Future()
    turn.future.set_running_or_notify_cancel()
    call.turn = turn
    q.put_nowait(b"\0" * 320)
    task = call._spawn(call.answer(turn, q))
    return turn, q, task


@pytest.mark.parametrize("interrupt", ["barge", "close"])
def test_streaming_turn_is_cancelled_and_cannot_transfer(interrupt):
    async def main():
        call = session()
        turn, q, task = streaming_turn(call)
        try:
            await asyncio.wait_for(call.ws.audio.wait(), 1)
            if interrupt == "barge":
                heard(call, "Подождите")
                call.on_event(Event.BARGE_IN, 0)
                call.on_event(Event.CALLER_SPEECH, 0)
                await asyncio.sleep(0.02)
            else:
                await call.close()
            await asyncio.sleep(0)
            turn.future.set_result(None)
            q.put_nowait(None)
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0.02)
            assert "HANGUP" not in call.ws.commands
            assert call.transfer is None
            assert turn.cancelled.is_set()
        finally:
            if not turn.future.done():
                turn.future.set_result(None)
            if not call.finished:
                await call.close()
    asyncio.run(main())


@pytest.mark.parametrize("unavailable", [None, "capacity"])
def test_local_greeting_does_not_wait_for_provider_connection(unavailable):
    async def main():
        call = session()
        call.unavailable_reason = unavailable
        async def messages():
            yield SimpleNamespace(type=1, data="MEDIA_START format:slin")
            await call.ws.audio.wait()
        call.ws.messages = messages
        release = ThreadEvent()
        call.llm.warm = lambda: release.wait(1)
        try:
            await asyncio.wait_for(call.run(), 0.2)
            assert call.ws.audio.is_set()
        finally:
            release.set()
            if not call.finished:
                await call.close()
    asyncio.run(main())


def test_cancel_before_first_audio_releases_answer_task():
    async def main():
        call = session()
        q = asyncio.Queue()
        turn = Turn(call, [], q.put_nowait)
        turn.future = Future()  # queued behind another task, no worker can deliver a sentinel
        call.turn, call.audio_q = turn, q
        task = call._spawn(call.answer(turn, q))
        await asyncio.sleep(0)
        heard(call, "Стойте")
        call.on_event(Event.CALLER_SPEECH, 0)
        call.on_event(Event.BARGE_IN, 0)
        try:
            await asyncio.wait_for(asyncio.shield(task), 0.1)
            assert not call.history
        finally:
            await call.close()
    asyncio.run(main())


@pytest.mark.parametrize("text, interrupted", [("Здравствуй.", False), ("", False), ("Подождите, у меня вопрос", True),
                                                ("Здравствуйте. Ну.", False), ("Угу.", False), ("Стоп.", True)])
def test_an_interruption_by_the_agent_echo_or_noise_is_ignored(text, interrupted):
    async def main():
        call = session()
        call.echo.said("Здравствуйте. Я Маша, голосовой помощник компании.")
        turn, q, task = streaming_turn(call)
        try:
            await asyncio.wait_for(call.ws.audio.wait(), 1)
            heard(call, text)
            call.on_event(Event.BARGE_IN, 0)
            await asyncio.sleep(0.02)
            assert turn.cancelled.is_set() is interrupted
            assert ("FLUSH_MEDIA" in call.ws.commands) is interrupted
        finally:
            if not turn.future.done():
                turn.future.set_result(None)
            q.put_nowait(None)
            await asyncio.gather(task, return_exceptions=True)
            if not call.finished:
                await call.close()
    asyncio.run(main())


def test_a_turn_of_the_agent_echo_is_not_answered():
    async def main():
        call = session()
        call.echo.said("Здравствуйте! Чем могу помочь?")
        replies = []
        call.llm.stream = lambda *a, **k: replies.append(a) or iter(())
        for text, echo_possible in (("Здравствуйте.", True), ("Здравствуйте.", False)):
            q = asyncio.Queue()
            done = Future()
            done.set_result(text)
            turn = Turn(call, [done], q.put_nowait, echo_possible=echo_possible)
            turn.run()
            items = [q.get_nowait() for _ in range(q.qsize())]
            assert items[-1] is None and (len(items) == 1) is echo_possible  # echo: no audio, only the end mark
        assert len(replies) == 1  # only the turn that started after the agent fell silent reached the model
        await call.close()
    asyncio.run(main())


def test_slow_recognition_at_the_start_of_a_reply_waits_for_the_words():
    """Recognition slower than BARGE_IN_CONFIRM_S used to interrupt blindly; in the first seconds of a reply,
    where the canceller may still leave echo, it waits for the words and the echo does not cut the agent off."""
    async def main(monkeypatch_confirm):
        call = session()
        call.echo.said("Здравствуйте. Я Маша, голосовой помощник компании.")
        slow = Future()
        call._recognize_range = lambda start, end: slow
        turn, q, task = streaming_turn(call)
        try:
            await asyncio.wait_for(call.ws.audio.wait(), 1)
            call.on_event(Event.BARGE_IN, 0)
            await asyncio.sleep(monkeypatch_confirm + 0.05)
            assert not turn.cancelled.is_set()  # past the confirm time: still waiting for the words
            slow.set_result("Здравствуйте.")
            await asyncio.sleep(0.02)
            assert not turn.cancelled.is_set() and "FLUSH_MEDIA" not in call.ws.commands
        finally:
            if not turn.future.done():
                turn.future.set_result(None)
            q.put_nowait(None)
            await asyncio.gather(task, return_exceptions=True)
            if not call.finished:
                await call.close()
    import voice_agent_runtime.call as call_module
    confirm = call_module.BARGE_IN_CONFIRM_S
    call_module.BARGE_IN_CONFIRM_S = 0.05
    try:
        asyncio.run(main(0.05))
    finally:
        call_module.BARGE_IN_CONFIRM_S = confirm


def test_a_backchannel_over_the_talking_agent_is_not_answered():
    async def main():
        call = session()
        call.echo.said("Здравствуйте! Мы разрабатываем бесплатную айпи-АТС.")
        replies = []
        call.llm.stream = lambda *a, **k: replies.append(a) or iter(())
        call.playback = call.loop.create_future()  # the agent is still talking
        for text in ("Здравствуйте. Ну.", "Угу."):
            q = asyncio.Queue()
            done = Future()
            done.set_result(text)
            Turn(call, [done], q.put_nowait, echo_possible=True).run()
            assert [q.get_nowait() for _ in range(q.qsize())] == [None]
        assert not replies
        call.playback.set_result(None)  # the agent has finished: "Да." right after its question is an answer
        done = Future()
        done.set_result("Да.")
        Turn(call, [done], asyncio.Queue().put_nowait, echo_possible=True).run()
        assert len(replies) == 1
        call.playback = None
        await call.close()
    asyncio.run(main())


def test_repeated_interruption_events_recognize_only_the_current_phrase_once():
    async def main():
        call = session()
        ranges = []

        def recognize(start, end):
            ranges.append((round(start, 2), round(end, 2)))
            future = Future()
            future.set_result("")  # noise: not an interruption
            return future
        call._recognize_range = recognize
        call.detector.pos, call.detector.speech_start = 17.5, 17.2
        try:
            for _ in range(20):  # BARGE_IN comes on every voiced hop while the agent talks
                call.on_event(Event.BARGE_IN, 0)
            await asyncio.sleep(0.02)
            for _ in range(20):
                call.on_event(Event.BARGE_IN, 0)
            await asyncio.sleep(0.02)
            assert ranges == [(17.0, 17.5)]  # once, from the phrase start (with pre-roll), never from 0
            call.detector.pos = 18.2  # more of the phrase: checked again
            call.on_event(Event.BARGE_IN, 0)
            await asyncio.sleep(0.02)
            assert ranges == [(17.0, 17.5), (17.0, 18.2)]
        finally:
            await call.close()
    asyncio.run(main())
