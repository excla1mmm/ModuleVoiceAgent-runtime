"""A whole call over the media WebSocket: caller speech -> turn -> LLM (local scripted server) -> transfer.

Speech models are replaced by deterministic engines (voiced frames have energy, the transcript is
scripted); the WebSocket protocol, turn-taking, guards, history and the result API are the real ones.
"""

import asyncio
import json

import numpy as np
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from conftest import bundle_data
from test_llm import server, sse
from test_server import basic, bearer, make_runtime
from voice_agent_runtime.config import LlmProvider
from voice_agent_runtime.server import build_app


class ScriptedEngines:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.recognized = []  # (seconds of audio, terms) per recognition
        self.rates = []  # sample rate of every synthesis

    class Vad:
        def prob(self, hop):
            return 0.9 if np.abs(hop).max() > 0.05 else 0.0

    def new_vad(self):
        return self.Vad()

    def recognize(self, audio, terms=()):
        self.recognized.append((len(audio) / 8000, terms))
        return self.transcripts.pop(0) if self.transcripts else ""

    def synthesize(self, voice, text, rate=8000):
        self.rates.append(rate)
        return (np.ones(rate // 50, dtype=np.int16) * 500).tobytes()  # 20 ms per clause

    def phrase(self, voice, text, rate=8000):
        return self.synthesize(voice, text, rate)


SPEECH = (np.ones(8000, dtype=np.int16) * 8000).tobytes()  # 1 s of "speech"
SILENCE = b"\0" * 16000  # 1 s of silence


async def speak_and_wait(ws, stop_on_hangup=True, timeout=5.0, audio=(SPEECH, SILENCE)):
    """Sends 1 s of speech and 1 s of silence in 20 ms frames, collecting what the agent sends back."""
    commands = []

    async def receive():
        while True:
            msg = await ws.receive()
            if msg.type == WSMsgType.TEXT:
                commands.append(msg.data)
                if msg.data == "HANGUP" and stop_on_hangup:
                    return
            elif msg.type != WSMsgType.BINARY:
                return

    reader = asyncio.create_task(receive())
    for chunk in audio:
        for i in range(0, len(chunk), 320):
            await ws.send_bytes(chunk[i:i + 320])
            await asyncio.sleep(0.002)
    try:
        await asyncio.wait_for(asyncio.shield(reader), timeout)
    except TimeoutError:
        pass
    reader.cancel()
    return commands


def run_call(llm_body, transcripts, bundle, audio=(SPEECH, SILENCE), engines=None, check_body=None, media="slin"):
    """``check_body``: the topic filter is on and its requests get this answer."""
    srv, handler = server(body=llm_body, check_body=check_body)

    async def main():
        runtime = make_runtime()
        runtime.engines = engines or ScriptedEngines(transcripts)
        runtime.settings = runtime.settings.__class__(
            primary=LlmProvider("primary", f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", "m", "k"),
            aec=False,  # synthetic audio (a constant level): the high-pass filter of the canceller removes it
            topic_filter=check_body is not None)
        async with TestClient(TestServer(build_app(runtime))) as client:
            assert (await client.put("/v1/agents/3", json=bundle, headers=bearer())).status == 200
            ws = await client.ws_connect("/media?callerid=79991234567&callid=flow-1&agent=3", headers=basic(),
                                         protocols=("media",))
            frame = 640 if media == "slin16" else 320
            await ws.send_str(f"MEDIA_START connection_id:c1 format:{media} optimal_frame_size:{frame} ptime:20")
            commands = await speak_and_wait(ws, timeout=3.0, audio=audio)
            await ws.close()
            resp = await client.get("/v1/calls/flow-1/result?wait=2", headers=bearer())
            return commands, await resp.json()

    try:
        return asyncio.run(main()), handler.requests
    finally:
        srv.shutdown()


TRANSFER_TO_CEO = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
    "name": "transfer_call",
    "arguments": json.dumps({"extension": "100", "reason": "директор", "phrase": "Соединяю с приёмной."})}}]}}]})


def test_transfer_follows_route_to_and_hangs_up_after_grace():
    (commands, result), requests = run_call(TRANSFER_TO_CEO, ["Соедините с директором"],
                                            bundle_data(transfer_grace_seconds=0))
    assert commands == ["HANGUP"]
    assert result["outcome"] == "transferred"
    assert result["transfer"]["extension"] == "101" and result["transfer"]["requested"] == "100"
    turn = result["turns"][0]
    assert turn["caller"] == "Соедините с директором" and turn["agent"] == "Соединяю с приёмной."
    assert turn["tool"] == "transfer_call" and {"stt_ms", "llm_first_ms", "first_audio_ms", "latency_ms"} <= set(turn)
    body = requests[0][1]
    assert body["messages"][0]["role"] == "system" and "79991234567" in body["messages"][0]["content"]
    assert body["messages"][-1] == {"role": "user", "content": "Соедините с директором"}
    assert body["messages"][1]["role"] == "assistant"  # the greeting is part of the history


def test_a_wideband_call_speaks_at_16_khz_and_hears_the_caller_at_8():
    """slin16 (dialled for G.722 / Opus callers): the agent's audio is synthesized at 16 kHz, the caller's audio is
    brought down to the 8 kHz of the echo canceller, VAD and recognition."""
    engines = ScriptedEngines(["Соедините с директором"])
    speech16 = (np.ones(16000, dtype=np.int16) * 8000).tobytes()  # 1 s at 16 kHz
    (commands, result), _ = run_call(TRANSFER_TO_CEO, [], bundle_data(transfer_grace_seconds=0),
                                     audio=(speech16, b"\0" * 32000), engines=engines, media="slin16")
    assert result["outcome"] == "transferred" and commands == ["HANGUP"]
    assert engines.rates.count(8000) == 1  # only the greeting warmed up when the agent was loaded
    assert engines.rates.count(16000) >= 2  # the call's greeting and reply
    assert engines.recognized and 0.6 < engines.recognized[-1][0] < 1.6  # ~1 s of speech, measured at 8 kHz


def test_unknown_number_is_not_transferred():
    body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
        "name": "transfer_call", "arguments": json.dumps({"extension": "89990000000", "phrase": "Соединяю."})}}]}}]})
    (commands, result), _ = run_call(body, ["Соедините с оператором"], bundle_data(transfer_grace_seconds=0))
    assert commands == []
    assert result["outcome"] == "caller_hangup" and result["transfer"] is None
    assert "не могу соединить" in result["turns"][0]["agent"]


def test_end_call_without_farewell_keeps_talking():
    body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "end_call",
                                                                                 "arguments": "{}"}}]}}]})
    (commands, result), _ = run_call(body, ["Нет, спасибо"], bundle_data(goodbye_grace_seconds=0))
    assert commands == []
    assert result["turns"][0]["tool"] is None and result["turns"][0]["agent"] == "Хорошо. Чем ещё могу помочь?"


def test_long_turn_is_recognized_in_pieces_with_hotwords():
    """1.5 s of speech, a 250 ms pause, 1 s more: the second recognition gets only the speech after the pause."""
    usage_chunk = {"choices": [], "usage": {"prompt_tokens": 1500, "completion_tokens": 12}}
    body = sse({"choices": [{"delta": {"content": "Соединяю."}}]}, usage_chunk)
    engines = ScriptedEngines(["Соедините меня", "с ТС, пожалуйста"])
    audio = (SPEECH + SPEECH[:8000], b"\0" * 4000, SPEECH, SILENCE)
    bundle = bundle_data(hotwords=["АТС"])
    (commands, result), requests = run_call(body, [], bundle, audio=audio, engines=engines)
    assert [round(seconds, 1) for seconds, _ in engines.recognized] == [1.6, 1.2]
    assert "АТС" in engines.recognized[0][1]
    assert result["turns"][0]["caller"] == "Соедините меня с АТС, пожалуйста"  # pieces joined, "ТС" restored
    assert requests[0][1]["stream_options"] == {"include_usage": True}


def test_an_unrequested_transfer_is_offered_by_name():
    """Call 19: "есть у вас Иванова?" - the model transferred; the offer names the employee."""
    body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
        "name": "transfer_call", "arguments": json.dumps({"extension": "210", "phrase": "Соединяю."})}}]}}]})
    (commands, result), _ = run_call(body, ["А Иванова у вас работает?"], bundle_data(transfer_grace_seconds=0))
    assert commands == [] and result["transfer"] is None
    assert result["turns"][0]["agent"] == "Могу соединить: Иванова Светлана Петровна, главный бухгалтер. Соединить?"
