import pytest

from voice_agent_runtime.protocol import (FRAME_BYTES, MediaStart, ProtocolError, call_params, frames,
                                          parse_control)


def test_media_start_plain_text():
    msg = parse_control("MEDIA_START connection_id:abc123 channel:WebSocket/voiceagent/0x1 "
                        "channel_id:1736.12 format:slin optimal_frame_size:320 ptime:20")
    assert msg.name == "MEDIA_START"
    start = MediaStart.from_control(msg)
    assert (start.connection_id, start.channel, start.format, start.optimal_frame_size, start.ptime) == (
        "abc123", "WebSocket/voiceagent/0x1", "slin", 320, 20)


def test_media_start_json():
    msg = parse_control('{"event":"MEDIA_START","connection_id":"x","format":"slin","optimal_frame_size":320}')
    assert msg.name == "MEDIA_START" and MediaStart.from_control(msg).optimal_frame_size == 320


def test_other_events():
    assert parse_control("MEDIA_XOFF").name == "MEDIA_XOFF"
    with pytest.raises(ProtocolError):
        parse_control("  ")


def test_unsupported_format():
    with pytest.raises(ProtocolError, match="slin"):
        MediaStart.from_control(parse_control("MEDIA_START format:ulaw"))


def test_call_params_from_v_options():
    params = call_params("/media?callerid=%2B7%20(999)%20123-45-67&callid=mikopbx-1736.42&agent=3")
    assert (params.caller, params.call_id, params.agent_id) == ("+79991234567", "mikopbx-1736.42", "3")


@pytest.mark.parametrize("path", ["/media?agent=3", "/media?callid=a&agent=x", "/media?callid=a%20b&agent=3"])
def test_call_params_rejects(path):
    with pytest.raises(ProtocolError):
        call_params(path)


def test_frames_are_padded_to_20ms():
    out = frames(b"\x01" * (FRAME_BYTES * 2 + 10))
    assert [len(f) for f in out] == [320, 320, 320]
    assert out[-1][10:] == b"\0" * 310
    assert frames(b"") == []
