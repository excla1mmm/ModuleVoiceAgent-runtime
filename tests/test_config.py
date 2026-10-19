import json

import pytest

from voice_agent_runtime.config import DEFAULT_LLM_MODEL, SUPPORTED_VOICES, VOICES, ConfigError, Settings
from voice_agent_runtime.tenants import (TenantRegistry, authenticate, basic_credentials, bearer_credentials,
                                         parse_tenants, token_hash)


def test_defaults_use_cloud_ru_gpt_oss():
    s = Settings.from_env({"LLM_API_KEY": "k"})
    assert s.primary.model == DEFAULT_LLM_MODEL and s.primary.extra == {"reasoning_effort": "low"}
    assert s.secondary is None and s.hedge_ms == 600 and s.turn_end_ms == 600 and s.question_end_ms == 300
    assert not s.tls_enabled and s.default_voice == "ru_RU-dmitri-medium" and not s.log_transcripts


def test_secondary_provider_and_errors(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    s = Settings.from_env({"LLM_API_KEY": "k", "LLM2_URL": "https://example.org/v1/chat/completions",
                           "LLM2_MODEL": "m2", "LLM2_API_KEY_FILE": str(key_file), "LLM2_EXTRA_JSON": "{}"})
    assert s.secondary.api_key == "from-file" and s.secondary.model == "m2"
    with pytest.raises(ConfigError):
        Settings.from_env({})
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_TLS_CERT": "/c"})
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_DEFAULT_VOICE": "ru_RU-irina-medium"})
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "LLM2_URL": "https://x/v1", "LLM2_API_KEY": "k"})  # no model


def test_tenants_file_reload(tmp_path, monkeypatch):
    path = tmp_path / "tenants.json"
    path.write_text(json.dumps({"tenants": [{"id": "pbx-a", "token_sha256": [token_hash("one")]}]}))
    registry = TenantRegistry(path)
    assert registry.verify("pbx-a", "one") and not registry.verify("pbx-a", "two")
    path.write_text(json.dumps({"tenants": [{"id": "pbx-a", "token_sha256": [token_hash("one"), token_hash("two")]}]}))
    import os
    os.utime(path, (1, 1))
    registry._checked_at = 0
    assert registry.verify("pbx-a", "two")  # rotation: both tokens valid
    path.write_text("not json")
    os.utime(path, (2, 2))
    registry._checked_at = 0
    assert registry.verify("pbx-a", "two")  # a broken file keeps the loaded tenants


def test_parse_tenants_rejects_bad_entries():
    with pytest.raises(ValueError):
        parse_tenants({"tenants": [{"id": "x", "token_sha256": "abc"}]})
    with pytest.raises(ValueError):
        parse_tenants({"tenants": [{"id": "pbx-a", "token_sha256": "zz"}]})


def test_credentials_parsing():
    assert basic_credentials("Basic cGJ4LWE6c2VjcmV0") == ("pbx-a", "secret")
    assert basic_credentials("Basic !!!") is None
    assert bearer_credentials("Bearer pbx-a.secret.with.dots") == ("pbx-a", "secret.with.dots")
    assert bearer_credentials("Bearer nodot") is None
    registry = TenantRegistry.from_mapping({"tenants": [{"id": "pbx-a", "token_sha256": token_hash("secret")}]})
    assert authenticate("Bearer pbx-a.secret", registry, "bearer") == "pbx-a"
    assert authenticate("Bearer pbx-a.secret", registry, "basic") is None


def test_speech_and_usage_settings():
    s = Settings.from_env({"LLM_API_KEY": "k"})
    assert s.stt_chunks and s.stt_chunk_pause_ms == 160 and s.stt_chunk_min_ms == 1000
    assert s.hotwords and s.hotword_boost == 0.45 and s.primary.stream_usage
    assert str(s.usage_db) == "/data/usage.sqlite3" and s.usage_flush_s == 30
    s = Settings.from_env({"LLM_API_KEY": "k", "VA_STT_CHUNKS": "false", "VA_HOTWORDS": "no", "VA_USAGE_DB": "",
                           "LLM_STREAM_USAGE": "false", "VA_HOTWORD_BOOST": "0"})
    assert not s.stt_chunks and not s.hotwords and s.usage_db is None and not s.primary.stream_usage
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_STT_CHUNK_PAUSE_MS": "10"})


def test_voices_to_load():
    s = Settings.from_env({"LLM_API_KEY": "k"})
    assert s.voices == SUPPORTED_VOICES and s.default_voice == "ru_RU-dmitri-medium"
    assert {"silero-ekaterina", "silero-oksana", "silero-zinaida", "silero-roman", "silero-eduard"} <= set(s.voices)
    s = Settings.from_env({"LLM_API_KEY": "k", "VA_VOICES": " silero-oksana, ru_RU-dmitri-medium,silero-oksana ",
                           "VA_DEFAULT_VOICE": "silero-oksana"})
    assert s.voices == ("silero-oksana", "ru_RU-dmitri-medium") and s.default_voice == "silero-oksana"
    for env in ({"VA_VOICES": "silero-xenia"}, {"VA_VOICES": " , "},
                {"VA_VOICES": "silero-oksana"},  # the default voice dmitri is not loaded
                {"VA_DEFAULT_VOICE": "ru_RU-ruslan-medium"}):
        with pytest.raises(ConfigError):
            Settings.from_env({"LLM_API_KEY": "k", **env})


def test_voice_catalog_has_commercial_voices_only():
    assert set(VOICES) == {"ru_RU-dmitri-medium", "ru_RU-denis-medium", "silero-ekaterina", "silero-oksana",
                           "silero-zinaida", "silero-roman", "silero-eduard"}
    assert all(v.id == key and v.engine in ("piper", "silero") and v.gender in ("male", "female") and v.name
               for key, v in VOICES.items())
    assert all(v.speaker.startswith("ru_") for v in VOICES.values() if v.engine == "silero")


def test_topic_filter_and_timezone():
    s = Settings.from_env({"LLM_API_KEY": "k"})
    assert s.topic_filter and s.timezone == "Europe/Moscow"
    s = Settings.from_env({"LLM_API_KEY": "k", "VA_TOPIC_FILTER": "false", "VA_TIMEZONE": "Asia/Yekaterinburg"})
    assert not s.topic_filter and s.timezone == "Asia/Yekaterinburg"
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_API_KEY": "k", "VA_TIMEZONE": "Mars/Base"})
