"""Silero text preparation and model download without the models (torch is not needed)."""

import hashlib

import pytest

from voice_agent_runtime.speech import SILERO_FILE, fetch_silero, silero_chunks


def test_sentences_are_packed_up_to_the_limit():
    text = "Здравствуйте! Компания Ромашка. Чем могу помочь?"
    assert silero_chunks(text) == [text]
    assert silero_chunks(text, limit=31) == ["Здравствуйте! Компания Ромашка.", "Чем могу помочь?"]
    assert silero_chunks(text, limit=30) == ["Здравствуйте!", "Компания Ромашка.", "Чем могу помочь?"]
    assert silero_chunks("  Раз.\n\nДва   три.  ", limit=100) == ["Раз. Два три."]


def test_a_long_sentence_is_cut_at_a_space():
    sentence = " ".join(["слово"] * 50)  # 299 characters, no sentence end
    pieces = silero_chunks("Начало. " + sentence, limit=100)
    assert pieces[0] == "Начало."
    assert all(len(p) <= 100 for p in pieces) and " ".join(pieces[1:]) == sentence


def test_stress_marks_and_unspeakable_pieces():
    assert silero_chunks("C++ и Си") == ["C и Си"]  # "+" would be read as a stress mark
    assert silero_chunks("OK. 100%. Да.") == ["OK. 100%. Да."]
    assert silero_chunks("Hello world. ...") == []  # Silero rejects text with nothing it can say
    assert silero_chunks("") == []


def test_model_is_downloaded_once_and_checked(tmp_path):
    source = tmp_path / "src.pt"
    source.write_bytes(b"model bytes")
    good = hashlib.sha256(b"model bytes").hexdigest()
    models = tmp_path / "models"
    path = fetch_silero(models, source.as_uri(), good)
    assert path == models / SILERO_FILE and path.read_bytes() == b"model bytes"
    source.write_bytes(b"changed upstream")
    assert fetch_silero(models, source.as_uri(), good).read_bytes() == b"model bytes"  # cached, not fetched again


def test_damaged_download_is_rejected(tmp_path):
    source = tmp_path / "src.pt"
    source.write_bytes(b"tampered")
    models = tmp_path / "models"
    (models / SILERO_FILE).parent.mkdir(parents=True)
    (models / SILERO_FILE).write_bytes(b"truncated")  # a damaged cache is fetched again
    with pytest.raises(RuntimeError, match="checksum"):
        fetch_silero(models, source.as_uri(), hashlib.sha256(b"original").hexdigest())
    assert not (models / SILERO_FILE).with_suffix(".part").exists()


def test_hand_set_stress_marks_reach_silero_and_other_plus_signs_do_not():
    assert silero_chunks("Компания МИК+О, з+амок на двери") == ["Компания МИК+О, з+амок на двери"]
    assert silero_chunks("Звоните +7, 2+2") == ["Звоните 7, 2 2"]


def test_stress_marks_are_removed_where_they_are_not_understood():
    from voice_agent_runtime.hotwords import normalize_terms
    from voice_agent_runtime.normalize import unstressed
    assert unstressed("«МИК+О», з+амок, +7 999, c++") == "«МИКО», замок, +7 999, c++"
    assert normalize_terms(["МИК+О"]) == ["МИКО"]
