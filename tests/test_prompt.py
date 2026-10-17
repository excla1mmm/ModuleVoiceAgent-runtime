import json

import pytest

from conftest import bundle_data
from voice_agent_runtime.bundle import BundleError, parse_bundle
from voice_agent_runtime.prompt import (MAX_TOKENS, RECOMMENDED_TOKENS, compact_numbers, estimate_tokens,
                                        prompt_tokens, static_prompt, system_prompt, tools)


def test_compact_numbers():
    assert compact_numbers("2 900 рублей и 12 900 ₽, 1 200 000") == "2900 рублей и 12900 ₽, 1200000"
    assert compact_numbers("с 9 до 18, 3 месяца") == "с 9 до 18, 3 месяца"


def test_prompt_contains_bundle_data_with_digits():
    bundle = parse_bundle("3", bundle_data())
    text = system_prompt(static_prompt(bundle), "79991234567")
    assert "«Телеком Сервис»" in text
    assert "2900 рублей" in text and "12900 рублей" in text and "2 900" not in text
    assert "- Иванова Светлана Петровна — главный бухгалтер — Бухгалтерия — 210 (также: Света)" in text
    assert "Волков Андрей Михайлович — генеральный директор — Руководство — 100 — напрямую не соединять, переводить на 101" in text
    assert "- Бухгалтерия (общая очередь) — 2100" in text
    assert "Дополнительные правила компании:" in text
    assert text.endswith("Номер, с которого звонит клиент: 79991234567.")
    assert "пиши цифрами" in text


def test_unknown_caller():
    bundle = parse_bundle("3", bundle_data())
    assert system_prompt(static_prompt(bundle), "").endswith("неизвестен.")


def test_tools_follow_callback_setting():
    names = [t["function"]["name"] for t in tools(parse_bundle("3", bundle_data()))]
    assert names == ["transfer_call", "request_callback", "end_call"]
    off = parse_bundle("3", bundle_data(callbacks_enabled=False))
    assert [t["function"]["name"] for t in tools(off)] == ["transfer_call", "end_call"]
    assert "Заявки на обратный звонок не принимаются" in static_prompt(off)


def test_size_estimate_and_limits():
    small = parse_bundle("3", bundle_data())
    assert prompt_tokens(small) < RECOMMENDED_TOKENS
    assert estimate_tokens("а" * 320) == 100
    # ~130 employees + ~90 entries was the measured safe size (~7k tokens)
    big = bundle_data(directory=[{"number": str(1000 + i), "name": f"Сотрудник Номер {i} Отчество",
                                  "position": "менеджер по продажам", "department": "Отдел продаж"} for i in range(130)],
                      knowledge=[{"topic": f"Вопрос {i}", "answer": "Ответ на вопрос клиента о услугах компании, "
                                  "ценах и сроках подключения, примерно в одно-два предложения."} for i in range(90)])
    tokens = prompt_tokens(parse_bundle("3", big))
    assert RECOMMENDED_TOKENS * 0.6 < tokens < MAX_TOKENS


def test_transfer_targets_and_route_to():
    bundle = parse_bundle("3", bundle_data())
    assert bundle.resolve_transfer("210") == "210"
    assert bundle.resolve_transfer("100") == "101"  # CEO -> assistant
    assert bundle.resolve_transfer("2100") == "2100"
    assert bundle.resolve_transfer("89991234567") is None  # never an arbitrary number from the model
    assert bundle.resolve_transfer("") is None


@pytest.mark.parametrize("change, message", [
    ({"schema": 2}, "schema"), ({"voice": "ru RU/irina"}, "voice"), ({"extension": "x"}, "extension"),
    ({"company_name": ""}, "company_name"), ({"directory": [{"number": "1a", "name": "x"}]}, "number"),
    ({"goodbye_grace_seconds": 99}, "goodbye_grace_seconds"), ({"knowledge": "x"}, "knowledge"),
])
def test_invalid_bundles(change, message):
    with pytest.raises(BundleError, match=message):
        parse_bundle("3", bundle_data(**change))


def test_bundle_round_trips_through_json():
    assert parse_bundle("3", json.loads(json.dumps(bundle_data()))).version == "v1"
