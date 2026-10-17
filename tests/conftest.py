import pytest


def bundle_data(**overrides):
    data = {
        "schema": 1, "version": "v1", "name": "Секретарь", "extension": "7777",
        "company_name": "Телеком Сервис", "greeting": "Здравствуйте! Компания «Телеком Сервис», чем могу помочь?",
        "voice": "ru_RU-dmitri-medium", "rules": "Тариф «Бизнес» стоит 12 900 рублей.",
        "fallback_extension": "100", "callbacks_enabled": True,
        "transfer_grace_seconds": 2, "goodbye_grace_seconds": 3,
        "directory": [
            {"number": "210", "name": "Иванова Светлана Петровна", "position": "главный бухгалтер",
             "department": "Бухгалтерия", "aliases": ["Света"]},
            {"number": "100", "name": "Волков Андрей Михайлович", "position": "генеральный директор",
             "department": "Руководство", "route_to": "101"},
            {"number": "101", "name": "Алексеева Наталья Викторовна", "position": "помощник директора",
             "department": "Руководство"},
        ],
        "queues": [{"number": "2100", "name": "Бухгалтерия (общая очередь)"}],
        "knowledge": [{"topic": "Тарифы", "answer": "«Старт» — 2 900 рублей в месяц, «Бизнес» — 12 900 рублей."}],
    }
    data.update(overrides)
    return data


@pytest.fixture
def bundle_json():
    return bundle_data()
