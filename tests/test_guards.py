import json
import pytest

from voice_agent_runtime.guards import (PHRASES, CallContext, ReplyFilter, callback_phone, decide, leaked_content,
                                        missing_callback_fields)


def stream(chunks):
    f = ReplyFilter()
    said = []
    for chunk in chunks:
        said += f.feed(chunk)
    said += f.finish()
    return said, f.reply


def test_clauses_are_spoken_as_they_arrive():
    f = ReplyFilter()
    assert f.feed("Офис на Профсоюзной, ") == ["Офис на Профсоюзной,"]
    assert f.feed("дом 65") == []
    assert f.finish() == ["дом 65"]
    assert f.reply == "Офис на Профсоюзной, дом 65"


def test_reasoning_leak_is_cut():
    said, reply = stream(["Соединяю с бухгалтерией. ", "(Waiting for user)(We need to ..."])
    assert said == ["Соединяю с бухгалтерией."]
    assert "Waiting" not in reply


def test_leak_inside_a_clause():
    said, _ = stream(["Хорошо(We should transfer) дальше"])
    assert said == ["Хорошо"]


def test_one_question_per_reply():
    said, reply = stream(["Как к вам обращаться? ", "И когда вам удобно? ", "Ещё текст."])
    assert said == ["Как к вам обращаться?"]
    assert reply == "Как к вам обращаться?"


def test_glued_second_message_is_cut():
    said, reply = stream(["Записал вашу заявку.Буду рад помочь ещё"])
    assert said == ["Записал вашу заявку."]
    assert reply == "Записал вашу заявку."


def test_leaked_json_message_keeps_only_content():
    said, reply = stream(['{"role":"assistant",', '"content":"Мы работаем с 9:00 до 19:00."}'])
    assert said == ["Мы работаем с 9:00 до 19:00."]
    assert leaked_content("not json") == ""


def ctx(**kw):
    return CallContext(caller=kw.pop("caller", "79991234567"), callbacks_enabled=kw.pop("callbacks", True), **kw)


def test_end_call_without_farewell_keeps_the_line():
    d = decide("end_call", "{}", "Нет, спасибо", "", ctx())
    assert d.tool is None and d.phrase == PHRASES["anything_else"]


def test_end_call_with_farewell():
    d = decide("end_call", "{}", "Спасибо, до свидания", "", ctx())
    assert d.tool == "end_call" and d.phrase == PHRASES["goodbye"]


def test_poka_at_the_end_is_a_farewell_but_not_in_the_middle():
    assert decide("end_call", "{}", "Ну всё, пока", "", ctx()).tool == "end_call"
    assert decide("end_call", "{}", "Пока не надо, расскажите о тарифах", "", ctx()).tool is None


@pytest.mark.parametrize("text", [
    "Я бы хотел уже тебя сбросить, если честно.", "Заверши звонок тоже.", "Я тебя прошу, заверши звонок.",
    "Давайте закончим разговор", "Всё, кладу трубку", "Положите трубку", "Хватит", "Ну до встречи",
])
def test_an_explicit_request_to_end_the_call_ends_it(text):
    d = decide("end_call", "{}", text, "", ctx())
    assert d.tool == "end_call" and d.phrase == PHRASES["goodbye"]


@pytest.mark.parametrize("text", ["Нет.", "Ничего не надо.", "Кажется, пока что ничем.", "Хватит ли этого тарифа?",
                                  "Как завершить оформление заказа?"])
def test_a_refusal_or_question_is_not_a_request_to_end(text):
    assert decide("end_call", "{}", text, "", ctx()).tool is None


def test_stage_directions_are_not_said():
    said, reply = stream(["*Звонок ", "завершён.* ", "Хорошего дня!"])
    assert said == ["Хорошего дня!"] and reply == "Хорошего дня!"
    said, _ = stream(["Тариф стоит **2900 рублей**, ", "это *(улыбается)* недорого."])
    assert said == ["Тариф стоит 2900 рублей,", "это недорого."]


def test_a_reply_of_only_narration_asks_what_else_is_needed():
    said, reply = stream(["*Звонок завершён.*"])
    assert said == [] and reply == ""
    d = decide(None, "", "Зачем?", reply, ctx())
    assert d.tool is None and d.phrase == PHRASES["anything_else"]


def test_transfer_with_answer_and_no_request_becomes_an_offer():
    d = decide("transfer_call", '{"extension":"2200","phrase":"Соединяю"}', "Какой тариф мне подойдёт?",
               "Для офиса подойдёт «Стандарт».", ctx())
    assert d.tool is None and d.phrase == PHRASES["offer_transfer"]


def test_needed_is_not_an_explicit_request():
    d = decide("transfer_call", '{"extension":"2200"}', "Мне нужен тариф подешевле", "Есть «Старт».", ctx())
    assert d.tool is None


def test_explicit_transfer_uses_the_model_phrase():
    d = decide("transfer_call", '{"extension":"2100","reason":"счёт","phrase":"Соединяю с бухгалтерией."}',
               "Соедините с бухгалтерией", "", ctx())
    assert d.tool == "transfer_call" and d.phrase == "Соединяю с бухгалтерией." and not d.phrase_cached


def test_transfer_without_phrase_uses_the_cached_phrase():
    d = decide("transfer_call", '{"extension":"2100"}', "Позовите оператора", "", ctx())
    assert d.phrase == PHRASES["transfer"] and d.phrase_cached


def test_callback_without_request_becomes_an_offer_then_is_accepted():
    c = ctx()
    c.caller_said = ("Меня зовут Иван", "Завтра я свободен")
    args = '{"name":"Иван","phone":"79991234567","preferred_time":"завтра","topic":"тарифы"}'
    d = decide("request_callback", args, "Дорого у вас", "", c)
    assert d.tool is None and d.phrase == PHRASES["offer_callback"] and c.callback_wanted
    d = decide("request_callback", args, "Да", "", c)
    assert d.callback == {"name": "Иван", "phone": "79991234567", "preferred_time": "завтра", "topic": "тарифы"}


def test_incomplete_callback_asks_the_first_missing_field():
    d = decide("request_callback", '{"name":"Иван","phone":"","preferred_time":"","topic":"перезвонить клиенту"}',
               "Перезвоните мне, я Иван", "", ctx(caller=""))
    assert d.callback is None and d.phrase == PHRASES["ask_phone"]
    assert "phone" in d.tool_result and "topic" in d.tool_result


def test_this_number_means_the_caller_number():
    assert callback_phone("на этот номер", "79991234567") == "79991234567"
    assert callback_phone("+7 999 123-45-67", "1") == "+79991234567"
    assert callback_phone("12", "79991234567") == ""


def test_topic_with_perezvon_is_missing():
    assert missing_callback_fields({"name": "a", "phone": "1", "preferred_time": "b", "topic": "Перезвонить"}) == ["topic"]


def test_callbacks_disabled():
    d = decide("request_callback", "{}", "Перезвоните мне", "", ctx(callbacks=False))
    assert d.tool is None and d.callback is None


def test_an_english_note_is_not_said_but_a_brand_name_is():
    said, _ = stream(["Proceed with request_callback using phone 333. ", "Перезвонить вам на этот номер?"])
    assert said == ["Перезвонить вам на этот номер?"]
    said, _ = stream(["Подключим Zoom и Microsoft Teams за день."])
    assert said == ["Подключим Zoom и Microsoft Teams за день."]


def test_a_short_internal_caller_number_is_a_valid_callback_phone():
    assert callback_phone("333", "333") == "333"
    assert callback_phone("", "333") == "333"
    assert callback_phone("444", "333") == ""


@pytest.mark.parametrize("text", ["А какие отделы есть?", "С кем ты меня можешь соединить?", "Куда вы меня переводите?"])
def test_a_question_about_where_to_be_connected_is_answered_with_the_departments(text):
    d = decide("transfer_call", '{"extension": "1", "phrase": "Соединяю вас с отделом телефонии"}', text, "", ctx())
    assert d.tool is None and d.list_options


@pytest.mark.parametrize("text", ["Соедините с кем-нибудь", "Можете соединить с бухгалтерией?", "Да",
                                  "Мне с директором надо связаться"])
def test_a_request_to_be_connected_still_transfers(text):
    assert decide("transfer_call", '{"extension": "1"}', text, "", ctx()).tool == "transfer_call"


def test_no_second_offer_when_the_answer_already_asks():
    d = decide("transfer_call", '{"extension": "2588"}', "А есть какие-то модули?",
               "Есть платные модули. Хотите связаться с менеджером?", ctx())
    assert d.tool is None and d.phrase == ""
    d = decide("transfer_call", '{"extension": "2588"}', "Расскажи про модули", "Есть платные модули.", ctx())
    assert d.tool is None and d.phrase == PHRASES["offer_transfer"]


def test_departments_are_named_once_without_the_line_kind():
    from voice_agent_runtime.guards import options_phrase
    names = ["Телефония (внутренние)", "Телефония (мобильные)", "СБИС (внутренние)", "Отдел продаж"]
    assert options_phrase(names) == "Могу соединить с отделами: Телефония, СБИС, Отдел продаж. Куда вас соединить?"
    assert options_phrase([]) == PHRASES["offer_transfer"]


def test_tool_arguments_printed_as_text_are_not_said():
    said, reply = stream(["Перевожу к менеджеру отдела продаж.", '{"phrase":"Соединяю","extension":"2588"}'])
    assert said == ["Перевожу к менеджеру отдела продаж."] and "extension" not in reply


@pytest.mark.parametrize("text", ["Хватит уже.", "Всё, уйди.", "Да всё уже, уйди", "Отстань", "Всё, спасибо"])
def test_more_ways_to_ask_to_end_the_call(text):
    assert decide("end_call", "{}", text, "", ctx()).tool == "end_call"


@pytest.mark.parametrize("text", ["Ничем.", "Да ничем, я тебе сказал уже пять раз.", "Нет.", "Ничего не надо",
                                  # call 20
                                  "Да всё, наверное, больше ничем.", "Больше ничем.", "Да всё.", "Всё.",
                                  "Всё, больше ничего.", "Наверное, ничего.", "Спасибо, больше ничего не нужно.",
                                  "Пока что ничем."])
def test_nothing_more_after_anything_else_ends_the_call(text):
    asked = ctx()
    asked.asked_anything_else = True
    assert decide("end_call", "{}", text, "", asked).tool == "end_call"
    assert decide("end_call", "{}", text, "", ctx()).tool is None  # after an offer it is not the end


@pytest.mark.parametrize("text", ["Всё так.", "Да всё понятно, но есть вопрос", "Больше вопросов нет?",
                                  "Пока не знаю."])
def test_other_answers_to_anything_else_do_not_end_the_call(text):
    assert decide("end_call", "{}", text, "", ctx(asked_anything_else=True)).tool is None


def test_anything_else_questions_are_recognized():
    from voice_agent_runtime.guards import ANYTHING_ELSE
    for said in ("Хорошо. Чем ещё могу помочь?", "Чем я могу вам помочь?", "Нужно что-то ещё?",
                 "Могу ли я ещё чем-нибудь помочь?", "Чем ещё могу быть полезна?"):
        assert ANYTHING_ELSE.search(said), said
    assert not ANYTHING_ELSE.search("Соединить вас с менеджером, который подберёт решение под вас?")


def test_a_made_up_name_or_time_is_asked_for():
    c = ctx(caller="333")
    c.callback_wanted = True
    args = '{"name":"Клиент","phone":"333","preferred_time":"сейчас","topic":"подсистема телефонии"}'
    d = decide("request_callback", args, "Создай мне заявочку на обратный звонок.", "", c)
    assert d.callback is None and d.phrase == PHRASES["ask_name_and_time"]  # both missing: one question
    c.caller_said = ("Создай мне заявочку на обратный звонок.", "Меня зовут Анатолий.")
    args = '{"name":"Анатолий","phone":"333","preferred_time":"сейчас","topic":"подсистема телефонии"}'
    d = decide("request_callback", args, "Подожди", "", c)
    assert d.callback is None and d.phrase == PHRASES["ask_preferred_time"]
    d = decide("request_callback", args, "Давай в 6 вечера", "", c)
    assert d.callback["name"] == "Анатолий" and d.phrase.startswith("Записываю заявку: Анатолий")


def test_the_request_itself_is_not_a_topic():
    from voice_agent_runtime.guards import missing_callback_fields
    assert "topic" in missing_callback_fields({"name": "А", "phone": "1", "preferred_time": "1", "topic": "Обратный звонок"})


def test_no_second_confirmation_when_the_model_confirmed():
    c = ctx()
    c.callback_wanted = True
    c.caller_said = ("Я Анатолий, перезвоните 15 сентября",)
    args = '{"name":"Анатолий","phone":"9774276045","preferred_time":"15.09.2026","topic":"телефония"}'
    d = decide("request_callback", args, "Перезвони на 977 427 60 45", "Записал новую заявку с указанным номером.", c)
    assert d.callback is not None and d.phrase == ""


def test_special_tokens_and_filler_are_not_said():
    said, _ = stream(["Анатолий, в какое время вам удобно перезвонить?(ожидаем).......<|start|>assistant<|channel|>final"])
    assert said == ["Анатолий,", "в какое время вам удобно перезвонить?"]
    said, _ = stream(["Секунду", "................", "<|start|>assistant"])
    assert said == ["Секунду"]


def callback_ctx(*said, caller="333"):
    c = ctx(caller=caller)
    c.callback_wanted = True
    c.caller_said = said
    return c


def test_no_request_is_saved_while_the_agent_is_still_asking():
    c = callback_ctx("Я Арбуз", "перезвоните завтра в 2 часа дня")
    args = '{"name":"Арбуз","phone":"333","preferred_time":"завтра в 14:00","topic":"модуль телефонии"}'
    d = decide("request_callback", args, "Да", "Удобно перезвонить завтра в 14:00?", c)
    assert d.callback is None and d.phrase == "" and "дождись ответа" in d.tool_result


def test_phone_digits_are_not_a_time():
    c = callback_ctx("Меня зовут Арбуз", "Нет, перезвони на номер 9 775 392 14 05")
    args = '{"name":"Арбуз","phone":"9775392 1405","preferred_time":"сейчас","topic":"модуль телефонии"}'
    d = decide("request_callback", args, "Нет, перезвони на номер 9 775 392 14 05", "", c)
    assert d.callback is None and d.phrase == PHRASES["ask_preferred_time"]


def test_an_incomplete_number_is_asked_in_full():
    c = callback_ctx("Я Егор", "перезвоните завтра в 14:00")
    args = '{"name":"Егор","phone":"897742765","preferred_time":"завтра в 14:00","topic":"лицензия"}'
    d = decide("request_callback", args, "Мой номер 897742765", "", c)
    assert d.callback is None and d.phrase == PHRASES["ask_phone_full"]
    args = '{"name":"Егор","phone":"89774276512","preferred_time":"завтра в 14:00","topic":"лицензия"}'
    d = decide("request_callback", args, "8 977 427 65 12", "", c)
    assert d.callback["phone"] == "89774276512"
    assert d.phrase == ("Записываю заявку: Егор, перезвоним завтра в 14:00 на номер 89774276512. "
                        "Если что-то не так, поправьте. Чем ещё могу помочь?")


def test_the_callers_own_number_is_read_back_as_such():
    c = callback_ctx("Я Егор", "перезвоните в 6 вечера")
    args = '{"name":"Егор","phone":"","preferred_time":"18:00","topic":"лицензия"}'
    d = decide("request_callback", args, "на этот номер", "", c)
    assert "на номер, с которого вы звоните" in d.phrase and d.callback["phone"] == "333"


def test_anything_else_counts_only_as_the_last_question():
    from voice_agent_runtime.guards import ANYTHING_ELSE, last_question
    reply = "Чем ещё могу помочь? Удобно ли вам, чтобы специалист позвонил сейчас?"
    assert not ANYTHING_ELSE.search(last_question(reply))
    assert ANYTHING_ELSE.search(last_question("Записываю заявку: Егор. Чем ещё могу помочь?"))
    assert last_question("Хорошо.") == ""


def test_invisible_characters_are_not_said():
    _, reply = stream(["Номер, который вы указали ранее: 897\u200b\u200b\u200b-742-765."])
    assert reply == "Номер, который вы указали ранее: 897-742-765."


def test_a_farewell_after_anything_else_ends_the_call_once():
    """Call 17: "...Чем ещё могу помочь?" - "Всё так." - "Спасибо, всего доброго!" + end_call."""
    d = decide("end_call", "{}", "Всё так.", "Спасибо, всего доброго!", ctx(asked_anything_else=True))
    assert d.tool == "end_call" and d.phrase == ""
    assert decide("end_call", "{}", "Всё так.", "Спасибо, всего доброго!", ctx()).tool is None
    assert decide("end_call", "{}", "А когда перезвоните?", "Всего доброго!", ctx(asked_anything_else=True)).tool is None
    assert decide("end_call", "{}", "Всё так.", "Хорошо.", ctx(asked_anything_else=True)).tool is None


def test_a_callback_asked_for_earlier_is_not_offered_again():
    """Call 17: "давай заявочку оставим на обратное" - "Как к вам обращаться?" - "Меня Егор зовут."."""
    args = json.dumps({"name": "Егор", "phone": "", "preferred_time": "", "topic": "Подбор тарифа"})
    c = ctx(caller_said=("Ну, давай заявочку оставим на обратное. Звонок на",))
    assert decide("request_callback", args, "Меня Егор зовут.", "", c).phrase != PHRASES["offer_callback"]


def test_a_placeholder_time_is_asked():
    """Call 17: preferred_time "не уточнено" passed because the caller had said "сейчас" about something else."""
    args = json.dumps({"name": "Егор", "phone": "89774276045", "preferred_time": "не уточнено", "topic": "Тариф"})
    c = ctx(callback_wanted=True, caller_said=("Меня Егор зовут.", "А сейчас рабочее время?"))
    d = decide("request_callback", args, "8977 427 6045", "", c)
    assert d.callback is None and d.phrase == PHRASES["ask_preferred_time"]


def test_a_topic_that_names_something_is_kept():
    """Call 18: "заявка на обратный звонок с отделом продаж" was taken for an empty topic."""
    base = {"name": "Толик", "phone": "333", "preferred_time": "завтра"}
    assert missing_callback_fields({**base, "topic": "заявка на обратный звонок с отделом продаж"}) == []
    assert missing_callback_fields({**base, "topic": "перезвонить клиенту"}) == ["topic"]
    assert missing_callback_fields({**base, "topic": "обратный звонок"}) == ["topic"]


def test_no_transfer_without_a_request_while_a_callback_is_arranged():
    """Call 18: "По какому вопросу вам перезвонить?" - "Хочу поговорить с отделом продаж." - transferred."""
    args = json.dumps({"extension": "2588", "phrase": "Соединяю вас с отделом продаж"})
    d = decide("transfer_call", args, "Хочу поговорить с отделом продаж.", "", ctx(callback_wanted=True))
    assert d.tool is None and d.phrase == PHRASES["offer_transfer"]
    assert decide("transfer_call", args, "Соедините с отделом продаж", "", ctx(callback_wanted=True)).tool == "transfer_call"


def test_a_stutter_is_not_said():
    """Call 19: "Внутерьний внутренний внутренний... Внутренний номер Дмитрия — 222."."""
    f = ReplyFilter()
    said = f.feed("Внутерьний внутренний внутренний... Внутренний номер Дмитрия — 222. ") + f.finish()
    assert said == ["Внутренний номер Дмитрия —", "222."]
    f = ReplyFilter()
    assert f.feed("Да, да, да, конечно. ") + f.finish() == ["Да,", "да,", "да,", "конечно."]
