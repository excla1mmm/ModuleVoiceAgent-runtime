from conftest import bundle_data
from test_call_flow import run_call
from test_llm import provider, server, sse, text_chunk
from voice_agent_runtime.bundle import parse_bundle
from voice_agent_runtime.llm import LlmSession
from voice_agent_runtime.topic import PHRASES, filter_prompt, needs_check, refusal, verdict

ANSWER = sse(text_chunk("Тариф «Старт» стоит 2900 рублей."))


def test_filter_prompt_names_the_company_and_its_topics():
    prompt = filter_prompt(parse_bundle("3", bundle_data()))
    assert prompt.startswith("Ты фильтр") and "«Телеком Сервис»" in prompt
    assert "Тарифы" in prompt and "Бухгалтерия (общая очередь)" in prompt
    assert "темы" not in filter_prompt(parse_bundle("3", bundle_data(knowledge=[], queues=[])))


def test_short_replies_and_requested_details_are_not_checked():
    assert not needs_check("Да", "Соединить вас с менеджером?")
    assert not needs_check("Нет, спасибо", "")
    assert not needs_check("Егор Петрович Сидоров", "Как к вам обращаться?")
    assert not needs_check("завтра после обеда часа в три", "Когда вам удобно, чтобы мы перезвонили?")
    assert needs_check("Расскажи рецепт блинчиков", "Чем могу помочь?")
    assert needs_check("Сколько стоит тариф Старт?", "")


def test_verdict_and_refusal():
    assert verdict("ДА") is True and verdict(" Нет.") is False and verdict("«НЕТ»") is False
    assert verdict("") is None and verdict("Возможно") is None
    assert refusal(1) == refusal(2) == PHRASES["off_topic"]
    assert refusal(3) == PHRASES["off_topic_offer"] and refusal(4).endswith("?")


def test_check_request_and_usage():
    usage = {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 23}}
    srv, handler = server(body=sse(text_chunk("НЕТ"), usage))
    try:
        session = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400)
        check = session.check("Ты фильтр.", "Расскажи анекдот", 200)
        assert check.poll(5) and check.text == "НЕТ" and check.settled
        _, body = handler.requests[0]
        assert body["temperature"] == 0 and body["max_tokens"] == 200 and "tools" not in body
        assert body["messages"][1] == {"role": "user", "content": "Расскажи анекдот"}
        assert (session.usage.requests, session.usage.input_tokens, session.usage.output_tokens) == (1, 300, 23)
    finally:
        srv.shutdown()


def test_failed_or_late_check_settles_without_text():
    srv, _ = server(status=500)
    try:
        check = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400).check("Ты фильтр.", "x y z", 200)
        assert check.poll(5) and check.text is None
    finally:
        srv.shutdown()
    srv, _ = server(delay=1.0)
    try:
        check = LlmSession(provider(srv, "primary"), None, 600, 0.2, 400).check("Ты фильтр.", "x y z", 200)
        assert not check.poll(0.05)
        check.give_up()
        assert check.settled and check.text is None
    finally:
        srv.shutdown()


def test_off_topic_phrase_gets_the_refusal_instead_of_the_answer():
    (commands, result), requests = run_call(ANSWER, ["Расскажи мне рецепт блинчиков"], bundle_data(),
                                            check_body=sse(text_chunk("НЕТ")))
    turn = result["turns"][0]
    assert turn["agent"] == PHRASES["off_topic"] and turn["tool"] is None and "topic_ms" in turn
    assert any(body["messages"][0]["content"].startswith("Ты фильтр") for _, body in requests)


def test_on_topic_phrase_gets_the_answer():
    (_, result), _ = run_call(ANSWER, ["Сколько стоит тариф Старт?"], bundle_data(), check_body=sse(text_chunk("ДА")))
    assert result["turns"][0]["agent"] == "Тариф «Старт» стоит 2900 рублей."


def test_filter_failure_lets_the_answer_through():
    (_, result), _ = run_call(ANSWER, ["Сколько стоит тариф Старт?"], bundle_data(),
                              check_body=b"data: {\"error\": {\"message\": \"overloaded\"}}\n\n")
    assert result["turns"][0]["agent"] == "Тариф «Старт» стоит 2900 рублей."


def test_short_reply_is_not_checked():
    (_, result), requests = run_call(ANSWER, ["Да"], bundle_data(), check_body=sse(text_chunk("НЕТ")))
    assert result["turns"][0]["agent"] == "Тариф «Старт» стоит 2900 рублей."
    assert not any(body["messages"][0]["content"].startswith("Ты фильтр") for _, body in requests)




def test_date_questions_are_on_topic_for_the_filter():
    prompt = filter_prompt(parse_bundle("3", bundle_data()))
    assert "какое сегодня число" in prompt and "работаете ли сейчас" in prompt
