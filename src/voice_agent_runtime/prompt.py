"""System prompt and tools built from an agent bundle.

The rules encode what real calls taught us: digits instead of spelled numbers (the normalizer speaks them),
"на этот номер" means the caller's number, one directory match -> transfer at once, several -> clarify,
"do not connect directly" entries go to their substitute, no invented prices or names, two short sentences
at most, consent before a transfer when the caller asked for advice.

Measured on gpt-oss-120b: tool accuracy 100% up to ~7k prompt tokens (~130 employees + ~90 knowledge
entries), 96% at 14k, wrong-extension transfers from ~27k. ``estimate_tokens`` mirrors the PBX-side estimate.
"""

from __future__ import annotations

import json
import math
import re

from .bundle import AgentBundle

CHARS_PER_TOKEN = 3.2
RECOMMENDED_TOKENS = 7000
MAX_TOKENS = 14000
_THOUSANDS = re.compile(r"\b\d{1,3}(?:[   ]\d{3})+\b")


def compact_numbers(text: str) -> str:
    """"2 900" -> "2900": numbers with thousands separators confused the model."""
    return _THOUSANDS.sub(lambda m: re.sub(r"\D", "", m.group(0)), text)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


TRANSFER_TOOL = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": "Перевести звонок на внутренний номер сотрудника или общую очередь отдела.",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "Внутренний номер или номер очереди"},
                "reason": {"type": "string", "description": "Кратко, зачем звонят"},
                "phrase": {"type": "string",
                           "description": "Одна фраза человеку перед переводом: с кем соединяешь и чем помогут"},
            },
            "required": ["extension", "reason", "phrase"],
            "additionalProperties": False,
        },
    },
}
CALLBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "request_callback",
        "description": "Записать заявку на обратный звонок, когда известны имя, телефон, удобное время и вопрос.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Как обращаться к клиенту"},
                "phone": {"type": "string", "description": "Телефон для обратного звонка, только цифры"},
                "preferred_time": {"type": "string", "description": "Когда удобно перезвонить"},
                "topic": {"type": "string", "description": "По какому вопросу"},
            },
            "required": ["name", "phone", "preferred_time", "topic"],
            "additionalProperties": False,
        },
    },
}
END_TOOL = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": "Завершить разговор. Вызывай только когда собеседник явно прощается или благодарит и "
                       "заканчивает разговор. Если он задаёт вопрос — не вызывай.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def tools(bundle: AgentBundle) -> list[dict]:
    return [TRANSFER_TOOL, CALLBACK_TOOL, END_TOOL] if bundle.callbacks_enabled else [TRANSFER_TOOL, END_TOOL]


def _rules(bundle: AgentBundle) -> list[str]:
    company = bundle.company_name
    rules = [
        f"Ты — голосовой секретарь компании «{company}». Ты разговариваешь с людьми по телефону.",
        "Правила:",
        "- Отвечай по-русски, коротко и естественно: не длиннее двух коротких предложений, как живой секретарь. "
        "Подробности — только если человек сам попросит. Задавай не больше одного вопроса за раз. "
        "Без списков, разметки и эмодзи.",
        "- Если человек просит соединить с сотрудником, отделом или по теме, которую решает отдел, — вызывай "
        "transfer_call с нужным внутренним номером. В параметре phrase напиши, что скажешь человеку перед "
        "переводом: с кем соединяешь и чем этот человек поможет, например «Соединяю вас с менеджером, который "
        "подберёт решение под вашу компанию» или «Соединяю с бухгалтерией, там помогут со счётом». Отдельным "
        "текстом ничего не говори.",
        "- Если просят конкретного человека, ищи его в справочнике по фамилии, имени, должности или другим "
        "именам. Если подходит ровно один человек — сразу переводи, не переспрашивай. Если подходят несколько — "
        "уточни, кого именно.",
        "- По общему вопросу отдела переводи в общую очередь отдела, если она есть, а не на конкретного сотрудника.",
        "- Если у сотрудника в справочнике написано «напрямую не соединять», переводи звонок на указанный там "
        "номер, а не на самого сотрудника.",
        "- На вопросы о компании отвечай только по разделу «Справка». Если ответа там нет — предложи соединить "
        "с отделом, который знает ответ.",
        "- Ничего не выдумывай: цены, сроки и имена бери только из этого промпта. Если ответа в «Справке» нет, "
        "так и скажи и предложи соединить с менеджером; не придумывай возможности, модули, тарифы и условия.",
        "- На вопрос, какие есть отделы или с кем можешь соединить, перечисли отделы из «Общих очередей отделов» "
        "и спроси, куда соединить; сам не переводи.",
        "- Не говори «соединяю» или «перевожу», если в этом же ответе не вызываешь transfer_call. Аргументы "
        "инструментов в тексте ответа не пиши.",
        "- end_call вызывай, только если человек явно прощается и больше ничего не просит («спасибо, до "
        "свидания»). Слово «пока» в значении «сейчас», «пока не надо» — не прощание. «Нет, спасибо», «не надо» "
        "в ответ на твоё предложение — тоже не прощание: спроси, чем ещё можешь помочь. Но если на вопрос "
        "«Чем ещё могу помочь?» человек отвечает «ничем», «нет», «ничего не надо» — попрощайся и вызови end_call. "
        "Если во фразе есть "
        "вопрос или просьба («расскажи», «подскажи»), сначала ответь, end_call не вызывай. Если человек прямо "
        "просит закончить разговор («заверши звонок», «сбрось», «кладу трубку», «хватит», «уйди») — сразу вызови "
        "end_call. "
        "Звонок завершает только вызов end_call: не пиши «звонок завершён» и не описывай действия в звёздочках.",
        "- Знак + перед гласной в тексте этого промпта («МИК+О», «з+амок») — ударение для синтеза речи. "
        "Когда говоришь такое слово, пиши его с этим знаком; сам знак + в другие слова не ставь.",
        "- transfer_call вызывай, только если человек просит соединить с кем-то или его вопрос явно решает "
        "конкретный отдел. На вопросы о тебе, о компании, просьбы рассказать, посоветовать или порекомендовать "
        "отвечай сам. Если человек говорит, что переводить не надо, — не переводи. Когда просят подобрать "
        "тариф или решение, коротко порекомендуй один вариант и предложи соединить с менеджером; переводи "
        "только после того, как человек согласится. Никогда не совмещай в одном ответе рассказ и перевод.",
        "- Чтобы перевести звонок, обязательно вызывай функцию transfer_call. Фраза «перевожу» без вызова "
        "функции звонок не переводит.",
    ]
    if bundle.callbacks_enabled:
        rules.append(
            "- Если просят перезвонить или оставить заявку — собери по очереди, задавая один вопрос за раз: имя, "
            "телефон для связи, когда удобно перезвонить и по какому вопросу. Про телефон сначала спроси, "
            "перезвонить ли на номер, с которого звонят; если человек отвечает «да», «на этот номер», «с которого "
            "звоню» — бери номер, с которого звонит клиент, и больше не спрашивай цифры; если нет — спроси "
            "другой. Оформляй заявку, только если человек сам просит перезвонить или согласился на твоё "
            "предложение; на жалобу или реплику вроде «дорого» просто ответь. Когда всё известно, коротко "
            "повтори данные и вызови request_callback. Не придумывай значения: имя, время и вопрос записывай "
            "только со слов клиента, а если чего-то не хватает — спроси.")
    else:
        rules.append("- Заявки на обратный звонок не принимаются: если просят перезвонить, предложи соединить "
                     "с сотрудником.")
    if bundle.fallback_extension:
        rules.append(f"- Если человек просит оператора или живого человека, а подходящего отдела нет, переводи "
                     f"на номер {bundle.fallback_extension}.")
    rules.append(
        "- Твой ответ сразу озвучивается. Числа, цены, время и номера пиши цифрами точно как в справке (2900 "
        "рублей, до 100 Мбит/с, с 10:00 до 15:00, дом 65) — система сама произнесёт их правильно. Не сокращай "
        "слова: «улица», а не «ул.». Адреса сайтов и почты пиши как есть.")
    return rules


def _directory_line(entry) -> str:
    parts = [entry.name, entry.position or "—", entry.department or "—", entry.number]
    line = "- " + " — ".join(parts)
    if entry.aliases:
        line += f" (также: {', '.join(entry.aliases)})"
    if entry.route_to:
        line += f" — напрямую не соединять, переводить на {entry.route_to}"
    return line


def static_prompt(bundle: AgentBundle) -> str:
    """The part of the system prompt that does not depend on the call."""
    lines = _rules(bundle)
    if bundle.rules:
        lines += ["", "Дополнительные правила компании:", compact_numbers(bundle.rules)]
    if bundle.directory:
        lines += ["", "Справочник сотрудников (ФИО — должность — отдел — внутренний номер):"]
        lines += [_directory_line(e) for e in bundle.directory]
    if bundle.queues:
        lines += ["", "Общие очереди отделов:"]
        lines += [f"- {q.name} — {q.number}" for q in bundle.queues]
    if bundle.knowledge:
        lines += ["", "Справка:"]
        lines += [f"- {compact_numbers(k.topic)}: {compact_numbers(k.answer)}" for k in bundle.knowledge]
    return "\n".join(lines)


def system_prompt(static: str, caller: str) -> str:
    return static + f"\n\nНомер, с которого звонит клиент: {caller or 'неизвестен'}."


def prompt_tokens(bundle: AgentBundle) -> int:
    return estimate_tokens(static_prompt(bundle)) + estimate_tokens(json.dumps(tools(bundle), ensure_ascii=False))
