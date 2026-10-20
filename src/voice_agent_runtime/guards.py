"""Guards on model output. Every rule here was needed in a real call.

* gpt-oss sometimes prints a whole message as JSON (``{"role":"assistant","content":...}``): only the
  content is said.
* it sometimes appends reasoning notes such as ``(Waiting for user)(We ...``: everything from the note on
  is dropped; a clause that is an English note ("Proceed with request_callback using phone 333.") is not said.
* one question per reply, and a second message glued to the first without a space (``...звонка.Буду``)
  is cut.
* ``end_call`` without a farewell, an explicit request to end the call ("заверши звонок", "сбрось", "хватит
  уже", "уйди") or "ничем" / "нет" in answer to "Чем ещё могу помочь?" becomes "Хорошо. Чем ещё могу помочь?"
  ("нет, спасибо" answering an offer is not the end of the conversation). After "Чем ещё могу помочь?" a reply
  without a question that the model answered with a farewell ("Всё так." -> "Спасибо, всего доброго!") ends it:
  the farewell has already been said.
* it sometimes narrates an action instead of calling the tool (``*Звонок завершён.*``): text between single
  asterisks is not said; ``**bold**`` markers are dropped and the words kept. A reply left empty asks what else
  the caller needs.
* a transfer that comes together with an answer, when the caller did not ask to be connected, becomes an
  offer (not a second one if the answer already asks); the transfer happens after "да". A transfer on a question
  ("А какие отделы есть?", "С кем можешь соединить?") does not happen either: the agent names the departments.
* tool arguments printed as text (``...{"phrase": ..., "extension": "2588"}``) are not said.
* ``request_callback`` only after the caller asked for it or accepted an offer; incomplete requests are not
  saved, the first missing field is asked by us. A name the caller never said ("Клиент") or a time when the
  caller named none ("сейчас" made up) or a placeholder time ("не уточнено") counts as missing; a callback the
  caller asked for earlier in the call ("давай заявочку оставим") is not offered again; no second confirmation
  when the model confirmed. A topic made only of callback words ("перезвонить клиенту") is asked for, one that
  names something ("заявка на обратный звонок с отделом продаж") is not (call 18). While a callback is being
  arranged, a transfer the caller did not ask for in so many words becomes an offer.
* a stutter of the model ("Внутерьний внутренний внутренний... Внутренний номер") is not said: a clause
  with one word three times in a row.
* a transfer turned into an offer names whom the model wanted to transfer to (call 19: "есть у вас Егор?"
  was answered with the generic "Соединить вас с менеджером...?").
* special tokens and filler of the model (``<|start|>assistant``, ``(ожидаем)......``) are not said. One callback per call: later calls update it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CLAUSE_END = re.compile(r"(?<=[.!?,:;—])\s+")
# a note "(We ...", tool arguments as JSON, a special token "<|start|>" or a run of dots
REASONING_LEAK = re.compile(r"\(\s*[A-Za-z]|\{\s*\"|<\||\.{4,}")
GLUED = re.compile(r"[.!?](?=[А-ЯЁA-Z(<])")  # "...звонка?Какой" / "...перезвонить?(ожидаем)"
INVISIBLE = re.compile(r"[\u200b-\u200f\u2060-\u2064\ufeff\u00ad]")  # zero-width and soft hyphens: noise for TTS
GLUED_REMARK = re.compile(r"(?<=[.!?])\([^)\d]*\)?\s*$")  # "...перезвонить?(ожидаем)": a remark, not speech
LATIN_WORD = re.compile(r"[A-Za-z_]{2,}")
CYRILLIC_LETTER = re.compile(r"[А-Яа-яЁё]")
FAREWELL = re.compile(r"до свидан|всего добр|всего хорош|хорошего дня|до связи|до встречи|прощай|\bпока\W*$", re.I)
END_REQUEST = re.compile(
    r"заверш\w*\s+(?:этот\s+|наш\s+)?(?:звон|разговор)|заверши(?:те)?\b|завершай(?:те)?\b"
    r"|законч\w*\s+(?:этот\s+|наш\s+)?(?:звон|разговор)|давай(?:те)?\s+законч"
    r"|сбрось|сбросьте|сбросить|сброшу|сбрасыва|полож\w*\s+трубк|клад\w*\s+трубк"
    r"|отключа(?:юсь|йся|йтесь)|\bхватит(?=\W*(?:уже\b|$|[,.!]))|\b(?:уйди|уходи|отстань|отвали)\b"
    r"|^\W*вс[её]\W+(?:спасибо|пока|хватит|уйди)", re.I)
# "ничем" / "нет" / "ничего не надо" answering "Чем ещё могу помочь?" ends the conversation; answering an offer
# ("Соединить вас?") it does not. Call 20: "Да всё, наверное, больше ничем." and "Больше ничем." - filler words
# may come first, and "всё" on its own ("Да всё.") is the same answer; "Всё так." is not.
_REFUSAL_LEAD = r"(?:да|нет|ну|а|так|вс[её]|наверное|пожалуй|кажется|вроде|пока(?:\s+что)?|уже|больше|спасибо" \
                r"|в\s+общем|в\s+принципе)"
REFUSAL = re.compile(rf"^\W*(?:{_REFUSAL_LEAD}\W+)*(?:(?:ничем|ничего|не\s+надо|не\s+нужно|нет)\b|вс[её]\W*$)", re.I)
ANYTHING_ELSE = re.compile(r"чем (?:(?:ещ[её]|я)\s+)*могу\s+(?:вам\s+)?(?:помочь|быть полезн)|что[- ]?(?:то|нибудь) ещ[её]"
                           r"|ещ[её] (?:вопрос|чем)|могу ли (?:я )?(?:ещ[её]|чем)", re.I)
OPTIONS_QUESTION = re.compile(r"\bс кем(?!-)\b|\bкуда\b|\bкак(?:ие|ой|ая|их)\b|\bчто за\b", re.I)
EXPLICIT_TRANSFER = re.compile(r"соедин|переключ|перевед|позов|оператор|живым|челове|с кем-нибудь|свяжите", re.I)
CALLBACK_INTENT = re.compile(r"перезвон|позвоните|свяжитесь|связаться|заявк|обратн", re.I)

PHRASES = {
    "transfer": "Соединяю, оставайтесь на линии.",
    "goodbye": "Всего доброго, до свидания!",
    "offer_transfer": "Соединить вас с менеджером, который подберёт решение под вас?",
    "anything_else": "Хорошо. Чем ещё могу помочь?",
    "offer_callback": "Могу оформить заявку: менеджер перезвонит и подберёт выгодный вариант. Оформить?",
    "unavailable": "Извините, сервис временно недоступен. Соединяю вас с сотрудником.",
    "transfer_denied": "К сожалению, не могу соединить с этим номером. Чем ещё могу помочь?",
    "options": "Могу соединить с отделами: {names}. Куда вас соединить?",
    "ask_name": "Как к вам обращаться?",
    "ask_phone": "Перезвонить вам на номер, с которого вы звоните?",
    "ask_preferred_time": "Когда вам удобно, чтобы мы перезвонили?",
    "ask_name_and_time": "Как к вам обращаться и когда вам удобно, чтобы мы перезвонили?",
    "ask_phone_full": "Продиктуйте, пожалуйста, номер полностью, вместе с кодом.",
    "ask_topic": "По какому вопросу вам перезвонить?",
    "offer_named": "Могу соединить: {name}. Соединить?",
}
CALLBACK_FIELDS = ("name", "phone", "preferred_time", "topic")
# words of a callback request itself: a topic made only of them says nothing about the caller's question
CALLBACK_WORD = re.compile(r"перезвон|позвон|обратн|звон|заявк|клиент|абонент|запрос|вопрос|связ|просьб|оставит|"
                           r"оформ|нужн|хоч", re.I)
PLACEHOLDER_TIME = re.compile(r"не\s*(?:уточн|указ|известн|определ|назван)|неизвестн|уточнит|\?", re.I)
PLACEHOLDER_NAMES = {"клиент", "абонент", "звонящий", "гость", "пользователь", "неизвестно", "не указано", "аноним"}
# a time the caller named: words, or numbers that look like a time or a date (not the digits of a phone)
TIME_SAID = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\bв\s+\d{1,2}\b|\b\d{1,2}\s*(?:час|ч\b)|\b\d{1,2}[./]\d{1,2}\b"
                       r"|\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)"
                       r"|утр|вечер|д[её]н|ночь|сейчас|сегодня|завтра|послезавтра|понедельн|вторн|сред[уаы]|четверг"
                       r"|пятниц|суббот|воскресен|час|минут|обед|любое|когда угодно|неделе|недел|январ|феврал|март"
                       r"|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр", re.I)
TOOL_RESULTS = {"request_callback": "Заявка на обратный звонок сохранена.",
                "transfer_call": "Звонок переведён.", "end_call": "Разговор завершён."}


def leaked_content(raw: str) -> str:
    """The content of a message the model printed as JSON instead of plain text."""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return ""
    return data["content"] if isinstance(data, dict) and isinstance(data.get("content"), str) else ""


class ReplyFilter:
    """Turns streamed text deltas into clauses to speak, applying the leak and one-message guards.

    ``said`` holds exactly what was emitted for speech, which is what goes into the conversation history.
    """

    def __init__(self) -> None:
        self.raw = ""
        self.pending = ""
        self.closed = False  # nothing more is said: a question was asked or a leak was found
        self.said: list[str] = []
        self.staged = False  # inside a *stage direction*

    @property
    def reply(self) -> str:
        return " ".join(self.said).strip()

    def _unstaged(self, clause: str) -> str:
        """The clause without *stage directions*; one may span several clauses."""
        kept = []
        for i, part in enumerate(clause.replace("**", "").split("*")):
            if i:
                self.staged = not self.staged
            if not self.staged:
                kept.append(part)
        return "".join(kept)

    def _emit(self, clauses: list[str]) -> list[str]:
        out = [c for c in (GLUED_REMARK.sub("", re.sub(r"\s+", " ", INVISIBLE.sub("", self._unstaged(c)))).strip()
                           for c in clauses)
               if c and not english_note(c) and not stutter(c)]
        self.said += out
        return out

    def feed(self, delta: str) -> list[str]:
        self.raw += delta
        if self.raw.lstrip().startswith("{") or self.closed:
            return []  # leaked message JSON is parsed at the end
        self.pending += delta
        leak = REASONING_LEAK.search(self.pending)
        if leak:  # say what came before the leaked note and stop
            head, self.pending, self.closed = self.pending[:leak.start()], "", True
            return self._emit(CLAUSE_END.split(head))
        parts = CLAUSE_END.split(self.pending)
        out: list[str] = []
        for clause in parts[:-1]:
            out += self._emit([clause])
            if clause.rstrip().endswith("?"):
                self.closed, self.pending = True, ""
                return out
        self.pending = parts[-1]
        glued = GLUED.search(self.pending)
        if glued:  # "...звонка?Какой..." / "...звонка.Буду..": keep only the first message
            first, self.pending, self.closed = self.pending[:glued.end()], "", True
            out += self._emit(CLAUSE_END.split(first))
        return out

    def finish(self) -> list[str]:
        if self.raw.lstrip().startswith("{") and not self.said:
            inner = ReplyFilter()
            clauses = inner.feed(leaked_content(self.raw)) + inner.finish()
            return self._emit(clauses)
        if self.closed or not self.pending.strip():
            return []
        rest, self.pending = self.pending, ""
        return self._emit([rest])


def last_question(reply: str) -> str:
    """The last question of a reply ("" when it asks nothing): what the caller's next words answer."""
    questions = re.findall(r"[^.!?]*\?", reply)
    return questions[-1].strip() if questions else ""


def options_phrase(queue_names: list[str]) -> str:
    """The departments the agent can connect with, from the bundle queues ("Телефония (внутренние)" and
    "Телефония (мобильные)" are one department)."""
    names: list[str] = []
    for name in queue_names:
        short = re.sub(r"\s*\([^)]*\)", "", name).strip(" -")
        if short and short.casefold() not in {n.casefold() for n in names}:
            names.append(short)
    if not names:
        return PHRASES["offer_transfer"]
    return PHRASES["options"].format(names=", ".join(names[:8]))


def stutter(clause: str) -> bool:
    """A word (4 letters or more) three times in a row, endings aside: the model looping, not speech."""
    stems = [w[:4] for w in re.findall(r"[а-яёa-z]+", clause.lower())]
    return any(len(a) == 4 and a == b == c for a, b, c in zip(stems, stems[1:], stems[2:], strict=False))


def named_offer(name: str) -> str:
    return PHRASES["offer_named"].format(name=name)


def english_note(clause: str) -> bool:
    """An English reasoning note rather than speech: three Latin words or more and over twice as many Latin
    letters as Cyrillic ones (brand names inside a Russian sentence stay)."""
    latin = LATIN_WORD.findall(clause)
    return len(latin) >= 3 and sum(map(len, latin)) > 2 * len(CYRILLIC_LETTER.findall(clause))


def parse_args(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def missing_callback_fields(data: dict, said: str | None = None) -> list[str]:
    """Fields still to ask; with ``said`` (everything the caller said in the call) a name or time the model
    made up counts as missing."""
    missing = [k for k in CALLBACK_FIELDS if not str(data.get(k, "")).strip()]
    if "topic" not in missing and not any(len(w) >= 4 and not CALLBACK_WORD.match(w)
                                          for w in re.findall(r"[а-яёa-z0-9]+", str(data["topic"]).lower())):
        missing.append("topic")  # "перезвонить клиенту" is the request itself, not the caller's question
    if said is not None:
        heard = {w[:4] for w in re.findall(r"[а-яёa-z]{3,}", said.lower())}
        name_words = [w for w in re.findall(r"[а-яёa-z]{3,}", str(data.get("name", "")).lower())]
        if "name" not in missing and (not name_words or str(data["name"]).strip().lower() in PLACEHOLDER_NAMES
                                      or not any(w[:4] in heard for w in name_words)):
            missing.append("name")
        if "preferred_time" not in missing and (not TIME_SAID.search(said)
                                                 or PLACEHOLDER_TIME.search(str(data["preferred_time"]))):
            missing.append("preferred_time")
    return [k for k in CALLBACK_FIELDS if k in missing]


def callback_readback(callback: dict, caller: str) -> str:
    """The saved request read back, so a wrong name, time or number can be corrected at once."""
    phone = callback["phone"]
    to = ("на номер, с которого вы звоните" if caller and re.sub(r"\D", "", phone) == re.sub(r"\D", "", caller)
          else f"на номер {phone}")
    return (f"Записываю заявку: {callback['name']}, перезвоним {callback['preferred_time']} {to}. "
            "Если что-то не так, поправьте. Чем ещё могу помочь?")


def callback_phone(raw: str, caller: str) -> str:
    """Digits of the phone to call back; "на этот номер" (no digits) means the number the caller uses, which
    may be a short internal number. Another number must be complete: 10 digits or more."""
    digits = re.sub(r"[^\d+]", "", raw or "")
    if len(re.sub(r"\D", "", digits)) >= 10:
        return digits
    if digits and caller and re.sub(r"\D", "", digits) == re.sub(r"\D", "", caller):
        return caller
    return caller if not digits and caller else ""


@dataclass
class ToolDecision:
    """What a finished turn does: which tool really runs and which extra phrase is said."""
    tool: str | None
    args: dict = field(default_factory=dict)
    phrase: str = ""  # text to say after the model's reply ("" = nothing)
    phrase_cached: bool = True  # phrase is a fixed phrase (pre-synthesized) rather than model text
    tool_result: str | None = None
    callback: dict | None = None
    list_options: bool = False  # the caller asked where they can be connected: name the departments


@dataclass
class CallContext:
    caller: str
    callbacks_enabled: bool
    callback_wanted: bool = False  # the caller asked for a callback or was offered one
    asked_anything_else: bool = False  # the agent's last words asked "Чем ещё могу помочь?"
    caller_said: tuple[str, ...] = ()  # the caller's earlier phrases (a callback name or time must come from them)
    off_topic: int = 0  # off-topic phrases in a row (topic.py)


def decide(tool: str | None, raw_args: str, caller_text: str, reply: str, ctx: CallContext) -> ToolDecision:
    """Applies the tool guards to what the model returned for one caller utterance."""
    args = parse_args(raw_args)
    if tool is None and not reply.strip():
        return ToolDecision(None, phrase=PHRASES["anything_else"])  # nothing was said (e.g. only *narration*)
    done = ctx.asked_anything_else and REFUSAL.search(caller_text) and "?" not in caller_text
    if tool == "end_call" and ctx.asked_anything_else and "?" not in caller_text and FAREWELL.search(reply):
        done = True  # "Всё так." after "...Чем ещё могу помочь?": the model already said goodbye
    if tool == "end_call" and not (FAREWELL.search(caller_text) or END_REQUEST.search(caller_text) or done):
        return ToolDecision(None, phrase=PHRASES["anything_else"])  # "нет, спасибо" is not a goodbye
    asked = reply.rstrip().endswith("?")
    if tool == "transfer_call" and OPTIONS_QUESTION.search(caller_text) and (
            "?" in caller_text or not EXPLICIT_TRANSFER.search(caller_text)):
        # "А какие отделы есть?": a question about the options, not a request to be connected
        return ToolDecision(None, phrase="" if asked else PHRASES["offer_transfer"], list_options=not reply.strip())
    if tool == "transfer_call" and not EXPLICIT_TRANSFER.search(caller_text) and (
            reply.strip() or "?" in caller_text or ctx.callback_wanted):
        return ToolDecision(None, phrase="" if asked else PHRASES["offer_transfer"])  # answered and transferred: ask
    if tool == "transfer_call":
        phrase = str(args.get("phrase", "")).strip()
        if phrase and not reply.strip():
            return ToolDecision(tool, args, phrase=phrase, phrase_cached=False)
        return ToolDecision(tool, args, phrase="" if reply.strip() else PHRASES["transfer"])
    if tool == "end_call":
        return ToolDecision(tool, args, phrase="" if FAREWELL.search(reply) else PHRASES["goodbye"])
    if tool == "request_callback" and not ctx.callbacks_enabled:
        return ToolDecision(None, phrase=PHRASES["anything_else"])
    if tool == "request_callback" and not (ctx.callback_wanted or any(
            CALLBACK_INTENT.search(said) for said in (*ctx.caller_said, caller_text))):
        ctx.callback_wanted = True  # the next "да" may go straight into the request
        return ToolDecision(None, phrase=PHRASES["offer_callback"])
    if tool == "request_callback":
        ctx.callback_wanted = True
        args = dict(args)
        raw_phone = str(args.get("phone", ""))
        args["phone"] = callback_phone(raw_phone, ctx.caller)
        if reply.rstrip().endswith("?"):  # it is still asking (time, number, "всё верно?"): the answer first
            return ToolDecision(tool, args, tool_result="Заявка не сохранена: сначала дождись ответа на свой "
                                "вопрос, потом вызови request_callback ещё раз.")
        missing = missing_callback_fields(args, " ".join((*ctx.caller_said, caller_text)))
        if missing:  # the model called too early or made a field up: ask for what is missing ourselves
            partial = 0 < len(re.sub(r"\D", "", str(raw_phone))) and "phone" in missing
            ask = ("ask_phone_full" if partial and missing[0] == "phone"
                   else "ask_name_and_time" if missing[0] == "name" and "preferred_time" in missing
                   else "ask_" + missing[0])
            return ToolDecision(tool, args, phrase=PHRASES[ask],
                                tool_result="Заявка не сохранена, не хватает: " + ", ".join(missing) + ".")
        callback = {k: str(args[k]).strip()[:500] for k in CALLBACK_FIELDS}
        if "заявк" in reply.lower():  # the model confirmed it itself: no second confirmation
            return ToolDecision(tool, args, callback=callback)
        return ToolDecision(tool, args, phrase=callback_readback(callback, ctx.caller), phrase_cached=False,
                            callback=callback)
    return ToolDecision(tool, args)
