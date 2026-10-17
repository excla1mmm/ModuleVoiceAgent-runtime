"""Guards on model output. Every rule here was needed in a real call.

* gpt-oss sometimes prints a whole message as JSON (``{"role":"assistant","content":...}``): only the
  content is said.
* it sometimes appends reasoning notes such as ``(Waiting for user)(We ...``: everything from the note on
  is dropped; a clause that is an English note ("Proceed with request_callback using phone 333.") is not said.
* one question per reply, and a second message glued to the first without a space (``...звонка.Буду``)
  is cut.
* ``end_call`` without a farewell, an explicit request to end the call ("заверши звонок", "сбрось", "хватит
  уже", "уйди") or "ничем" / "нет" in answer to "Чем ещё могу помочь?" becomes "Хорошо. Чем ещё могу помочь?"
  ("нет, спасибо" answering an offer is not the end of the conversation).
* it sometimes narrates an action instead of calling the tool (``*Звонок завершён.*``): text between single
  asterisks is not said; ``**bold**`` markers are dropped and the words kept. A reply left empty asks what else
  the caller needs.
* a transfer that comes together with an answer, when the caller did not ask to be connected, becomes an
  offer (not a second one if the answer already asks); the transfer happens after "да". A transfer on a question
  ("А какие отделы есть?", "С кем можешь соединить?") does not happen either: the agent names the departments.
* tool arguments printed as text (``...{"phrase": ..., "extension": "2588"}``) are not said.
* ``request_callback`` only after the caller asked for it or accepted an offer; incomplete requests are not
  saved, the first missing field is asked by us. A name the caller never said ("Клиент") or a time when the
  caller named none ("сейчас" made up) counts as missing; no second confirmation when the model confirmed.
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
# ("Соединить вас?") it does not
REFUSAL = re.compile(r"^\W*(?:да\W+|нет\W+|ну\W+)?(?:ничем|ничего|не надо|не нужно|нет)\b", re.I)
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
    "callback_saved": "Записал вашу заявку, специалист перезвонит в указанное время. Могу ещё чем-нибудь помочь?",
    "unavailable": "Извините, сервис временно недоступен. Соединяю вас с сотрудником.",
    "transfer_denied": "К сожалению, не могу соединить с этим номером. Чем ещё могу помочь?",
    "options": "Могу соединить с отделами: {names}. Куда вас соединить?",
    "ask_name": "Как к вам обращаться?",
    "ask_phone": "Перезвонить вам на номер, с которого вы звоните?",
    "ask_preferred_time": "Когда вам удобно, чтобы мы перезвонили?",
    "ask_topic": "По какому вопросу вам перезвонить?",
}
CALLBACK_FIELDS = ("name", "phone", "preferred_time", "topic")
PLACEHOLDER_NAMES = {"клиент", "абонент", "звонящий", "гость", "пользователь", "неизвестно", "не указано", "аноним"}
TIME_SAID = re.compile(r"\d|утр|вечер|д[её]н|ночь|сейчас|сегодня|завтра|послезавтра|понедельн|вторн|сред[уаы]|четверг"
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
        out = [c for c in (GLUED_REMARK.sub("", re.sub(r"\s+", " ", self._unstaged(c))).strip() for c in clauses)
               if c and not english_note(c)]
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
    if "topic" not in missing and re.search(r"перезвон|обратн\w*\s+звон", str(data["topic"]), re.I):
        missing.append("topic")  # "перезвонить клиенту" is the request itself, not the caller's question
    if said is not None:
        heard = {w[:4] for w in re.findall(r"[а-яёa-z]{3,}", said.lower())}
        name_words = [w for w in re.findall(r"[а-яёa-z]{3,}", str(data.get("name", "")).lower())]
        if "name" not in missing and (not name_words or str(data["name"]).strip().lower() in PLACEHOLDER_NAMES
                                      or not any(w[:4] in heard for w in name_words)):
            missing.append("name")
        if "preferred_time" not in missing and not TIME_SAID.search(said):
            missing.append("preferred_time")
    return [k for k in CALLBACK_FIELDS if k in missing]


def callback_phone(raw: str, caller: str) -> str:
    """Digits of the phone to call back; "на этот номер" (no digits) means the number the caller uses, which
    may be a short internal number."""
    digits = re.sub(r"[^\d+]", "", raw or "")
    if len(re.sub(r"\D", "", digits)) >= 5:
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


def decide(tool: str | None, raw_args: str, caller_text: str, reply: str, ctx: CallContext) -> ToolDecision:
    """Applies the tool guards to what the model returned for one caller utterance."""
    args = parse_args(raw_args)
    if tool is None and not reply.strip():
        return ToolDecision(None, phrase=PHRASES["anything_else"])  # nothing was said (e.g. only *narration*)
    done = ctx.asked_anything_else and REFUSAL.search(caller_text) and "?" not in caller_text
    if tool == "end_call" and not (FAREWELL.search(caller_text) or END_REQUEST.search(caller_text) or done):
        return ToolDecision(None, phrase=PHRASES["anything_else"])  # "нет, спасибо" is not a goodbye
    asked = reply.rstrip().endswith("?")
    if tool == "transfer_call" and OPTIONS_QUESTION.search(caller_text) and (
            "?" in caller_text or not EXPLICIT_TRANSFER.search(caller_text)):
        # "А какие отделы есть?": a question about the options, not a request to be connected
        return ToolDecision(None, phrase="" if asked else PHRASES["offer_transfer"], list_options=not reply.strip())
    if tool == "transfer_call" and not EXPLICIT_TRANSFER.search(caller_text) and (reply.strip() or "?" in caller_text):
        return ToolDecision(None, phrase="" if asked else PHRASES["offer_transfer"])  # answered and transferred: ask
    if tool == "transfer_call":
        phrase = str(args.get("phrase", "")).strip()
        if phrase and not reply.strip():
            return ToolDecision(tool, args, phrase=phrase, phrase_cached=False)
        return ToolDecision(tool, args, phrase="" if reply.strip() else PHRASES["transfer"])
    if tool == "end_call":
        return ToolDecision(tool, args, phrase=PHRASES["goodbye"])
    if tool == "request_callback" and not ctx.callbacks_enabled:
        return ToolDecision(None, phrase=PHRASES["anything_else"])
    if tool == "request_callback" and not (ctx.callback_wanted or CALLBACK_INTENT.search(caller_text)):
        ctx.callback_wanted = True  # the next "да" may go straight into the request
        return ToolDecision(None, phrase=PHRASES["offer_callback"])
    if tool == "request_callback":
        ctx.callback_wanted = True
        args = dict(args)
        args["phone"] = callback_phone(str(args.get("phone", "")), ctx.caller)
        missing = missing_callback_fields(args, " ".join((*ctx.caller_said, caller_text)))
        if missing:  # the model called too early or made a field up: ask for the first missing one ourselves
            return ToolDecision(tool, args, phrase=PHRASES["ask_" + missing[0]],
                                tool_result="Заявка не сохранена, не хватает: " + ", ".join(missing) + ".")
        callback = {k: str(args[k]).strip()[:500] for k in CALLBACK_FIELDS}
        confirmed = "заявк" in reply.lower()  # the model said it itself: no second confirmation
        return ToolDecision(tool, args, phrase="" if confirmed else PHRASES["callback_saved"], callback=callback)
    return ToolDecision(tool, args)
