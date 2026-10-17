"""Turns the model's answer into text Piper can read: numbers, units, times, dates, abbreviations, Latin.

The model writes numbers as digits exactly as in the knowledge base (it copies digits reliably but made
mistakes spelling them out: 2900 became "девятьсот", 12900 "тринадцать тысяч"). Here they become words:

* after prepositions that take the genitive ("до", "от", "с" ...) the numeral is declined:
  "до 100" -> "до ста", "с 10:00 до 15:00" -> "с десяти до пятнадцати";
* a unit agrees with its number: "1 Гбит/с" -> "один гигабит в секунду", "до 1 Гбит/с" ->
  "до одного гигабита в секунду", "2900 ₽/мес" -> "две тысячи девятьсот рублей в месяц";
* "в 14:00" -> "в четырнадцать часов"; "до 10 числа" -> "до десятого числа";
* long digit strings are phone numbers and are read in groups;
* Latin words, domains and e-mails are read in Russian: "lk.example.ru" -> "эл кей точка экзампл точка ру".

num2words declines Russian numerals itself (case and gender), so no morphological analyzer is needed.
"""

from __future__ import annotations

import re

from num2words import num2words

GENITIVE_PREPS = r"до|от|с|со|около|более|менее|свыше|из|без|после|для|кроме|вместо|у"
SEP = "[   ]"
NUM = rf"\d+(?:{SEP}\d{{3}})*(?:[.,]\d+)?"

# (pattern, (one, few, many), gender, tail)
UNIT_FORMS: list[tuple[str, tuple[str, str, str], str, str]] = [
    (r"Гбит/с", ("гигабит", "гигабита", "гигабит"), "m", " в секунду"),
    (r"Мбит/с", ("мегабит", "мегабита", "мегабит"), "m", " в секунду"),
    (r"Кбит/с", ("килобит", "килобита", "килобит"), "m", " в секунду"),
    (r"[КK][бb]/с", ("килобайт", "килобайта", "килобайт"), "m", " в секунду"),
    (r"М[бb]/с", ("мегабайт", "мегабайта", "мегабайт"), "m", " в секунду"),
    (r"ТБ\b", ("терабайт", "терабайта", "терабайт"), "m", ""),
    (r"ГБ\b", ("гигабайт", "гигабайта", "гигабайт"), "m", ""),
    (r"МБ\b", ("мегабайт", "мегабайта", "мегабайт"), "m", ""),
    (r"(?:₽|руб\.?)\s*/\s*мес\.?", ("рубль", "рубля", "рублей"), "m", " в месяц"),
    (r"(?:₽|руб\.?)\s*/\s*год\.?", ("рубль", "рубля", "рублей"), "m", " в год"),
    (r"(?:₽|руб\.(?=[\s,;)]|$)|руб\b)", ("рубль", "рубля", "рублей"), "m", ""),
    (r"%", ("процент", "процента", "процентов"), "m", ""),
    (r"мин\.(?=\s|$|[,;)])", ("минута", "минуты", "минут"), "f", ""),
    (r"ч\.(?=\s|$|[,;)])", ("час", "часа", "часов"), "m", ""),
]
UNIT_RE = re.compile(
    rf"(?:\b({GENITIVE_PREPS}){SEP}+)?({NUM})\s*({'|'.join(f'(?:{p})' for p, *_ in UNIT_FORMS)})", re.I)
_UNIT_PATTERNS = [(re.compile(p, re.I), forms, gender, tail) for p, forms, gender, tail in UNIT_FORMS]

ABBREVIATIONS = [
    (r"\b[Ww]i-?[Ff]i\b", "вай фай"), (r"\s*/\s*мес\.?", " в месяц"), (r"№\s*", "номер "),
    (r"\bул\.\s*", "улица "), (r"\bд\.\s*(?=\d)", "дом "), (r"\bкорп\.\s*", "корпус "), (r"\bоф\.\s*", "офис "),
    (r"\bпр-т\b\.?\s*", "проспект "), (r"\bг\.\s*(?=[А-ЯЁ])", "город "), (r"\bт\.\s?е\.", "то есть"),
    (r"\bт\.\s?д\.", "так далее"),
]
PHONE = re.compile(r"\+?\d[\d\-()   ]{5,}\d")
TIME = re.compile(rf"\b(?:({GENITIVE_PREPS}|в|к|на){SEP}+)?([01]?\d|2[0-3]):([0-5]\d)\b", re.I)
MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
DAY_OF_MONTH = re.compile(rf"\b(?:({GENITIVE_PREPS}|к|на|в){SEP}+)?([1-9]|[12]\d|3[01])(?:-?го)?{SEP}+(числа|числу|число|{MONTHS})\b", re.I)
PREP_NUMBER = re.compile(rf"\b({GENITIVE_PREPS}){SEP}+(\d+(?:{SEP}\d{{3}})*)(?![\d:.,])", re.I)
NUMBER = re.compile(NUM)

LATIN = re.compile(r"[A-Za-z]+")
# A stress mark set by hand in the agent texts: "+" right before a Cyrillic vowel at the start of or inside a word
# ("з+амок", "МИК+О"). Silero honours it; anything else with "+" ("+7 999") is not a stress mark.
STRESS = re.compile(r"(?<![^\sА-Яа-яЁё«\"(])\+(?=[АЕЁИОУЫЭЮЯаеёиоуыэюя])")


def unstressed(text: str) -> str:
    """The text without hand-set stress marks (for engines and matching that do not know them)."""
    return STRESS.sub("", text)
LATIN_WORDS = {"ru": "ру", "com": "ком", "www": "вэ вэ вэ", "net": "нет", "org": "орг", "example": "экзампл",
               "email": "имэйл", "wifi": "вай фай", "smart": "смарт", "tv": "ти ви", "ip": "ай пи", "sms": "эс эм эс"}
LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz", ["эй", "би", "си", "ди", "и", "эф", "джи", "эйч", "ай", "джей", "кей",
               "эл", "эм", "эн", "оу", "пи", "кью", "ар", "эс", "ти", "ю", "ви", "дабл ю", "экс", "уай", "зед"], strict=True))
TRANSLIT = [("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("oo", "у"), ("ee", "и"), ("x", "кс"), ("a", "а"),
            ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"), ("i", "и"), ("j", "дж"),
            ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"),
            ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("y", "и"), ("z", "з")]


def words(n: int, case: str = "nominative", gender: str = "m") -> str:
    return num2words(n, lang="ru", case=case, gender=gender)


def plural_form(n: int, forms: tuple[str, str, str], genitive: bool = False) -> str:
    """Picks the noun form that agrees with n; in the genitive "одного гигабита" but "двух гигабит"."""
    one, few, many = forms
    last_two, last = n % 100, n % 10
    if genitive:
        return few if last == 1 and last_two != 11 else many
    if 11 <= last_two <= 14:
        return many
    if last == 1:
        return one
    return few if 2 <= last <= 4 else many


def _digits(raw: str) -> str:
    return re.sub(SEP, "", raw)


def _unit(m: re.Match) -> str:
    prep, raw, unit_text = m.group(1), _digits(m.group(2)), m.group(3)
    forms, gender, tail = next((f, g, t) for p, f, g, t in _UNIT_PATTERNS if p.fullmatch(unit_text))
    case = "genitive" if prep else "nominative"
    if "," in raw or "." in raw:
        whole, frac = re.split(r"[.,]", raw, maxsplit=1)
        spoken = f"{words(int(whole), case, 'f')} и {words(int(frac), case)} {forms[1]}"
    else:
        n = int(raw)
        spoken = f"{words(n, case, gender)} {plural_form(n, forms, genitive=bool(prep))}"
    return (f"{prep} " if prep else "") + spoken + tail


def _phone(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if len(digits) < 7:
        return m.group(0)
    groups = [digits[:1], digits[1:4]] if len(digits) == 11 else []
    rest = digits[4:] if len(digits) == 11 else digits
    while len(rest) > 4:
        groups.append(rest[:3])
        rest = rest[3:]
    groups += [rest[:2], rest[2:]] if len(rest) == 4 else [rest]
    return ", ".join(_group(g) for g in groups if g)


def _group(g: str) -> str:
    return " ".join(words(int(d)) for d in g) if g.startswith("0") and len(g) > 1 else words(int(g))


def _time(m: re.Match) -> str:
    prep, h, mm = m.group(1), int(m.group(2)), int(m.group(3))
    low = (prep or "").lower()
    if low in ("в", "на", ""):
        hour = words(h)
        if mm == 0:
            spoken = f"{hour} {plural_form(h, ('час', 'часа', 'часов'))}"
        else:
            spoken = f"{hour} {'ноль ' + words(mm) if mm < 10 else words(mm)}"
    else:
        case = "dative" if low == "к" else "genitive"
        spoken = words(h, case)
        if mm:
            spoken += " " + ("ноль " + words(mm) if mm < 10 else words(mm, case))
    return f"{prep} {spoken}" if prep else spoken


def _day_of_month(m: re.Match) -> str:
    prep, day, noun = m.group(1), int(m.group(2)), m.group(3)
    low = noun.lower()
    case = {"числу": "dative", "число": "nominative"}.get(low, "genitive")
    if prep and prep.lower() == "к":
        case = "dative"
    spoken = num2words(day, lang="ru", to="ordinal", case=case, gender="n")
    return f"{prep + ' ' if prep else ''}{spoken} {noun}"


def _prep_number(m: re.Match) -> str:
    return f"{m.group(1)} {words(int(_digits(m.group(2))), 'genitive')}"


def _number(m: re.Match) -> str:
    raw = _digits(m.group(0))
    if "," in raw or "." in raw:
        whole, frac = re.split(r"[.,]", raw, maxsplit=1)
        return f"{words(int(whole))} и {words(int(frac))}"
    if len(raw) > 1 and raw.startswith("0"):  # codes are read digit by digit
        return " ".join(words(int(d)) for d in raw)
    return words(int(raw))


def _latin(m: re.Match) -> str:
    word = m.group(0).lower()
    if word in LATIN_WORDS:
        return LATIN_WORDS[word]
    if len(word) <= 3 or not re.search(r"[aeiouy]", word):  # abbreviations are spelled letter by letter
        return " ".join(LETTERS[c] for c in word)
    for lat, cyr in TRANSLIT:
        word = word.replace(lat, cyr)
    return word


def normalize(text: str) -> str:
    text = UNIT_RE.sub(_unit, text)
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    text = PHONE.sub(_phone, text)
    text = TIME.sub(_time, text)
    text = DAY_OF_MONTH.sub(_day_of_month, text)
    text = PREP_NUMBER.sub(_prep_number, text)
    text = NUMBER.sub(_number, text)
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", " точка ", text)
    text = text.replace("@", " собака ")
    text = LATIN.sub(_latin, text)
    text = re.sub(r"[«»\"“”]", "", text)  # quotes are not spoken
    return re.sub(r"\s{2,}", " ", text).strip()
