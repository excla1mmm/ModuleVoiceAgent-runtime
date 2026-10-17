"""Terms that bias speech recognition towards the words of one agent (8 kHz audio garbles rare words).

Sources, in priority order: the bundle's optional ``hotwords`` list, then terms derived from the bundle
itself: company name, directory names (each word of the name and the aliases), positions' departments,
queue names, knowledge topics, names in quotes («Бизнес», "Старт") and abbreviations (АТС, CRM) found in the
rules and answers. The list is capped at ``MAX_TERMS``.

Two uses:

* decoder biasing (``biasing.py``) for every term of three letters or more;
* a conservative text correction for abbreviations, which the decoder cannot fix: telephony often loses
  the first vowel ("АТС" -> "ТС"), and a biased decoder never writes a token where the model wrote nothing.
  An all-capital word is replaced only when it equals a known abbreviation without its first letter, it is
  not itself a known term and exactly one abbreviation matches. Other words are never changed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .bundle import AgentBundle
from .normalize import unstressed

MAX_TERMS = 300
MAX_TERM_LENGTH = 100
MIN_TERM_LETTERS = 3
_QUOTED = re.compile(r"«([^«»]{2,60})»|\"([^\"]{2,60})\"|“([^“”]{2,60})”")
_ABBREVIATION = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{2,8})(?![\w-])")
_WORD = re.compile(r"[\w-]+")
_PARENTHESES = re.compile(r"\([^)]*\)")


def normalize_terms(terms: Iterable[str]) -> list[str]:
    """Whitespace collapsed, deduplicated ignoring case, at least 3 letters, at most ``MAX_TERMS``."""
    result: list[str] = []
    seen: set[str] = set()
    for submitted in terms:
        term = " ".join(unstressed(str(submitted)).split())[:MAX_TERM_LENGTH].strip(" .,;:!?«»\"“”")
        key = term.casefold()
        if sum(ch.isalpha() for ch in term) < MIN_TERM_LETTERS or key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) == MAX_TERMS:
            break
    return result


def _quoted(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in _QUOTED.finditer(text)]


def bundle_terms(bundle: AgentBundle) -> tuple[str, ...]:
    """The agent's terms: explicit hotwords first, then what the bundle itself names."""
    terms: list[str] = list(bundle.hotwords)
    texts = [bundle.company_name, bundle.rules]
    terms += [bundle.company_name.strip("«»\"“” "), *_quoted(bundle.company_name)]
    for entry in bundle.directory:
        terms += _WORD.findall(entry.name)
        terms += entry.aliases
        if entry.department:
            terms.append(entry.department)
    terms += [_PARENTHESES.sub("", q.name) for q in bundle.queues]
    for item in bundle.knowledge:
        terms.append(item.topic)
        texts += [item.topic, item.answer]
    for text in texts:
        terms += _quoted(text)
        terms += _ABBREVIATION.findall(text)
    return tuple(normalize_terms(terms))


def abbreviation_fixes(terms: Iterable[str]) -> dict[str, str]:
    """Recognized form -> abbreviation: "ТС" -> "АТС". Ambiguous and known forms are left out."""
    known = {t for t in terms if _ABBREVIATION.fullmatch(t)}
    fixes: dict[str, str | None] = {}
    for term in known:
        heard = term[1:]
        if len(heard) >= 2 and heard not in known:
            fixes[heard] = None if heard in fixes else term
    return {heard: term for heard, term in fixes.items() if term is not None}


def correct(text: str, fixes: dict[str, str]) -> str:
    if not fixes or not text:
        return text
    return _ABBREVIATION.sub(lambda m: fixes.get(m.group(1), m.group(1)), text)
