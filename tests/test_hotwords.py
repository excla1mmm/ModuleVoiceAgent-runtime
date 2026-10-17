"""Hotwords: terms from the bundle, decoder biasing guards, abbreviation correction."""

import numpy as np
import pytest

from conftest import bundle_data
from voice_agent_runtime.biasing import MAX_TOKEN_BOOST, PhraseTree, compile_terms
from voice_agent_runtime.bundle import BundleError, parse_bundle
from voice_agent_runtime.hotwords import MAX_TERMS, abbreviation_fixes, bundle_terms, correct, normalize_terms


def test_terms_come_from_hotwords_first_then_the_bundle():
    bundle = parse_bundle("3", bundle_data(hotwords=["виртуальная АТС", " МикоПБХ "],
                                           rules="Тариф «Бизнес» стоит 12 900 рублей. Подключаем АТС и CRM."))
    terms = bundle_terms(bundle)
    assert terms[:2] == ("виртуальная АТС", "МикоПБХ")
    for expected in ("Телеком Сервис", "Иванова", "Светлана", "Света", "Бухгалтерия", "Руководство", "Тарифы",
                     "Старт", "Бизнес", "АТС", "CRM"):
        assert expected in terms
    assert "(общая очередь)" not in " ".join(terms)
    assert len({t.casefold() for t in terms}) == len(terms)


def test_hotwords_field_is_optional_and_validated():
    assert parse_bundle("3", bundle_data()).hotwords == ()
    assert parse_bundle("3", bundle_data(hotwords=None)).hotwords == ()
    for bad in ("АТС", [1], ["x" * 101], ["АТС"] * 201):
        with pytest.raises(BundleError):
            parse_bundle("3", bundle_data(hotwords=bad))


def test_normalize_terms_dedupes_and_caps():
    assert normalize_terms(["АТС", "атс", "  Мико   ПБХ ", "ИП", "«Старт»"]) == ["АТС", "Мико ПБХ", "Старт"]
    assert len(normalize_terms(f"термин{i}" for i in range(1000))) == MAX_TERMS


def test_abbreviation_with_lost_first_letter_is_restored():
    fixes = abbreviation_fixes(["АТС", "Бухгалтерия", "CRM"])
    assert fixes == {"ТС": "АТС", "RM": "CRM"}
    assert correct("Сколько стоит ТС для офиса?", fixes) == "Сколько стоит АТС для офиса?"
    assert correct("ТС, а потом ТС.", fixes) == "АТС, а потом АТС."


@pytest.mark.parametrize("text", [
    "Я абонент МТС.",  # a different abbreviation
    "Где моё ТСЖ?",  # a longer word
    "У нас тс в офисе.",  # not all capitals
    "ТСК и АТСК",  # not a whole word match
    "Здравствуйте, соедините меня с бухгалтерией, пожалуйста.",
])
def test_normal_text_is_never_changed(text):
    assert correct(text, abbreviation_fixes(["АТС", "Бухгалтерия", "CRM"])) == text


def test_ambiguous_or_known_forms_are_not_fixed():
    assert abbreviation_fixes(["АТС", "МТС"]) == {}  # "ТС" could be either
    assert abbreviation_fixes(["АТСЖ", "ТСЖ"]) == {"СЖ": "ТСЖ"}  # "ТСЖ" is a term of its own, never rewritten
    assert correct("ТС", {}) == "ТС"


# --- decoder biasing on synthetic scores ---------------------------------------------------------------------

VOCAB = {0: "<blk>", 1: " а", 2: "т", 3: "с", 4: " м", 5: "к", 6: " т"}


def bias():
    return compile_terms(["АТС"], VOCAB, blank=0, alpha=0.45)


def test_bonus_starts_at_the_third_letter():
    b = bias()
    node = b.tree.start
    assert b.bonus(node, 1) == 0.0  # " а"
    node = b.transition(node, 1)[0]
    assert b.bonus(node, 2) == 0.0  # "т"
    node = b.transition(node, 2)[0]
    assert b.bonus(node, 3) == pytest.approx(0.9)  # "с" completes the term
    assert b.bonus(node, 5) == 0.0  # leaving the term costs nothing


def test_select_keeps_blank_and_respects_the_cap():
    b = bias()
    node = b.transition(b.transition(b.tree.start, 1)[0], 2)[0]  # after " ат"
    scores = np.array([-1.0, -9, -9, -1.5, -9, -1.0, -9])
    assert b.select(scores, node, blank=0) == 0  # the model wrote nothing: that stands
    scores = np.array([-9.0, -9, -9, -1.5, -9, -1.0, -9])
    assert b.select(scores, node, blank=0) == 3  # a close continuation of the term wins
    scores = np.array([-9.0, -9, -9, -1.0 - MAX_TOKEN_BOOST - 0.1, -9, -1.0, -9])
    assert b.select(scores, node, blank=0) == 5  # beyond the cap the model's choice stands


def test_select_without_a_term_in_progress_is_argmax():
    b = bias()
    scores = np.array([-9.0, -3, -9, -9, -1, -9, -2])
    assert b.select(scores, b.tree.start, blank=0) == 4


def test_unspellable_terms_and_zero_alpha_compile_to_nothing():
    assert compile_terms(["CRM"], VOCAB, blank=0, alpha=0.45) is None
    assert compile_terms(["АТС"], VOCAB, blank=0, alpha=0.0) is None
    assert compile_terms(["АТС", "CRM"], VOCAB, blank=0, alpha=0.45).terms == 1


def test_phrase_tree_is_independent_of_token_segmentation():
    tree = PhraseTree([" атс", " тсж"])

    def walk(*tokens):
        node, total = tree.start, 0.0
        for token in tokens:
            node, delta = tree.step(node, token)
            total += delta
        return total

    assert walk(" а", "тс") == walk(" ат", "с") == walk(" атс") == pytest.approx(2.0)
    assert walk(" ам", "к") == 0.0  # no third letter of a term reached
