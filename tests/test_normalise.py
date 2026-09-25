from __future__ import annotations

import pandas as pd

from dxb_prices import normalise


def test_english_key_ignores_case_spacing_and_punctuation() -> None:
    assert normalise.key_en("DUBAI SPORTS CITY") == normalise.key_en("  Dubai  Sports City ")
    assert normalise.key_en("Al-Barsha") == normalise.key_en("al barsha")
    assert normalise.key_en("Bay & Park") == "bay and park"


def test_arabic_key_unifies_letter_variants() -> None:
    # ى / ي at the end of "Business Bay"
    assert normalise.key_ar("الخليج التجارى") == normalise.key_ar("الخليج التجاري")
    # hamza forms of alef, taa marbuta and diacritics
    assert normalise.key_ar("أبراج") == normalise.key_ar("ابراج")
    assert normalise.key_ar("إمارة") == normalise.key_ar("اماره")
    assert normalise.key_ar("مَدِينَة") == normalise.key_ar("مدينه")
    # tatweel and Arabic-Indic digits
    assert normalise.key_ar("نخـــلة ٢") == normalise.key_ar("نخلة 2")


def test_readable_title_cases_capitals_but_keeps_numerals() -> None:
    assert normalise.readable("PALM JUMEIRAH") == "Palm Jumeirah"
    assert normalise.readable("SOBHA HARTLAND II") == "Sobha Hartland II"
    assert normalise.readable("TOWER 2B") == "Tower 2B"
    assert normalise.readable("Il Primo") == "Il Primo"


def test_communities_sharing_an_arabic_name_are_merged() -> None:
    en = pd.Series(
        ["DUBAI MARITIME CITY"] * 3
        + ["Madinat Dubai Almelaheyah"] * 2
        + ["BUSINESS BAY", "Business Bay", "Palm Deira"]
    )
    ar = pd.Series(["مدينة دبي الملاحية"] * 5 + ["الخليج التجاري", "الخليج التجارى", "نخلة ديرة"])
    out = normalise.canonical_english(en, ar)
    assert out.iloc[:5].tolist() == ["Dubai Maritime City"] * 5
    assert out.iloc[5] == out.iloc[6] == "Business Bay"
    assert out.iloc[7] == "Palm Deira"


def test_distinct_places_stay_distinct() -> None:
    en = pd.Series(["Dubai Investment Park First", "Dubai Investment Park Second"])
    ar = pd.Series(["مجمع دبي للاستثمار الاول", "مجمع دبي للاستثمار الثاني"])
    out = normalise.canonical_english(en, ar)
    assert out.nunique() == 2


def test_missing_names_stay_missing() -> None:
    out = normalise.canonical_english(
        pd.Series(["Marsa Dubai", None]), pd.Series(["مرسى دبي", None])
    )
    assert out.iloc[0] == "Marsa Dubai"
    assert pd.isna(out.iloc[1])


def test_lookup_table_resolves_english_and_arabic_spellings() -> None:
    canonical = pd.Series(["Business Bay", "Business Bay"])
    raw_en = pd.Series(["BUSINESS BAY", "Business Bay"])
    raw_ar = pd.Series(["الخليج التجاري", "الخليج التجارى"])
    table = normalise.lookup_table(canonical, raw_en, raw_ar)
    assert normalise.resolve_name("business  bay", table) == "Business Bay"
    assert normalise.resolve_name("الخليج التجارى", table) == "Business Bay"
    assert normalise.resolve_name("Downtown", table) is None
