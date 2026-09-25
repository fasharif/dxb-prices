"""English and Arabic name normalisation.

DLD records the same place with different casing, spacing and punctuation
("DUBAI SPORTS CITY", "Dubai Sports City "), and Arabic names with different
spellings of the same letters (أ/إ/ا, ى/ي, ة/ه). Each name gets a matching
key; rows whose keys agree are the same place.

For communities the Arabic name is the tie-breaker: English spellings that
share an Arabic key are merged. In the 2026 data no English community name
maps to two different Arabic names, and every Arabic name with several
English spellings is one place (see docs/data.md).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_AR_TATWEEL = "ـ"
_AR_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",  # أ -> ا
        "إ": "ا",  # إ -> ا
        "آ": "ا",  # آ -> ا
        "ٱ": "ا",  # ٱ -> ا
        "ى": "ي",  # ى -> ي
        "ة": "ه",  # ة -> ه
        "ؤ": "و",  # ؤ -> و
        "ئ": "ي",  # ئ -> ي
        # Arabic-Indic and Eastern Arabic-Indic digits -> ASCII
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
    }
)
_EN_PUNCT = re.compile(r"[\-_/,.'’`\"()\[\]]+")
_SPACES = re.compile(r"\s+")


def display_en(value: str) -> str:
    """Tidy an English name for display: NFKC, trimmed, single spaces."""
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def key_en(value: str) -> str:
    """Matching key for an English name: case-folded, punctuation and extra spaces removed."""
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    text = _EN_PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def key_ar(value: str) -> str:
    """Matching key for an Arabic name: diacritics and tatweel removed, letter variants unified."""
    text = unicodedata.normalize("NFKC", value)
    text = _AR_DIACRITICS.sub("", text).replace(_AR_TATWEEL, "")
    text = text.translate(_AR_LETTER_MAP)
    text = _EN_PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


_ROMAN = re.compile(r"^[IVX]+$")


def readable(display: str) -> str:
    """Turn an all-capitals name into title case; leave mixed-case names alone."""
    if not display.isupper():
        return display
    words = []
    for word in display.split(" "):
        if _ROMAN.match(word) or any(ch.isdigit() for ch in word):
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


def _most_common(labels: pd.Series) -> str:
    counts = labels.value_counts()
    # Ties resolve alphabetically so the result does not depend on row order.
    top = counts[counts == counts.iloc[0]].index
    return str(sorted(top)[0])


def _merge_by_arabic(en_key: pd.Series, ar_key: pd.Series) -> dict[str, str]:
    """Union English keys that share an Arabic key; returns key -> group representative."""
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    pairs = pd.DataFrame({"en": en_key, "ar": ar_key}).dropna().drop_duplicates()
    for _, keys in pairs.groupby("ar")["en"]:
        first, *rest = sorted(keys)
        for other in rest:
            a, b = find(first), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {k: find(k) for k in parent}


def canonical_english(en: pd.Series, ar: pd.Series | None = None) -> pd.Series:
    """Map raw English names to one readable label per place.

    With ``ar`` given, English spellings that share an Arabic name are treated
    as one place: DLD's Arabic area names are the registry names, while the
    English ones vary ("DUBAI MARITIME CITY" and "Madinat Dubai Almelaheyah"
    are both مدينة دبي الملاحية).
    """
    en_display = en.astype("string").map(display_en, na_action="ignore")
    en_key = en_display.map(key_en, na_action="ignore")
    group_key = en_key.copy()
    if ar is not None:
        ar_key = ar.astype("string").map(key_ar, na_action="ignore")
        merged = _merge_by_arabic(en_key, ar_key)
        group_key = en_key.map(lambda k: merged.get(k, k), na_action="ignore")
    candidates = en_display.map(readable, na_action="ignore")
    labels = (
        pd.DataFrame({"group": group_key, "label": candidates})
        .dropna()
        .groupby("group")["label"]
        .agg(_most_common)
    )
    return group_key.map(labels).astype("string")


def lookup_table(
    canonical: pd.Series, raw_en: pd.Series, raw_ar: pd.Series | None
) -> dict[str, str]:
    """Key -> canonical label, from English and Arabic spellings, for use at serving time."""
    table: dict[str, str] = {}
    pairs = pd.DataFrame(
        {"canon": canonical, "en": raw_en, "ar": raw_ar if raw_ar is not None else pd.NA}
    )
    for row in pairs.dropna(subset=["canon"]).drop_duplicates().itertuples(index=False):
        canon = str(row.canon)
        table.setdefault(key_en(canon), canon)
        if isinstance(row.en, str):
            table.setdefault(key_en(row.en), canon)
        if isinstance(row.ar, str):
            table.setdefault(key_ar(row.ar), canon)
    return table


def resolve_name(value: str, table: dict[str, str]) -> str | None:
    """Find the canonical label for a user-supplied English or Arabic name."""
    for key in (key_en(value), key_ar(value)):
        if key in table:
            return table[key]
    return None
