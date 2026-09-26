from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from dxb_prices import features
from dxb_prices.config import FeatureRules
from dxb_prices.features import MISSING, OTHER

RULES = FeatureRules(min_rows_community=3, min_rows_project=3, min_rows_poi=3)


def frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    base: dict[str, object] = {
        "community": "Business Bay",
        "community_ar": "الخليج التجاري",
        "project": "Canal Heights",
        "rooms": "1",
        "sub_type": "Flat",
        "area_sqm": 75.0,
        "is_off_plan": False,
        "is_freehold": True,
        "month": "2026-01",
        "nearest_metro": "Business Bay Metro Station",
        "nearest_mall": "Dubai Mall",
        "nearest_landmark": "Burj Khalifa",
        "price_aed": 1.5e6,
    }
    return pd.DataFrame([base | dict(r) for r in rows])


def training_frame() -> pd.DataFrame:
    return frame(
        [{"month": "2026-01"}] * 3
        + [
            {
                "month": "2026-02",
                "community": "Marsa Dubai",
                "community_ar": "مرسى دبي",
                "project": "Marina Crest",
            }
        ]
        * 3
        + [{"project": "Tiny Project", "month": "2026-02"}]
    )


def test_levels_keep_only_common_training_categories() -> None:
    spec = features.fit(training_frame(), RULES)
    assert spec.levels["community"] == ["Business Bay", "Marsa Dubai", OTHER, MISSING]
    assert "Tiny Project" not in spec.levels["project"]
    assert spec.reference_month == "2026-01"
    assert spec.community_rows == {"Business Bay": 4, "Marsa Dubai": 3}


def test_unseen_rare_and_missing_values_are_mapped_explicitly() -> None:
    spec = features.fit(training_frame(), RULES)
    later = frame(
        [
            {
                "community": "Al Jaddaf",
                "project": "Tiny Project",
                "month": "2026-05",
                "nearest_metro": None,
            }
        ]
    )
    x = features.transform(later, spec)
    assert x.loc[0, "community"] == OTHER
    assert x.loc[0, "project"] == OTHER
    assert x.loc[0, "nearest_metro"] == MISSING
    assert x.loc[0, "month_index"] == 4.0


def test_fitting_uses_only_the_rows_it_is_given() -> None:
    train = training_frame()
    test = frame([{"community": "Palm Jumeirah", "month": "2026-03"}] * 50)
    spec = features.fit(train, RULES)
    assert "Palm Jumeirah" not in spec.levels["community"]
    assert "Palm Jumeirah" not in spec.community_rows
    assert features.fit(train, RULES).to_dict() == spec.to_dict()
    # Transforming new rows never changes the spec.
    features.transform(test, spec)
    assert "Palm Jumeirah" not in spec.levels["community"]


def test_matrix_has_fixed_columns_and_category_order() -> None:
    spec = features.fit(training_frame(), RULES)
    a = features.transform(frame([{"community": "Marsa Dubai"}]), spec)
    b = features.transform(frame([{"community": "Business Bay"}]), spec)
    assert list(a.columns) == list(features.FEATURES)
    for col in features.CATEGORICAL_FEATURES:
        assert list(a[col].cat.categories) == list(b[col].cat.categories) == spec.levels[col]
    for col in features.NUMERIC_FEATURES:
        assert a[col].dtype == np.float64


def test_no_price_information_reaches_the_features() -> None:
    assert not any("price" in f for f in features.FEATURES)
    spec = features.fit(training_frame(), RULES)
    x1 = features.transform(frame([{"price_aed": 1.0}]), spec)
    x2 = features.transform(frame([{"price_aed": 9e9}]), spec)
    pd.testing.assert_frame_equal(x1, x2)


def test_context_lookups_come_from_training_rows() -> None:
    spec = features.fit(training_frame(), RULES)
    assert spec.project_context["Marina Crest"]["community"] == "Marsa Dubai"
    assert spec.community_context["Business Bay"]["nearest_mall"] == "Dubai Mall"
    assert spec.community_context["Business Bay"]["is_freehold"] is True
    assert spec.community_names["الخليج التجاري"] == "Business Bay"


def test_spec_round_trips_through_json() -> None:
    spec = features.fit(training_frame(), RULES)
    again = features.FeatureSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec


def test_transform_reports_missing_columns() -> None:
    spec = features.fit(training_frame(), RULES)
    with pytest.raises(KeyError, match="community"):
        features.transform(frame([{}]).drop(columns=["community"]), spec)


def test_fit_refuses_empty_training_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        features.fit(training_frame().iloc[0:0], RULES)
