from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dxb_prices.api.app import create_app
from dxb_prices.api.estimator import Estimator
from dxb_prices.api.schemas import EstimateRequest, EstimateResponse
from dxb_prices.model import PriceModel

VALID: dict[str, Any] = {
    "community": "Business Bay",
    "size_sqm": 80,
    "rooms": "1",
    "off_plan": False,
    "transaction_date": "2026-04-15",
}


@pytest.fixture(scope="module")
def client(tiny_model_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(tiny_model_dir)) as c:
        yield c


def test_health_and_metadata(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model").json()
    assert info["trained_on_months"] == ["2026-01", "2026-02", "2026-03"]
    names = {c["name"] for c in client.get("/communities").json()}
    assert {"Business Bay", "Marsa Dubai", "Dubai Maritime City"} <= names


def test_estimate_returns_price_range_and_top_factors(client: TestClient) -> None:
    r = client.post("/estimate", json=VALID)
    assert r.status_code == 200, r.text
    body = EstimateResponse.model_validate(r.json())
    assert body.estimate_aed > 0
    assert body.range_80_aed.low <= body.estimate_aed <= body.range_80_aed.high
    assert body.community == "Business Bay"
    assert len(body.top_factors) == 5
    assert all(f.label and f.value for f in body.top_factors)
    assert body.community_median_per_sqm_aed is not None
    assert body.model.trained_on_months[-1] == "2026-03"


@pytest.mark.parametrize(("project", "variant"), [(None, "community"), ("Canal Heights", "full")])
def test_factors_multiply_up_to_the_estimate(
    client: TestClient, project: str | None, variant: str
) -> None:
    body = client.post("/estimate", json=VALID | {"project": project}).json()
    assert body["model_variant"] == variant
    per_sqm = body["base_per_sqm_aed"] * (1 + body["other_factors_effect_pct"] / 100)
    for f in body["top_factors"]:
        per_sqm *= 1 + f["effect_pct"] / 100
    # Rounding: base to 10 AED, effects to 0.1 percentage points, estimate to 10 AED.
    assert per_sqm == pytest.approx(body["estimate_per_sqm_aed"], rel=0.005)


def test_without_a_project_the_community_level_model_answers_and_says_so(
    client: TestClient,
) -> None:
    body = client.post("/estimate", json=VALID).json()
    assert body["model_variant"] == "community"
    assert body["project"] is None
    note = next(w for w in body["warnings"] if w.startswith("No project given"))
    assert "community-level model" in note
    assert "test month its median error was" in note
    assert all(f["feature"] != "project" for f in body["top_factors"])


def test_a_known_project_uses_the_full_model_without_a_warning(client: TestClient) -> None:
    body = client.post("/estimate", json=VALID | {"project": "canal heights"}).json()
    assert body["model_variant"] == "full"
    assert body["project"] == "Canal Heights"
    assert body["warnings"] == []


def test_projects_are_listed_per_community(client: TestClient) -> None:
    r = client.get("/projects", params={"community": "business bay"})
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert names == sorted(names)
    assert {"Canal Heights", "Bay Square", "Peninsula One"} <= set(names)
    assert "Marina Crest" not in names
    assert all(p["training_sales"] > 0 for p in r.json())
    unknown = client.get("/projects", params={"community": "Busines Bay"})
    assert unknown.status_code == 404
    assert "Business Bay" in unknown.json()["suggestions"]
    assert client.get("/projects").status_code == 422


def test_names_resolve_in_either_language_and_any_case(client: TestClient) -> None:
    maritime = client.post("/estimate", json=VALID | {"community": "Madinat Dubai Almelaheyah"})
    assert maritime.json()["community"] == "Dubai Maritime City"
    english = client.post("/estimate", json=VALID).json()
    for name in ("business  bay", "BUSINESS BAY", "الخليج التجارى"):
        other = client.post("/estimate", json=VALID | {"community": name}).json()
        assert other["community"] == "Business Bay"
        assert other["estimate_aed"] == english["estimate_aed"]


def test_bigger_flats_cost_more(client: TestClient) -> None:
    small = client.post("/estimate", json=VALID | {"size_sqm": 60}).json()["estimate_aed"]
    big = client.post("/estimate", json=VALID | {"size_sqm": 160, "rooms": "3"}).json()
    assert big["estimate_aed"] > small


def test_unknown_community_gets_suggestions(client: TestClient) -> None:
    r = client.post("/estimate", json=VALID | {"community": "Busines Bay"})
    assert r.status_code == 404
    assert "Business Bay" in r.json()["suggestions"]


@pytest.mark.parametrize(
    "change",
    [
        {"size_sqm": 5},
        {"size_sqm": 99999},
        {"rooms": "7"},
        {"rooms": -1},
        {"community": ""},
        {"community": "   "},
        {"community": " x "},
        {"sub_type": "Villa"},
        {"transaction_date": "next week"},
        {"transaction_date": "1500-01-01"},
        {"transaction_date": "9999-12-31"},
        {"unexpected": 1},
    ],
)
def test_invalid_input_is_rejected_with_422(client: TestClient, change: dict[str, Any]) -> None:
    assert client.post("/estimate", json=VALID | change).status_code == 422


@pytest.mark.parametrize("size", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_sizes_are_rejected_with_422(client: TestClient, size: str) -> None:
    # Python's JSON parser accepts these literals; the error body must still be valid JSON.
    body = json.dumps(VALID).replace('"size_sqm": 80', f'"size_sqm": {size}')
    r = client.post("/estimate", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail[0]["loc"] == ["body", "size_sqm"]
    assert "finite" in detail[0]["msg"]


def test_blank_project_counts_as_not_given(client: TestClient) -> None:
    r = client.post("/estimate", json=VALID | {"project": "   "})
    assert r.status_code == 200
    assert r.json()["project"] is None


def test_missing_required_field_is_rejected(client: TestClient) -> None:
    body = {k: v for k, v in VALID.items() if k != "off_plan"}
    assert client.post("/estimate", json=body).status_code == 422


def test_rooms_accepts_numbers(client: TestClient) -> None:
    assert client.post("/estimate", json=VALID | {"rooms": 0}).status_code == 200
    assert client.post("/estimate", json=VALID | {"rooms": 6}).status_code == 200


def test_warnings_explain_weak_inputs(client: TestClient) -> None:
    unknown = client.post("/estimate", json=VALID | {"project": "Nowhere Towers"}).json()
    assert unknown["model_variant"] == "community"
    assert any(
        "not in the training data, so the estimate comes from the community-level model" in w
        for w in unknown["warnings"]
    )
    elsewhere = client.post("/estimate", json=VALID | {"project": "Marina Crest"}).json()
    assert any("is recorded in Marsa Dubai" in w for w in elsewhere["warnings"])
    later = client.post("/estimate", json=VALID | {"transaction_date": "2027-06-01"}).json()
    assert any("after the newest training data" in w for w in later["warnings"])


def test_known_project_is_used(client: TestClient) -> None:
    r = client.post("/estimate", json=VALID | {"project": "canal heights"}).json()
    assert r["project"] == "Canal Heights"
    assert not any("not in the training data" in w for w in r["warnings"])


@pytest.mark.parametrize("with_project", [True, False])
def test_the_api_answers_what_the_evaluation_scores(
    client: TestClient, tiny_model: PriceModel, clean_fixture: pd.DataFrame, with_project: bool
) -> None:
    """Each test-month sale sent to the API gets the estimate the pipeline scored for it."""
    from dxb_prices import serving

    sales = clean_fixture[
        (clean_fixture["month"] == "2026-04")
        & clean_fixture["community"].isin(tiny_model.spec.community_rows)
        & (clean_fixture["rooms"] != "unknown")  # the API only takes known room counts
    ].head(60)
    rows = serving.model_rows(
        serving.requests_from_sales(sales, with_project=with_project), tiny_model.spec
    )
    expected = serving.estimate(tiny_model, rows)["estimate"]
    variants, estimates = rows["variant"].tolist(), expected.tolist()
    for sale, variant, estimate in zip(sales.to_dict("records"), variants, estimates, strict=True):
        project = sale["project"] if isinstance(sale["project"], str) else None
        body = {
            "community": sale["community"],
            "project": project if with_project else None,
            "size_sqm": float(sale["area_sqm"]),
            "rooms": sale["rooms"],
            "off_plan": bool(sale["is_off_plan"]),
            "sub_type": sale["sub_type"],
            "transaction_date": str(sale["transaction_date"].date()),
        }
        answer = client.post("/estimate", json=body).json()
        assert answer["model_variant"] == variant
        assert answer["estimate_aed"] == round(estimate, -3)


def test_thin_communities_are_flagged(tiny_model_dir: Path) -> None:
    model = PriceModel.load(tiny_model_dir)  # a private copy: the spec is edited below
    model.spec.community_rows["Business Bay"] = 3
    out = Estimator(model).estimate(EstimateRequest.model_validate(VALID))
    assert any("Only 3 training sales" in w for w in out.warnings)


def test_service_without_a_model_reports_unavailable(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "missing")) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert "dxb-prices train" in health.json()["detail"]
        assert c.post("/estimate", json=VALID).status_code == 503


def test_a_corrupt_model_file_leaves_the_service_unavailable(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    for f in tiny_model_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    (broken / "model.lgb").write_text("not a model", encoding="utf-8")
    with TestClient(create_app(broken)) as c:
        assert c.get("/health").status_code == 503
