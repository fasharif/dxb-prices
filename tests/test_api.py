from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dxb_prices.api.app import create_app
from dxb_prices.api.estimator import Estimator, factor_value
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
        {"sub_type": "Villa"},
        {"transaction_date": "next week"},
        {"unexpected": 1},
    ],
)
def test_invalid_input_is_rejected_with_422(client: TestClient, change: dict[str, Any]) -> None:
    assert client.post("/estimate", json=VALID | change).status_code == 422


def test_missing_required_field_is_rejected(client: TestClient) -> None:
    body = {k: v for k, v in VALID.items() if k != "off_plan"}
    assert client.post("/estimate", json=body).status_code == 422


def test_rooms_accepts_numbers(client: TestClient) -> None:
    assert client.post("/estimate", json=VALID | {"rooms": 0}).status_code == 200
    assert client.post("/estimate", json=VALID | {"rooms": 6}).status_code == 200


def test_project_factor_wording() -> None:
    unspecified = "__unspecified__"
    assert factor_value("project", "x", unspecified, None).startswith("not given")
    assert factor_value("project", "x", unspecified, "Nowhere").startswith(
        "'Nowhere' not recognised"
    )
    assert (
        factor_value("project", "Canal Heights", "Canal Heights", "canal heights")
        == "Canal Heights"
    )
    assert factor_value("rooms", "1", unspecified, None) == "1"


def test_warnings_explain_weak_inputs(client: TestClient) -> None:
    unknown = client.post("/estimate", json=VALID | {"project": "Nowhere Towers"}).json()
    assert any("not in the training data" in w for w in unknown["warnings"])
    elsewhere = client.post("/estimate", json=VALID | {"project": "Marina Crest"}).json()
    assert any("is recorded in Marsa Dubai" in w for w in elsewhere["warnings"])
    later = client.post("/estimate", json=VALID | {"transaction_date": "2027-06-01"}).json()
    assert any("after the newest training data" in w for w in later["warnings"])


def test_known_project_is_used(client: TestClient) -> None:
    r = client.post("/estimate", json=VALID | {"project": "canal heights"}).json()
    assert r["project"] == "Canal Heights"
    assert not any("not in the training data" in w for w in r["warnings"])


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
