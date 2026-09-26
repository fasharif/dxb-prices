"""The Streamlit page and its API client.

The page test starts the real API (tiny fixture model) on a local port and
drives the page with Streamlit's AppTest, so the whole path UI -> HTTP ->
model is exercised.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from dxb_prices.api.app import create_app
from dxb_prices.ui.client import ApiClient, ApiError

PAGE = Path(__file__).resolve().parents[1] / "src" / "dxb_prices" / "ui" / "streamlit_app.py"


def mock_client(handler: object) -> ApiClient:
    return ApiClient(base_url="http://api.test", transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_client_returns_estimates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/estimate"
        return httpx.Response(200, json={"estimate_aed": 1_000_000})

    assert mock_client(handler).estimate({"community": "x"})["estimate_aed"] == 1_000_000


def test_client_passes_on_suggestions_for_unknown_communities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "detail": "Community 'x' is not in the training data.",
                "suggestions": ["Business Bay"],
            },
        )

    with pytest.raises(ApiError) as info:
        mock_client(handler).estimate({"community": "x"})
    assert info.value.suggestions == ["Business Bay"]


def test_client_summarises_validation_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "size_sqm"],
                        "msg": "Input should be greater than or equal to 18",
                    }
                ]
            },
        )

    with pytest.raises(ApiError, match="size_sqm: Input should be greater"):
        mock_client(handler).estimate({})


def test_client_asks_for_a_communitys_projects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/projects"
        assert request.url.params["community"] == "Business Bay"
        return httpx.Response(200, json=[{"name": "Canal Heights", "training_sales": 40}])

    assert mock_client(handler).projects("Business Bay")[0]["name"] == "Canal Heights"


def test_client_reports_an_unreachable_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="cannot reach the API"):
        mock_client(handler).communities()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_api(tiny_model_dir: Path) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(tiny_model_dir), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("API did not start")
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def test_page_shows_an_estimate_from_the_api(
    live_api: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DXB_API_URL", live_api)
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.run()
    assert not at.exception
    community, project = at.selectbox[0], at.selectbox[1]
    assert "Business Bay" in community.options
    assert "Canal Heights" in project.options
    community.select("Marsa Dubai")
    at.run()
    # The project list follows the community.
    project = at.selectbox[1]
    assert project.options[0] == "Not given"
    assert "Marina Crest" in project.options and "Canal Heights" not in project.options
    project.select("Marina Crest")
    at.number_input[0].set_value(90.0)
    at.button[0].click()
    at.run()
    assert not at.exception
    assert at.metric[0].label == "Estimated price"
    assert at.metric[0].value.startswith("AED ")
    assert len(at.table) == 1
    assert "the model that knows the project" in at.caption[-1].value


def test_page_says_when_the_community_level_model_answers(
    live_api: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DXB_API_URL", live_api)
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("No project given" in w.value for w in at.warning)
    assert "community-level model" in at.caption[-1].value


def test_page_explains_when_the_api_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DXB_API_URL", f"http://127.0.0.1:{_free_port()}")
    at = AppTest.from_file(str(PAGE), default_timeout=30)
    at.run()
    assert at.error and "cannot reach the API" in at.error[0].value
