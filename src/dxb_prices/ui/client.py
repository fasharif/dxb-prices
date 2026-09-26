"""Small HTTP client for the estimate API, used by the Streamlit page."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or []


@dataclass
class ApiClient:
    base_url: str = field(default_factory=lambda: os.environ.get("DXB_API_URL", DEFAULT_API_URL))
    timeout: float = 15.0
    transport: httpx.BaseTransport | None = None

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport)

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        try:
            with self._client() as c:
                r = c.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        if r.status_code != 200:
            raise ApiError(f"API returned HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def communities(self) -> list[dict[str, Any]]:
        data = self._get("/communities")
        if not isinstance(data, list):
            raise ApiError("unexpected /communities response")
        return data

    def projects(self, community: str) -> list[dict[str, Any]]:
        data = self._get("/projects", params={"community": community})
        if not isinstance(data, list):
            raise ApiError("unexpected /projects response")
        return data

    def model_info(self) -> dict[str, Any]:
        data = self._get("/model")
        if not isinstance(data, dict):
            raise ApiError("unexpected /model response")
        return data

    def estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._client() as c:
                r = c.post("/estimate", json=payload)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200:
            return dict(body)
        if r.status_code == 404:
            raise ApiError(str(body.get("detail", "not found")), body.get("suggestions", []))
        if r.status_code == 422:
            problems = body.get("detail", [])
            text = "; ".join(
                f"{'.'.join(str(x) for x in p.get('loc', [])[1:])}: {p.get('msg')}"
                for p in problems
                if isinstance(p, dict)
            )
            raise ApiError(f"invalid input: {text or r.text[:200]}")
        raise ApiError(f"API returned HTTP {r.status_code}: {r.text[:200]}")
