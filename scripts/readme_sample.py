"""Regenerate the sample API responses in README.md from a trained model.

Sends one request with a project and the same request without it through the
real FastAPI app (in process, no server needed) and writes both into the block
between ``<!-- sample:start -->`` and ``<!-- sample:end -->``.

Usage: python scripts/readme_sample.py [MODEL_DIR]  (default: artifacts/model)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from dxb_prices import config
from dxb_prices.api.app import create_app

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- sample:start -->", "<!-- sample:end -->"

REQUEST: dict[str, Any] = {
    "community": "Business Bay",
    "project": "Peninsula Three",
    "size_sqm": 65,
    "rooms": "1",
    "off_plan": False,
    "transaction_date": "2026-08-15",
}


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def render(with_project: dict[str, Any], without: dict[str, Any]) -> str:
    request_without = {k: v for k, v in REQUEST.items() if k != "project"}
    return "\n".join(
        [
            START,
            "Request, `POST /estimate`:",
            "",
            "```json",
            json.dumps(REQUEST, ensure_ascii=False),
            "```",
            "",
            "Response:",
            "",
            "```json",
            _json(with_project),
            "```",
            "",
            "The same request without the project",
            f"(`{json.dumps(request_without, ensure_ascii=False)}`) is answered by the",
            f"community-level model: AED {without['estimate_aed']:,}, 80% range AED "
            f"{without['range_80_aed']['low']:,} to {without['range_80_aed']['high']:,}, "
            "with this warning:",
            "",
            *[f"> {w}" for w in without["warnings"]],
            END,
        ]
    )


def main(model_dir: Path) -> None:
    with TestClient(create_app(model_dir)) as client:
        responses = []
        for body in (REQUEST, {k: v for k, v in REQUEST.items() if k != "project"}):
            r = client.post("/estimate", json=body)
            r.raise_for_status()
            responses.append(r.json())
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"README.md has no {START} ... {END} block")
    block = render(*responses)
    README.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print(f"updated the sample block in {README}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else config.MODEL_DIR)
