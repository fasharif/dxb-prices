from __future__ import annotations

from pathlib import Path

from dxb_prices import fixture
from tests.conftest import FIXTURE_CSV


def test_generator_reproduces_the_committed_fixture(tmp_path: Path) -> None:
    out = tmp_path / "fixture.csv"
    rows = fixture.write(out)
    assert rows == 613
    assert out.read_bytes() == FIXTURE_CSV.read_bytes()


def test_generator_is_deterministic_and_seed_dependent() -> None:
    assert fixture.generate(seed=7) == fixture.generate(seed=7)
    assert fixture.generate(seed=7) != fixture.generate(seed=8)
