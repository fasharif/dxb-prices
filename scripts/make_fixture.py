"""Regenerate tests/fixtures/transactions_synthetic.csv (synthetic data, fixed seed)."""

from __future__ import annotations

from pathlib import Path

from dxb_prices import fixture

TARGET = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "transactions_synthetic.csv"


def main() -> None:
    rows = fixture.write(TARGET)
    print(f"wrote {rows} synthetic rows to {TARGET}")


if __name__ == "__main__":
    main()
