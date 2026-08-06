"""Small shared helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def month_index(period: str) -> int:
    """'YYYY-MM' -> integer month count. Weeks ('YYYY-Www') map to fractional months."""
    if "W" in period:
        y, w = period.split("-W")
        return int(y) * 12 + (int(w) - 1) * 12 / 52.0
    y, m = period.split("-")[:2]
    return int(y) * 12 + int(m) - 1


def month_label(idx: float) -> str:
    i = int(round(idx))
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    a, b = month_index(start), month_index(end)
    return [month_label(i) for i in range(int(a), int(b) + 1)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fmt_int(n: float) -> str:
    return f"{int(n):,}"
