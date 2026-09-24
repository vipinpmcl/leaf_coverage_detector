from __future__ import annotations

import re
from pathlib import Path


_COVERAGE_RE = re.compile(
    r"leaf\s*coverage\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%",
    re.IGNORECASE,
)


def read_leaf_coverage(path: str | Path) -> float:
    """Read `Leaf Coverage: <number>%` from leaf_coverage.txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Leaf coverage file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    match = _COVERAGE_RE.search(text)
    if not match:
        raise ValueError(
            f"Could not parse leaf coverage from {path}. "
            "Expected e.g. 'Leaf Coverage: 3.4815%'"
        )

    value = float(match.group(1))
    if value < 0 or value > 100:
        raise ValueError(f"Leaf coverage must be in [0, 100], got {value}")
    return value


def classify_coverage(value: float, low_threshold: float, high_threshold: float) -> str:
    """Classify coverage using a gap-free boundary convention.

    < low  -> negative point
    [low, high] -> ignore
    > high and <= 100 -> SAM2 polygon annotation
    """
    if not (0 <= low_threshold < high_threshold <= 100):
        raise ValueError(
            "Coverage thresholds must satisfy 0 <= low_threshold < "
            "high_threshold <= 100."
        )

    if value < low_threshold:
        return "negative_point"
    if value <= high_threshold:
        return "ignore"
    return "sam2_polygon"
