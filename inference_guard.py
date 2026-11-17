"""Input-domain checks shared by notebook and CLI inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainViolation:
    feature: str
    observed_min: float
    observed_max: float
    allowed_min: float
    allowed_max: float
    outside_rows: int
    total_rows: int


def check_feature_domain(
    frame: pd.DataFrame,
    feature_cols: list[str],
    feature_min: dict[str, float],
    feature_max: dict[str, float],
) -> list[DomainViolation]:
    """Return features whose observed values leave the training min/max range."""
    if len(feature_min) != len(feature_cols) or len(feature_max) != len(feature_cols):
        raise ValueError("feature range metadata does not match feature_cols")

    violations: list[DomainViolation] = []
    for name in feature_cols:
        values = frame[name].to_numpy(dtype=float)
        lo = float(feature_min[name])
        hi = float(feature_max[name])
        outside = (values < lo) | (values > hi)
        if outside.any():
            violations.append(
                DomainViolation(
                    feature=name,
                    observed_min=float(np.nanmin(values)),
                    observed_max=float(np.nanmax(values)),
                    allowed_min=lo,
                    allowed_max=hi,
                    outside_rows=int(outside.sum()),
                    total_rows=len(values),
                )
            )
    return violations


def format_domain_violations(violations: list[DomainViolation]) -> str:
    if not violations:
        return "input domain: OK"
    lines = ["input domain: OUT OF TRAINING RANGE"]
    for v in violations:
        lines.append(
            f"  {v.feature}: observed [{v.observed_min:.3f}, {v.observed_max:.3f}], "
            f"train [{v.allowed_min:.3f}, {v.allowed_max:.3f}], "
            f"{v.outside_rows}/{v.total_rows} rows outside"
        )
    return "\n".join(lines)
