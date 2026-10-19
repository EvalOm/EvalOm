from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


def _confidence_level(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence_level must be a finite number between 0 and 1, or a percentage between 1 and 100.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("confidence_level must be finite.")
    if 0.0 < numeric < 1.0:
        return numeric
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    raise ValueError("confidence_level must be in the open interval (0, 1) or a percentage between 1 and 100.")


@dataclass(frozen=True)
class BinaryStatisticalResult:
    """A deterministic Wilson score interval for independent binary outcomes."""

    total_observations: int
    successes: int
    failures: int
    observed_success_rate: float | None
    confidence_level: float
    lower_bound: float | None
    upper_bound: float | None
    status: str
    method: str = "wilson_score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total_observations": self.total_observations,
            "successes": self.successes,
            "failures": self.failures,
            "observed_success_rate": self.observed_success_rate,
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "status": self.status,
        }


def wilson_score_interval(
    successes: int,
    total_observations: int,
    *,
    confidence_level: float = 0.95,
) -> BinaryStatisticalResult:
    """Calculate a two-sided Wilson score interval for a binomial proportion.

    The interval describes sampling uncertainty under the assumption that
    observations are independent and representative. It does not establish
    correctness, authenticity, or evidence sufficiency.
    """
    if isinstance(total_observations, bool) or not isinstance(total_observations, int):
        raise ValueError("total_observations must be a non-negative integer.")
    if total_observations < 0:
        raise ValueError("total_observations must be a non-negative integer.")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer between 0 and total_observations.")
    if successes < 0 or successes > total_observations:
        raise ValueError("successes must be an integer between 0 and total_observations.")
    level = _confidence_level(confidence_level)
    if total_observations == 0:
        return BinaryStatisticalResult(
            total_observations=0,
            successes=0,
            failures=0,
            observed_success_rate=None,
            confidence_level=level,
            lower_bound=None,
            upper_bound=None,
            status="insufficient_data",
        )

    observed = successes / total_observations
    z_value = statistics.NormalDist().inv_cdf((1.0 + level) / 2.0)
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / total_observations
    center = (observed + z_squared / (2.0 * total_observations)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            observed * (1.0 - observed) / total_observations
            + z_squared / (4.0 * total_observations * total_observations)
        )
        / denominator
    )
    lower = max(0.0, min(observed, center - half_width))
    upper = min(1.0, max(observed, center + half_width))
    return BinaryStatisticalResult(
        total_observations=total_observations,
        successes=successes,
        failures=total_observations - successes,
        observed_success_rate=observed,
        confidence_level=level,
        lower_bound=lower,
        upper_bound=upper,
        status="valid",
    )


calculate_binary_statistics = wilson_score_interval
calculate_wilson_interval = wilson_score_interval


__all__ = [
    "BinaryStatisticalResult",
    "calculate_binary_statistics",
    "calculate_wilson_interval",
    "wilson_score_interval",
]
