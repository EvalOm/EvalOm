from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Sequence


def _normalize_confidence_level(value: Any) -> float:
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


def _observations(values: Sequence[Any]) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("observations must be a finite sequence of numbers.")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("observations must contain only finite numbers.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("observations must contain only finite numbers.")
        normalized.append(numeric)
    return normalized


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@dataclass(frozen=True)
class ContinuousStatisticalResult:
    count: int
    mean: float | None
    variance: float | None
    standard_deviation: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    confidence_level: float
    lower_bound: float | None
    upper_bound: float | None
    status: str
    method: str = "percentile_bootstrap"
    statistic: str = "mean"
    resamples: int = 0
    seed: int | None = 0
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "statistic": self.statistic,
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "status": self.status,
            "resamples": self.resamples,
            "seed": self.seed,
            "limitations": list(self.limitations),
        }


def bootstrap_mean_interval(
    observations: Sequence[Any],
    *,
    confidence_level: float = 0.95,
    resamples: int = 1000,
    seed: int | None = 0,
    minimum_sample_size: int | None = None,
) -> ContinuousStatisticalResult:
    """Return summary statistics and a deterministic percentile bootstrap CI.

    Resampling individual observations assumes they are independent and
    representative. It does not correct dependence, bias, or distribution
    shift. A singleton's point estimate is reported without a meaningful
    uncertainty interval.
    """
    values = _observations(observations)
    level = _normalize_confidence_level(confidence_level)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None.")
    if minimum_sample_size is not None and (
        isinstance(minimum_sample_size, bool)
        or not isinstance(minimum_sample_size, int)
        or minimum_sample_size < 0
    ):
        raise ValueError("minimum_sample_size must be a non-negative integer or None.")

    if not values:
        return ContinuousStatisticalResult(
            count=0, mean=None, variance=None, standard_deviation=None,
            median=None, minimum=None, maximum=None,
            confidence_level=level, lower_bound=None, upper_bound=None,
            status="insufficient_data", resamples=resamples, seed=seed,
            limitations=("No observations are available.",),
        )

    mean = statistics.fmean(values)
    median = statistics.median(values)
    variance = statistics.variance(values) if len(values) > 1 else None
    standard_deviation = math.sqrt(variance) if variance is not None else None
    limitations: list[str] = [
        "The percentile bootstrap assumes independent, representative observations.",
        "The interval does not correct sampling bias, correlation, or distribution shift.",
    ]
    status = "valid"
    if len(values) == 1:
        status = "insufficient_data"
        limitations.append("A singleton supports point estimates but not meaningful uncertainty estimation.")
        lower_bound = upper_bound = None
    else:
        rng = random.Random(seed)
        bootstrap_means = sorted(
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(resamples)
        )
        alpha = (1.0 - level) / 2.0
        lower_bound = _percentile(bootstrap_means, alpha)
        upper_bound = _percentile(bootstrap_means, 1.0 - alpha)
        if len(set(values)) == 1:
            limitations.append("A zero-width interval for identical observations does not prove zero population uncertainty.")
    if minimum_sample_size is not None and len(values) < minimum_sample_size:
        status = "insufficient_data"
        limitations.append(f"Sample count {len(values)} is below the configured minimum of {minimum_sample_size}.")
    return ContinuousStatisticalResult(
        count=len(values), mean=mean, variance=variance,
        standard_deviation=standard_deviation, median=median,
        minimum=min(values), maximum=max(values),
        confidence_level=level, lower_bound=lower_bound,
        upper_bound=upper_bound, status=status, resamples=resamples,
        seed=seed, limitations=tuple(limitations),
    )


__all__ = ["ContinuousStatisticalResult", "bootstrap_mean_interval"]
