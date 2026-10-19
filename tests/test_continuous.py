import math

import pytest

from evalom.continuous import bootstrap_mean_interval


def test_summary_statistics_and_reproducible_bootstrap():
    first = bootstrap_mean_interval([1, 2, 3, 4, 5], resamples=500, seed=7)
    second = bootstrap_mean_interval([1, 2, 3, 4, 5], resamples=500, seed=7)

    assert first == second
    assert first.count == 5
    assert first.mean == pytest.approx(3.0)
    assert first.median == pytest.approx(3.0)
    assert first.variance == pytest.approx(2.5)
    assert first.standard_deviation == pytest.approx(math.sqrt(2.5))
    assert first.minimum == 1
    assert first.maximum == 5
    assert first.lower_bound <= first.upper_bound
    assert first.status == "valid"


def test_empty_and_singleton_samples_do_not_fabricate_uncertainty():
    empty = bootstrap_mean_interval([], resamples=10)
    singleton = bootstrap_mean_interval([4.5], resamples=10)

    assert empty.status == "insufficient_data"
    assert empty.mean is None
    assert empty.lower_bound is None
    assert singleton.mean == 4.5
    assert singleton.median == 4.5
    assert singleton.variance is None
    assert singleton.lower_bound is None
    assert singleton.status == "insufficient_data"


def test_degenerate_and_minimum_sample_results_are_explicit():
    result = bootstrap_mean_interval([2, 2, 2], resamples=20, seed=1, minimum_sample_size=5)

    assert result.mean == 2
    assert result.lower_bound == 2
    assert result.upper_bound == 2
    assert result.status == "insufficient_data"
    assert any("zero-width" in limitation for limitation in result.limitations)
    assert any("minimum" in limitation for limitation in result.limitations)


def test_confidence_level_and_resample_validation():
    narrower = bootstrap_mean_interval([1, 2, 3, 4, 5], confidence_level=0.90, resamples=100, seed=0)
    wider = bootstrap_mean_interval([1, 2, 3, 4, 5], confidence_level=0.99, resamples=100, seed=0)
    assert wider.lower_bound <= narrower.lower_bound
    assert wider.upper_bound >= narrower.upper_bound
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, 2], confidence_level=1)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, 2], resamples=0)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, True])
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, math.nan])
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, math.inf])


def test_negative_and_large_finite_values_are_supported():
    result = bootstrap_mean_interval([-1e100, 1e100], resamples=50, seed=2)
    assert math.isfinite(result.mean)
    assert math.isfinite(result.lower_bound)
    assert math.isfinite(result.upper_bound)
