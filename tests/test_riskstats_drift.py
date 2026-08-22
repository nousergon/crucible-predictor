"""Drift guard: this repo's downside-deviation path vs nousergon_lib (config-I7597).

`training/deflated_sharpe.py::downside_ic_stats` used to re-derive the Sortino
denominator. It now calls `nousergon_lib.quant.riskstats.downside_deviation`,
so it cannot drift — this file is what proves that stays true, and pins the
IC-space conventions the library deliberately does not decide (numerator is
`mean_ic`, the near-zero-dd sentinel, the n < 5 floor).

CORPUS is kept byte-identical to
`nousergon-lib/tests/test_quant_riskstats_drift_corpus.py`, which pins the
library's own answers against values written out from the definition.

Collected by plain `pytest` — deliberately NOT part of
`training/self_test_cases.py`'s battery, which only runs inside a training run.
"""

from __future__ import annotations

import math

import pytest
from nousergon_lib.quant import riskstats

from training.deflated_sharpe import downside_ic_stats

# Keep byte-identical to the nousergon-lib copy.
CORPUS: dict[str, list[float]] = {
    "mixed": [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.0, 0.02, -0.03, 0.005],
    "all_positive": [0.01, 0.02, 0.005, 0.03, 0.015],
    "all_negative": [-0.01, -0.02, -0.005, -0.04],
    "all_zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "zero_vol_positive": [0.01] * 8,
    "zero_vol_negative": [-0.01] * 8,
    "two_obs": [0.01, -0.01],
    "single_obs": [0.02],
    "empty": [],
    "tiny_downside": [0.01, 0.02, 0.03, -1e-9],
}

# downside_ic_stats declines n < 5 as "insufficient".
_SCORABLE = [n for n, v in CORPUS.items() if len(v) >= 5]


def _ref_dd_full(r: list[float], target: float = 0.0) -> float:
    """n-denominator downside deviation, from the definition — no lib call."""
    return math.sqrt(sum(min(0.0, x - target) ** 2 for x in r) / len(r))


@pytest.mark.parametrize("name", sorted(_SCORABLE))
def test_downside_deviation_matches_the_definition(name: str) -> None:
    r = CORPUS[name]
    out = downside_ic_stats(r)
    assert out["status"] == "ok", name
    assert out["downside_deviation"] == pytest.approx(
        round(_ref_dd_full(r), 6), rel=1e-9, abs=1e-9
    ), name


@pytest.mark.parametrize("name", sorted(_SCORABLE))
def test_downside_deviation_matches_the_library(name: str) -> None:
    r = CORPUS[name]
    out = downside_ic_stats(r)
    want = riskstats.downside_deviation(r, target=0.0, denominator="full")
    assert want is not None, name
    assert out["downside_deviation"] == pytest.approx(round(want, 6), rel=1e-9, abs=1e-9), name


def test_it_is_the_n_denominator_not_the_n_down_denominator() -> None:
    """config-I7271's convention. The two differ by sqrt(n / n_down).

    The retired side is written out from the definition, NOT asked of the
    library: config-I7638 deleted the ``"downside"`` branch from
    ``nousergon_lib.quant.riskstats`` (a call now raises ValueError), and a
    drift test whose verdict depends on which library version happens to be
    installed is the drift this file exists to catch.
    """
    r = CORPUS["mixed"]
    n, n_down = len(r), sum(1 for x in r if x < 0)
    out = downside_ic_stats(r)
    n_down_variant = math.sqrt(sum(x * x for x in r if x < 0.0) / n_down)
    assert n_down_variant is not None
    # rel=1e-4: `downside_deviation` is emitted rounded to 6 decimal places,
    # so the ratio of an unrounded value to a rounded one carries that much.
    assert n_down_variant / out["downside_deviation"] == pytest.approx(
        math.sqrt(n / n_down), rel=1e-4
    )


@pytest.mark.parametrize("name", sorted(_SCORABLE))
def test_sortino_of_ic_is_mean_over_the_library_deviation(name: str) -> None:
    r = CORPUS[name]
    out = downside_ic_stats(r)
    dd = riskstats.downside_deviation(r, denominator="full")
    assert dd is not None
    if dd <= 1e-12:
        # No bad-ranking day: undefined, reported as None + n_downside_days=0,
        # never as a measured zero.
        assert out["sortino_of_ic"] is None, name
        assert out["n_downside_days"] == 0, name
    else:
        assert out["sortino_of_ic"] == pytest.approx(
            round(sum(r) / len(r) / dd, 6), rel=1e-6, abs=1e-9
        ), name


def test_degenerate_series_are_declined_not_scored() -> None:
    for name in ("two_obs", "single_obs", "empty"):
        out = downside_ic_stats(CORPUS[name])
        assert out["status"] == "insufficient", name
        assert out["sortino_of_ic"] is None, name


def test_no_downside_days_reports_undefined_not_zero() -> None:
    out = downside_ic_stats(CORPUS["all_positive"])
    assert out["status"] == "ok"
    assert out["n_downside_days"] == 0
    assert out["downside_deviation"] == 0.0
    assert out["sortino_of_ic"] is None
