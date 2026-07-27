"""config#2889 — cpcv_meta_oos_ic's row_ids/return_preds capture.

The independent second-opinion IC (training/realized_ic_second_opinion.py)
needs the CPCV pipeline's own held-out test predictions PLUS an identity
(ticker) for each row so they can be rejoined against
score_performance_outcomes independently of meta_y. This pins that capture:
predictions/true/dates/ids stack across every test combo, are poppable
(NOT part of the JSON-serializable summary), and the existing
status/mean_ic/... contract is unchanged when return_preds is False.
"""
from __future__ import annotations

import numpy as np

from training.leakfree_meta_ic import cpcv_meta_oos_ic


def _linear_fit_predict(X_tr, y_tr, X_te):
    """Deterministic fit_predict_fn: OLS-ish scalar-feature linear fit."""
    X_tr = np.asarray(X_tr, dtype=float).ravel()
    y_tr = np.asarray(y_tr, dtype=float).ravel()
    X_te = np.asarray(X_te, dtype=float).ravel()
    if np.std(X_tr) < 1e-12:
        return np.zeros_like(X_te)
    slope = np.cov(X_tr, y_tr)[0, 1] / np.var(X_tr)
    intercept = y_tr.mean() - slope * X_tr.mean()
    return slope * X_te + intercept


def _make_dataset(n_dates=60, n_per_date=8, seed=7):
    rng = np.random.default_rng(seed)
    dates = []
    tickers = []
    X = []
    y = []
    for d in range(n_dates):
        for i in range(n_per_date):
            dates.append(f"2026-01-{d:03d}")
            tickers.append(f"T{i}")
            x = rng.normal()
            X.append([x])
            y.append(x * 0.5 + rng.normal(scale=0.1))
    return np.asarray(X), np.asarray(y), np.asarray(dates), np.asarray(tickers)


def test_return_preds_false_omits_oos_keys_default_behavior_unchanged():
    X, y, dates, tickers = _make_dataset()
    result = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
    )
    assert result["status"] == "ok"
    assert "_oos_preds" not in result
    assert "_oos_ids" not in result


def test_return_preds_true_stacks_preds_true_dates_ids():
    X, y, dates, tickers = _make_dataset()
    result = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
        row_ids=tickers, return_preds=True,
    )
    assert result["status"] == "ok"
    assert isinstance(result["_oos_preds"], np.ndarray)
    assert isinstance(result["_oos_true"], np.ndarray)
    assert isinstance(result["_oos_dates"], np.ndarray)
    assert isinstance(result["_oos_ids"], np.ndarray)
    n = len(result["_oos_preds"])
    assert n > 0
    assert len(result["_oos_true"]) == n
    assert len(result["_oos_dates"]) == n
    assert len(result["_oos_ids"]) == n
    # every stacked id is one of the known tickers
    assert set(result["_oos_ids"].tolist()) <= set(tickers.tolist())


def test_return_preds_true_without_row_ids_ids_is_none():
    X, y, dates, tickers = _make_dataset()
    result = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
        return_preds=True,
    )
    assert result["status"] == "ok"
    assert result["_oos_ids"] is None


def test_return_preds_true_on_insufficient_groups_returns_empty_arrays():
    # n_groups=6 needs >= k_test+1=3 non-empty contiguous groups; 2 unique
    # dates can't split into 3, so this trips the insufficient_groups
    # early-return path before the combo loop ever runs.
    X, y, dates, tickers = _make_dataset(n_dates=2, n_per_date=4)
    result = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
        row_ids=tickers, return_preds=True,
    )
    assert result["status"] == "insufficient_groups"
    assert result["_oos_preds"].tolist() == []
    assert result["_oos_ids"].tolist() == []


def test_summary_stats_unaffected_by_return_preds_flag():
    """The headline mean_ic/ics distribution must be identical whether or
    not return_preds is requested — return_preds is purely additive."""
    X, y, dates, tickers = _make_dataset()
    without = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
    )
    with_preds = cpcv_meta_oos_ic(
        X, y, dates, fit_predict_fn=_linear_fit_predict,
        forward_days=1, embargo_days=0, n_groups=6, k_test=2,
        row_ids=tickers, return_preds=True,
    )
    assert without["mean_ic"] == with_preds["mean_ic"]
    assert without["ics"] == with_preds["ics"]
