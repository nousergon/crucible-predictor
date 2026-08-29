"""Regression tests for the purged, embargoed L1 early-stop split (I9333).

Verified RED before the fix. ``training/purged_split`` did not exist, so the
whole file fails to import against the pre-fix tree — and, more usefully, every
property test here is also asserted against ``_legacy_tail_split``, an exact
reproduction of the split that shipped:

    split_idx = int(n_finite * 0.8)
    fit(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

so reinstating it fails these tests rather than quietly restoring the defect.

The panel fixtures are built to the MEASURED geometry, not to round numbers:

* the research panel is ~899 rows per date after the 2026-07-18 research
  universe expansion (27 tickers -> 902, measured over every
  ``s3://alpha-engine-research/signals/<date>/signals.json``), against ~22-26
  rows per date before it — a ~34x imbalance;
* ``kept`` over three consecutive weekly runs was 2410 -> 6006 -> 10465, so the
  80/20 tail cut validation blocks of 386 / 961 / 1675 rows = ~15 sparse-era
  dates / **1.07** dates / **1.87** dates;
* the volatility panel is ~167 trading dates at ~898 rows each.
"""
from __future__ import annotations

import pytest

from training.purged_split import (
    MAX_VAL_DATE_WEIGHT,
    MAX_VAL_ROW_FRACTION,
    MIN_TRAIN_DATE_MULTIPLE,
    MIN_VAL_DATES,
    MIN_VARYING_FEATURES,
    DegenerateDesignMatrixError,
    assert_design_matrix_supports_fit,
    build_purged_split,
    describe_design_matrix,
    val_ic_precision,
)

RESEARCH_HORIZON = 10          # `actual_fwd_10d`
VOL_HORIZON = 21               # cfg.FORWARD_DAYS


def _dates(n: int, start: int = 0) -> list[str]:
    """``n`` ordered pseudo-trading-date keys, lexicographically sortable."""
    return [f"2026-01-{i:04d}" for i in range(start, start + n)]


def _panel(spec: "list[tuple[str, int]]") -> list[str]:
    """Flatten ``[(date, rows_on_that_date), ...]`` into a row-ordered panel."""
    out: list[str] = []
    for d, k in spec:
        out.extend([d] * k)
    return out


def _measured_research_panel(n_dense_dates: int = 8) -> list[str]:
    """The measured 2026-08-28 shape: a long sparse era, a short dense one."""
    ds = _dates(93 + n_dense_dates)
    return _panel(
        [(d, 22) for d in ds[:93]] + [(d, 899) for d in ds[93:]]
    )


def _measured_volatility_panel() -> list[str]:
    return _panel([(d, 898) for d in _dates(167)])


def _legacy_tail_split(panel: list[str], train_frac: float = 0.80):
    """The split that shipped: a bare row-index cut. Returns (train, val)."""
    n = len(panel)
    cut = int(n * train_frac)
    return panel[:cut], panel[cut:]


# ── The defect this replaces, pinned ─────────────────────────────────────────

def test_legacy_tail_split_validation_block_is_one_cross_section():
    """The pre-fix split's val block spans ~1 date on the measured panel."""
    panel = _measured_research_panel()
    _, val = _legacy_tail_split(panel)
    assert len(set(val)) < MIN_VAL_DATES, (
        "the legacy 80/20 row cut is expected to span fewer than "
        f"{MIN_VAL_DATES} dates on the measured panel; it spanned "
        f"{len(set(val))}"
    )


def test_legacy_tail_split_has_no_purge():
    """Train's last date is adjacent to val's first — the leak, pinned."""
    panel = _measured_research_panel()
    train, val = _legacy_tail_split(panel)
    uniq = sorted(set(panel))
    gap = uniq.index(val[0]) - uniq.index(train[-1])
    assert gap <= 1, (
        "the legacy split leaves no purge between train and validation; "
        f"measured gap {gap} date(s)"
    )


# ── Purge and embargo ────────────────────────────────────────────────────────

def test_purge_and_embargo_separate_train_from_validation():
    panel = _measured_volatility_panel()
    split = build_purged_split(
        panel, label_horizon_days=VOL_HORIZON, train_frac=0.70,
        val_frac=0.15, test_gap=False,
    )
    assert split.ok, split.reason
    uniq = sorted(set(panel))
    last_train_date = panel[int(split.train_idx[-1])]
    first_val_date = panel[int(split.val_idx[0])]
    gap = uniq.index(first_val_date) - uniq.index(last_train_date)
    assert gap >= VOL_HORIZON + VOL_HORIZON, (
        "every training row's forward label window must close before the "
        f"validation block opens, plus the embargo; measured gap {gap} "
        f"date(s), required >= {VOL_HORIZON * 2}"
    )
    assert split.n_purged_rows > 0 and split.n_embargoed_rows > 0
    assert split.embargo_days == VOL_HORIZON


def test_no_training_row_label_window_reaches_the_validation_block():
    """The property, stated on rows rather than on the summary counters."""
    panel = _measured_volatility_panel()
    split = build_purged_split(
        panel, label_horizon_days=VOL_HORIZON, train_frac=0.70,
        val_frac=0.15, test_gap=False,
    )
    uniq = sorted(set(panel))
    pos = {d: i for i, d in enumerate(uniq)}
    val_open = pos[panel[int(split.val_idx[0])]]
    for i in split.train_idx:
        assert pos[panel[int(i)]] + VOL_HORIZON < val_open


def test_explicit_embargo_overrides_the_auto_default():
    panel = _measured_volatility_panel()
    split = build_purged_split(
        panel, label_horizon_days=VOL_HORIZON, train_frac=0.70,
        val_frac=0.15, embargo_days=0, test_gap=False,
    )
    assert split.ok and split.embargo_days == 0
    assert split.n_embargoed_rows == 0


def test_negative_gap_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        build_purged_split(
            _measured_volatility_panel(), label_horizon_days=VOL_HORIZON,
            train_frac=0.70, embargo_days=-1,
        )


# ── Multi-date validation ────────────────────────────────────────────────────

def test_validation_block_spans_at_least_the_date_floor():
    panel = _measured_volatility_panel()
    split = build_purged_split(
        panel, label_horizon_days=VOL_HORIZON, train_frac=0.70,
        val_frac=0.15, test_gap=False,
    )
    assert split.ok, split.reason
    assert split.n_val_dates >= MIN_VAL_DATES
    assert split.max_val_date_weight <= MAX_VAL_DATE_WEIGHT


def test_val_block_is_grown_to_the_date_floor_when_the_fraction_is_short():
    """A 5%% val fraction over equal dates still gets MIN_VAL_DATES dates."""
    panel = _panel([(d, 100) for d in _dates(200)])
    split = build_purged_split(
        panel, label_horizon_days=5, train_frac=0.95, val_frac=None,
    )
    assert split.ok, split.reason
    assert split.n_val_dates >= MIN_VAL_DATES


def test_cuts_land_on_date_boundaries_never_inside_a_cross_section():
    panel = _panel([(d, 100) for d in _dates(200)])
    split = build_purged_split(
        panel, label_horizon_days=5, train_frac=0.70, val_frac=0.15,
        test_gap=False,
    )
    assert split.ok, split.reason
    for block in (split.train_idx, split.val_idx, split.test_idx):
        if len(block) == 0:
            continue
        block_dates = {panel[int(i)] for i in block}
        for d in block_dates:
            assert panel.count(d) == sum(
                1 for i in block if panel[int(i)] == d
            ), f"date {d} is split across a block boundary"


# ── Insufficient, and what it does ───────────────────────────────────────────

def test_measured_research_panel_reports_insufficient_not_a_one_date_holdout():
    """The measured 2026-08-28 geometry cannot support a valid split.

    This is the true finding, not a failure: 90%% of the panel's rows sit in
    its last eight dates, so a validation block wide enough to be a statistic
    leaves nothing to train on. Reporting that is required; fitting on a
    one-date holdout instead is the coin flip.
    """
    split = build_purged_split(
        _measured_research_panel(), label_horizon_days=RESEARCH_HORIZON,
        train_frac=0.80, val_frac=None,
    )
    assert split.status == "insufficient"
    assert "required <= 0.4" in split.reason
    assert split.val_row_fraction > MAX_VAL_ROW_FRACTION
    # No partial split is ever handed back.
    assert len(split.train_idx) == 0 and len(split.val_idx) == 0


def test_insufficient_when_the_panel_has_too_few_dates():
    split = build_purged_split(
        _panel([(d, 100) for d in _dates(6)]),
        label_horizon_days=RESEARCH_HORIZON, train_frac=0.80,
    )
    assert split.status == "insufficient"
    assert f"required >= {MIN_VAL_DATES}" in split.reason


def test_insufficient_when_the_purge_starves_the_training_block():
    split = build_purged_split(
        _panel([(d, 400) for d in _dates(40)]),
        label_horizon_days=RESEARCH_HORIZON, train_frac=0.80,
    )
    assert split.status == "insufficient"
    assert str(MIN_TRAIN_DATE_MULTIPLE * RESEARCH_HORIZON) in split.reason


def test_one_dominant_date_is_refused_even_at_the_date_floor():
    """A date COUNT does not bound a single cross-section's leverage."""
    ds = _dates(200)
    panel = _panel(
        [(d, 400) for d in ds[:190]]
        + [(d, 5) for d in ds[190:199]] + [(ds[199], 40000)]
    )
    split = build_purged_split(
        panel, label_horizon_days=RESEARCH_HORIZON, train_frac=0.80,
    )
    assert split.status == "insufficient"
    assert "of the validation rows" in split.reason


def test_empty_panel_reports_insufficient():
    split = build_purged_split([], label_horizon_days=10, train_frac=0.8)
    assert split.status == "insufficient"


def test_unsorted_panel_raises_rather_than_returning_a_wrong_split():
    with pytest.raises(ValueError, match="non-decreasing date order"):
        build_purged_split(
            ["2026-01-02"] * 10 + ["2026-01-01"] * 10,
            label_horizon_days=10, train_frac=0.8,
        )


# ── The manifest block ───────────────────────────────────────────────────────

def test_manifest_block_carries_every_property_the_issue_requires():
    split = build_purged_split(
        _measured_volatility_panel(), label_horizon_days=VOL_HORIZON,
        train_frac=0.70, val_frac=0.15, test_gap=False,
    )
    block = split.as_manifest_block()
    for key in ("n_train_dates", "n_val_dates", "n_purged_rows",
                "embargo_days", "status", "max_val_date_weight",
                "val_row_fraction", "embargo_edges", "floors"):
        assert key in block, key
    assert block["embargo_edges"] == ["leading"]
    assert block["floors"]["min_val_dates"] == MIN_VAL_DATES


def test_two_sided_gap_records_both_edges():
    split = build_purged_split(
        _panel([(d, 100) for d in _dates(400)]), label_horizon_days=5,
        train_frac=0.60, val_frac=0.20, test_gap=True,
    )
    assert split.ok, split.reason
    assert split.as_manifest_block()["embargo_edges"] == ["leading", "trailing"]
    assert split.n_test_dates > 0


# ── The degenerate design matrix ─────────────────────────────────────────────

FEATURES = [
    "research_composite_score", "research_conviction", "sector_macro_modifier",
    "macro_spy_20d_return", "macro_spy_20d_vol", "macro_vix_level",
    "macro_vix_term_slope", "macro_yield_curve_slope", "macro_market_breadth",
]


def _matrix(n_varying: int, n_rows: int = 400):
    import numpy as np

    X = np.zeros((n_rows, len(FEATURES)), dtype=float)
    for j in range(n_varying):
        X[:, j] = np.linspace(0.0, 100.0, n_rows)
    return X


def test_the_measured_2026_08_28_matrix_is_refused():
    """One varying column of nine — the measured post-2026-07-18 state."""
    with pytest.raises(DegenerateDesignMatrixError) as exc:
        assert_design_matrix_supports_fit("research_gbm", _matrix(1), FEATURES)
    msg = str(exc.value)
    assert "1 varying column(s) of 9" in msg
    # The constant columns are NAMED — that is the deliverable.
    assert "research_conviction" in msg
    assert "sector_macro_modifier" in msg
    assert "macro_vix_level" in msg
    assert "-I9307" in msg


def test_a_matrix_at_the_floor_is_accepted():
    report = assert_design_matrix_supports_fit(
        "research_gbm", _matrix(MIN_VARYING_FEATURES), FEATURES,
    )
    assert report["n_varying"] == MIN_VARYING_FEATURES
    assert report["min_varying_features"] == MIN_VARYING_FEATURES


def test_describe_design_matrix_partitions_every_declared_column():
    report = describe_design_matrix(_matrix(3), FEATURES)
    assert report["n_varying"] + report["n_constant"] == len(FEATURES)
    assert set(report["varying_columns"]) == set(FEATURES[:3])


# ── val_ic precision ─────────────────────────────────────────────────────────

def test_val_ic_precision_discounts_overlapping_labels():
    import numpy as np

    rng = np.random.default_rng(0)
    dates = [d for d in _dates(40) for _ in range(50)]
    preds = rng.normal(size=len(dates))
    y = rng.normal(size=len(dates))
    out = val_ic_precision(preds, y, dates, label_horizon_days=10)
    assert out["n_val_dates_scored"] == 40
    # 40 contiguous dates against a 10-day overlapping label is 4 independent
    # observations, not 40.
    assert out["n_eff"] == pytest.approx(4.0)
    assert out["val_ic_se"] > out["per_date_ic_std"] / 40 ** 0.5


def test_val_ic_precision_reports_nothing_rather_than_a_number_it_lacks():
    out = val_ic_precision([], [], [], label_horizon_days=10)
    assert out["n_val_dates_scored"] == 0
    assert out["val_ic_se"] is None


# ── Composition with the I9271 fit-validity gate ─────────────────────────────

def test_refusal_reason_reaches_the_gate_finding():
    """A refused fit must not degrade to a bare "absent" in the verdict."""
    from training.l1_fit_validity import assert_l1_fits_valid, evaluate_l1_fits

    block = evaluate_l1_fits({
        "research_gbm": {
            "fitted": False,
            "reason": (
                "DegenerateDesignMatrixError: research_gbm: refusing to fit — "
                "the design matrix has 1 varying column(s) of 9 declared"
            ),
            "split": {"status": "insufficient", "n_val_dates": 1},
            "design_matrix": {"n_varying": 1, "n_constant": 8},
        },
    })
    verdict = block["arms"]["research_gbm"]
    assert verdict["status"] == "not_fitted"
    assert "1 varying column(s) of 9" in verdict["reason"]
    assert verdict["split"]["status"] == "insufficient"
    assert verdict["design_matrix"]["n_varying"] == 1
    assert block["status"] == "failed"
    with pytest.raises(Exception) as exc:
        assert_l1_fits_valid(block)
    assert "1 varying column(s) of 9" in str(exc.value)


def test_meta_trainer_no_longer_cuts_an_l1_split_on_row_indices():
    """The exact expressions that shipped, pinned out of the source.

    A source guard rather than a behavioural one because reinstating either
    expression is a one-line edit that every behavioural test would still pass
    — the defect was never a crash, it was a well-formed model.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "training" / "meta_trainer.py"
    text = src.read_text()
    # Comments quote the legacy expressions deliberately, so only executable
    # lines are examined.
    code = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "int(n_finite * 0.8)" not in code
    assert "X_research[:split_idx]" not in code
    assert "X_vol[n_train:val_end]" not in code
    assert "build_purged_split" in code
    assert "assert_design_matrix_supports_fit" in code
