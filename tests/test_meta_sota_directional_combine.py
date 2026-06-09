"""L4565 — SOTA directional-combine: the predictor-side COIN fix.

Root cause (L4549b, confirmed against the live 6/07 model): the volatility L1's
``expected_move`` — a directionless MAGNITUDE (E[|move|]) — entered the meta
Ridge as a large positive ADDITIVE directional term (coef +3.18, the single
highest-importance feature). COIN, a ~6σ ``expected_move`` outlier, was therefore
ranked the #1 alpha pick every day regardless of its strongly-negative momentum.

The fix has three composable, flag-gated levers (defaults preserve the champion
byte-for-byte):
  1. EXPECTED_MOVE_IN_META=false  — drop expected_move from the directional vector
  2. META_STANDARDIZE_ENABLED=true — z-score + winsorize the directional features
  3. RESIDUAL_MOMENTUM_ENABLED + MOMENTUM_L1_IN_META=false — swap dead raw momentum

These tests lock: (a) the feature-gate invariants (champion byte-identical at
defaults), (b) the encapsulated standardize+winsorize scaler (build / apply /
persist / legacy-compat), and (c) the behavioral guarantee — a COIN-shape row is
rank-1 under the champion config and NOT rank-1 under the fix.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import (
    META_DIRECTIONAL_FEATURES,
    META_FEATURES,
    MetaModel,
)
from training.meta_trainer import build_train_meta_features


# ── Lever 1: EXPECTED_MOVE_IN_META feature gate ────────────────────────────


class TestExpectedMoveGate:
    def test_defaults_byte_identical_to_meta_features(self):
        # resid off, momentum in, expected_move in → exactly META_FEATURES.
        assert build_train_meta_features(False, True, True) == list(META_FEATURES)

    def test_expected_move_false_drops_only_that_column(self):
        feats = build_train_meta_features(False, True, expected_move_in_meta=False)
        assert "expected_move" not in feats
        assert feats == [f for f in META_FEATURES if f != "expected_move"]
        assert len(feats) == len(META_FEATURES) - 1

    def test_full_sota_combo_drops_two_swaps_one(self):
        # The sota-directional-combine spec: expected_move OUT, raw momentum OUT,
        # residual momentum IN.
        feats = build_train_meta_features(
            True, momentum_l1_in_meta=False, expected_move_in_meta=False
        )
        assert "expected_move" not in feats
        assert "momentum_score" not in feats
        assert feats[-1] == "residual_momentum_score"
        assert len(feats) == len(META_FEATURES) - 1  # -2 dropped, +1 appended

    def test_module_constant_never_mutated(self):
        before = list(META_FEATURES)
        build_train_meta_features(True, False, False)
        build_train_meta_features(False, True, True)
        assert list(META_FEATURES) == before
        assert "expected_move" in META_FEATURES  # constant intact


# ── Lever 2: directional standardize + winsorize scaler ────────────────────


def _fit_data(feats, n=600, seed=1):
    rng = np.random.default_rng(seed)
    d = len(feats)
    X = rng.normal(0.0, 1.0, (n, d))
    # Give the macro columns a single shared value per "date" is not needed for
    # the unit tests — only that directional vs non-directional split holds.
    rc = feats.index("research_calibrator_prob")
    mo = feats.index("momentum_score")
    y = 0.6 * X[:, rc] - 0.4 * X[:, mo] + rng.normal(0.0, 0.5, n)
    return X, y


class TestDirectionalScaler:
    def test_scaler_keys_only_directional_columns(self):
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=True)
        assert m._scaler is not None
        # Every scaled column is a directional feature...
        assert all(f in META_DIRECTIONAL_FEATURES for f in m._scaler["directional"])
        # ...and no macro / regime column is scaled.
        for raw in ("macro_spy_20d_return", "macro_vix_level", "regime_intensity_z",
                    "expected_move"):
            assert raw not in m._scaler["directional"]

    def test_standardize_false_leaves_scaler_none(self):
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=False)
        assert m._scaler is None

    def test_winsor_bounds_extreme_directional_input(self):
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(
            X, y, feature_names=feats, standardize_directional=True, winsor=3.0
        )
        # A 100σ outlier on a directional feature is clipped to ±3σ before the
        # Ridge — its standardized value in the transformed row must be ±3.
        row = {f: 0.0 for f in feats}
        row["research_calibrator_prob"] = 100.0
        x = np.array([[row[f] for f in feats]], dtype=np.float64)
        xt = m._apply_scaler(x)
        j = feats.index("research_calibrator_prob")
        assert abs(xt[0, j]) <= 3.0 + 1e-9

    def test_macro_column_passes_through_untouched(self):
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=True)
        row = {f: 0.0 for f in feats}
        row["macro_vix_level"] = 7.3
        x = np.array([[row[f] for f in feats]], dtype=np.float64)
        xt = m._apply_scaler(x)
        j = feats.index("macro_vix_level")
        assert xt[0, j] == 7.3  # raw, unscaled

    def test_save_load_roundtrip_preserves_scaler_and_predictions(self):
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=True)
        path = os.path.join(tempfile.mkdtemp(), "mm.pkl")
        m.save(path)
        m2 = MetaModel.load(path)
        assert m2._scaler is not None
        assert m2._scaler["directional"] == m._scaler["directional"]
        rng = np.random.default_rng(7)
        for _ in range(20):
            row = {f: float(rng.normal()) for f in feats}
            assert abs(m.predict_single(row) - m2.predict_single(row)) < 1e-9

    def test_legacy_model_without_scaler_is_identity(self):
        # A model fit WITHOUT standardization (legacy / champion) has no scaler;
        # predict must equal the bare estimator on raw X (no transform leak).
        feats = build_train_meta_features(False, True, True)
        X, y = _fit_data(feats)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=False)
        assert m._scaler is None
        x = X[:5]
        np.testing.assert_allclose(m.predict(x), m._model.predict(x))


# ── Lever 3 + behavioral: the COIN-shape regression gate ───────────────────


def _coin_cross_section(feats, seed=3):
    """A daily cross-section of 30 tickers. One ('COIN') is a high-vol,
    negative-momentum, neutral-research name; the rest are ordinary. Returns
    (X_batch, coin_idx)."""
    rng = np.random.default_rng(seed)
    n = 30
    X = np.zeros((n, len(feats)))
    # ordinary names: modest vol, mixed momentum, varied research
    for j, f in enumerate(feats):
        if f == "expected_move":
            X[:, j] = np.abs(rng.normal(0.03, 0.008, n))  # ~cohort
        elif f == "research_calibrator_prob":
            X[:, j] = rng.uniform(0.45, 0.62, n)
        elif f == "momentum_score":
            X[:, j] = rng.normal(0.0, 0.4, n)
        else:
            X[:, j] = rng.normal(0.0, 0.3, n)
    coin = 0
    if "expected_move" in feats:
        X[coin, feats.index("expected_move")] = 0.093          # ~6σ outlier
    X[coin, feats.index("research_calibrator_prob")] = 0.515    # neutral
    X[coin, feats.index("momentum_score")] = -0.9               # strongly negative
    return X, coin


def _train_with_vol_signal(feats, standardize, seed=11):
    """Training data that reproduces the LIVE model's pathology: ``expected_move``
    carries a strong (spurious) positive in-sample correlation with y — the
    vol↔alpha relationship that taught the champion its +3.18 coefficient —
    while raw momentum is DEAD (zero coefficient, as observed: WF IC −0.002).
    research_calibrator_prob carries the honest directional signal. So under the
    champion, ``expected_move`` dominates; under the fix (column removed), the
    Ridge ranks by research and COIN's neutral research lands it mid-pack."""
    rng = np.random.default_rng(seed)
    n = 900
    X = rng.normal(0.0, 1.0, (n, len(feats)))
    if "expected_move" in feats:
        # Wide vol spread (a real universe has COIN-class high-vol names) so the
        # Ridge identifies a LARGE coefficient on this low-σ feature — exactly
        # the live model's +3.18 (top standardized importance). Tuned so
        # expected_move dominates y's variance over the honest research signal.
        X[:, feats.index("expected_move")] = np.abs(rng.normal(0.04, 0.03, n))
    rc = feats.index("research_calibrator_prob")
    # y: weak honest research signal + DOMINANT spurious expected_move; momentum
    # is pure noise → ≈0 learned weight, mirroring the dead-momentum reality.
    y = 0.2 * X[:, rc] + rng.normal(0.0, 0.1, n)
    if "expected_move" in feats:
        y = y + 12.0 * X[:, feats.index("expected_move")]
    m = MetaModel().fit(
        X, y, feature_names=feats, standardize_directional=standardize
    )
    return m


class TestCoinBehavioralGate:
    def test_champion_ranks_coin_first(self):
        # Champion config: expected_move IN, no standardization. The COIN-shape
        # row should be (mis-)ranked #1 by its huge expected_move.
        feats = build_train_meta_features(False, True, True)
        m = _train_with_vol_signal(feats, standardize=False)
        X, coin = _coin_cross_section(feats)
        preds = m.predict(X)
        rank = int(np.argsort(-preds).tolist().index(coin)) + 1
        assert rank == 1, f"expected COIN rank 1 under champion, got {rank}"

    def test_fix_demotes_coin(self):
        # Fix config: expected_move OUT + standardize + winsorize. The COIN-shape
        # row must NOT be rank 1 (its directional signals are neutral/negative).
        feats = build_train_meta_features(False, True, expected_move_in_meta=False)
        m = _train_with_vol_signal(feats, standardize=True)
        X, coin = _coin_cross_section(feats)
        preds = m.predict(X)
        rank = int(np.argsort(-preds).tolist().index(coin)) + 1
        assert rank > 1, f"COIN should not be rank 1 under the fix, got {rank}"


# ── Model-zoo whitelist ────────────────────────────────────────────────────


class TestZooWhitelist:
    def test_new_levers_are_whitelisted(self):
        from training.model_zoo import _ALLOWED_OVERRIDES
        assert "EXPECTED_MOVE_IN_META" in _ALLOWED_OVERRIDES
        assert "META_STANDARDIZE_ENABLED" in _ALLOWED_OVERRIDES

    def test_sota_spec_overrides_validate(self):
        from training.model_zoo import spec_overrides
        # The sota-directional-combine spec's overrides must pass validation.
        ov = {
            "EXPECTED_MOVE_IN_META": False,
            "META_STANDARDIZE_ENABLED": True,
            "RESIDUAL_MOMENTUM_ENABLED": True,
            "MOMENTUM_L1_IN_META": False,
        }
        with spec_overrides(ov):
            import config as cfg
            assert cfg.EXPECTED_MOVE_IN_META is False
            assert cfg.META_STANDARDIZE_ENABLED is True
