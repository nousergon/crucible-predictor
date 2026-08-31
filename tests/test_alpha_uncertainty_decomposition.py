"""alpha-engine-config-I9446 — the α̂-uncertainty channel must carry per-name content.

`predicted_alpha_std` is `BayesianRidge.predict(..., return_std=True)`, i.e.

    σ_pred(x)² = 1/α̂ + xᵀ Σ_w x
                 ^^^^   ^^^^^^^^^
                 scalar  per-name

The first term is learned once at fit time and is IDENTICAL for every name in a
batch. Measured over all 62 stored `predictions/{date}.json` artifacts from the
2026-06-01 BayesianRidge cutover to 2026-08-31 it carries 90–98% of σ²_pred on a
healthy champion, and the cross-sectional coefficient of variation of the
emitted total never exceeded 0.008 in ANY session. The executor's
Garlappi-Uppal-Wang penalty, whose Ω is an ESTIMATION-error covariance, was
therefore a uniform ridge for three months and nothing detected it.

These tests pin, in order:
  • the decomposition identity and that only the epistemic half varies;
  • that a registry-loaded champion (pickle, no `.pkl.meta.json` sidecar) can
    still be decomposed — the sidecar was the only source of the noise
    precision on load, and the registry prefix carries no sidecar;
  • that the producer emits both new fields;
  • that the gate records the cross-sectional CV of both halves; and
  • a REPLAY over two real stored sessions — one healthy champion, one whose
    posterior never left its prior — showing the detector separates them while
    a threshold on the emitted total could not.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pickle
import tempfile

import numpy as np
import pytest

from model.meta_model import META_FEATURES, MetaModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Real stored sessions -----------------------------------------------------
# `predicted_alpha_std` per name, straight out of
# s3://alpha-engine-research/predictor/predictions/{date}.json, paired with the
# `alpha_` of the champion `champion_version_id` names in that same artifact
# (read from s3://alpha-engine-research/predictor/registry/{version_id}/
# meta_model.pkl). Two sessions, deliberately chosen as the two regimes:

# Healthy champion. Epistemic share of variance ≈ 1.9%; the emitted total is
# still flat (CV 0.0018) because the σ_n floor swamps it.
SESSION_HEALTHY = {
    "date": "2026-08-31",
    "champion_version_id": "v3.0-meta-2026-08-14-119e069b",
    "alpha_": 107.28362438280664,
    "predicted_alpha_std": [
        0.097272, 0.097287, 0.097369, 0.097376, 0.097441, 0.097442, 0.097443,
        0.09745, 0.097453, 0.097461, 0.097464, 0.097464, 0.097468, 0.097469,
        0.097471, 0.097475, 0.097482, 0.097483, 0.097483, 0.097483, 0.097487,
        0.097491, 0.097492, 0.097492, 0.097501, 0.097509, 0.097512, 0.097605,
        0.098344,
    ],
}

# The 2026-08-21 champion, rolled back on 2026-08-31. Its BayesianRidge `sigma_`
# diagonal sits at exactly 1/lambda_ (0.0389) on 10 of 16 columns — the fit
# never updated off its prior there, and those columns are the market-wide
# macro features, identical for every ticker. So xᵀΣ_w x is itself a constant:
# epistemic share of variance ≈ 82%, and yet its cross-section is DEAD.
SESSION_DEGENERATE = {
    "date": "2026-08-26",
    "champion_version_id": "v3.0-meta-2026-08-21-7d3d1cce",
    "alpha_": 93.18750247305927,
    "predicted_alpha_std": [
        0.242014, 0.242018, 0.242023, 0.242023, 0.242024, 0.242024, 0.242024,
        0.242024, 0.242024, 0.242025, 0.242025, 0.242025, 0.242026, 0.242026,
        0.242027, 0.242027, 0.242027, 0.242028, 0.242028, 0.24203, 0.242031,
        0.242036, 0.242038, 0.24204, 0.242044, 0.242046, 0.242049, 0.242057,
        0.242075, 0.242086,
    ],
}


def _cv(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr) / np.mean(arr))


def _replay(session: dict) -> list[float]:
    """Decompose a stored session's totals through the real production path."""
    mm = MetaModel()
    mm._fitted = True
    mm._model = _FakeBR(session["alpha_"])
    out = []
    for total in session["predicted_alpha_std"]:
        epi, alea = mm.decompose_alpha_std(total)
        assert alea == pytest.approx(math.sqrt(1.0 / session["alpha_"]))
        out.append(epi)
    return out


class _FakeBR:
    """Minimal stand-in exposing only what the decomposition reads."""

    def __init__(self, alpha_: float):
        self.alpha_ = alpha_


def _synth(feature_list, n=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(feature_list)))
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] + rng.normal(scale=0.02, size=n)
    return X, y


class TestDecompositionIdentity:
    def test_halves_reconstruct_the_total_and_only_epistemic_varies(self):
        X, y = _synth(META_FEATURES)
        mm = MetaModel().fit(X, y, feature_names=META_FEATURES)

        aleatorics, epistemics, totals = [], [], []
        for row in X[:40]:
            feats = dict(zip(META_FEATURES, row))
            _, total = mm.predict_single_with_std(feats)
            epi, alea = mm.decompose_alpha_std(total)
            assert epi is not None and alea is not None
            # Exact by construction: epi² + alea² == total².
            assert epi**2 + alea**2 == pytest.approx(total**2, rel=1e-12)
            aleatorics.append(alea)
            epistemics.append(epi)
            totals.append(total)

        # The aleatoric half is one number, repeated.
        assert len(set(round(a, 15) for a in aleatorics)) == 1
        assert aleatorics[0] == pytest.approx(math.sqrt(1.0 / mm._model.alpha_))
        # The epistemic half is where the cross-section lives, and it is where
        # the cross-section is BIGGER — that is the whole point of the split.
        assert _cv(epistemics) > _cv(totals) * 10

    def test_legacy_ridge_decomposes_to_none_rather_than_a_fabricated_value(self):
        from sklearn.linear_model import Ridge

        X, y = _synth(META_FEATURES)
        mm = MetaModel()
        mm._model = Ridge(alpha=1.0).fit(X, y)
        mm._fitted = True
        mm._feature_names = list(META_FEATURES)

        assert mm.aleatoric_std() is None
        assert mm.decompose_alpha_std(0.1) == (None, None)
        # And the total itself is still None on that path, unchanged.
        _, total = mm.predict_single_with_std(dict(zip(META_FEATURES, X[0])))
        assert total is None

    def test_none_and_nonfinite_totals_never_fabricate(self):
        X, y = _synth(META_FEATURES)
        mm = MetaModel().fit(X, y, feature_names=META_FEATURES)
        assert mm.decompose_alpha_std(None) == (None, None)
        assert mm.decompose_alpha_std(float("nan")) == (None, None)
        assert mm.decompose_alpha_std(-1.0) == (None, None)
        # A total a hair BELOW the aleatoric floor (float noise) clamps to 0,
        # it does not produce a NaN from a negative square root.
        alea = mm.aleatoric_std()
        epi, _ = mm.decompose_alpha_std(alea * (1 - 1e-12))
        assert epi == 0.0


class TestSidecarIndependence:
    """`predictor/registry/{version_id}/` stores meta_model.pkl with NO
    `.pkl.meta.json`. Before I9446 the learned noise precision was restored only
    from that sidecar, so a registry-loaded champion reported None for it."""

    def test_load_without_sidecar_still_knows_its_noise_precision(self):
        X, y = _synth(META_FEATURES)
        mm = MetaModel().fit(X, y, feature_names=META_FEATURES)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "meta_model.pkl")
            mm.save(path)
            os.remove(path + ".meta.json")  # registry-prefix shape
            assert not os.path.exists(path + ".meta.json")
            loaded = MetaModel.load(path)

        assert loaded._learned_alpha is not None
        assert loaded.aleatoric_std() == pytest.approx(mm.aleatoric_std())
        assert loaded.metrics()["learned_alpha_noise_precision"] is not None


class TestProducerEmitsBothHalves:
    def test_prediction_dict_literal_carries_the_decomposition(self):
        path = os.path.join(_REPO, "inference", "stages", "run_inference.py")
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        literals = [
            {k.value for k in n.keys
             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            for n in ast.walk(tree) if isinstance(n, ast.Dict)
        ]
        emitting = [k for k in literals
                    if "ticker" in k and "predicted_direction" in k]
        assert emitting, "per-ticker prediction dict literal not found"
        for keys in emitting:
            assert "predicted_alpha_std" in keys
            assert "predicted_alpha_std_epistemic" in keys
            assert "predicted_alpha_std_aleatoric" in keys


class TestGateRecordsTheCrossSection:
    def _metrics(self, entries):
        from model.output_distribution_gate import validate_live_batch_invariant_health

        return validate_live_batch_invariant_health(entries).metrics

    def _entry(self, i, total, epi, alea):
        p_up = 0.5 + 0.005 * (i + 1)
        return {
            "ticker": f"T{i}",
            "predicted_direction": "UP",
            # The gate enforces confidence == |p_up-0.5|*2 as a units contract.
            "prediction_confidence": abs(p_up - 0.5) * 2.0,
            "predicted_alpha": 0.01 * (i - 10),
            "p_up": p_up,
            "p_down": 1.0 - p_up,
            "predicted_alpha_std": total,
            "predicted_alpha_std_epistemic": epi,
            "predicted_alpha_std_aleatoric": alea,
        }

    def test_epistemic_cv_is_recorded_and_exceeds_the_totals_cv(self):
        s = SESSION_HEALTHY
        alea = math.sqrt(1.0 / s["alpha_"])
        epis = _replay(s)
        entries = [self._entry(i, t, e, alea)
                   for i, (t, e) in enumerate(zip(s["predicted_alpha_std"], epis))]
        m = self._metrics(entries)

        assert m["n_alpha_uncertainty_present"] == len(entries)
        assert m["n_alpha_uncertainty_decomposed"] == len(entries)
        assert m["alpha_uncertainty_cv"] == pytest.approx(
            _cv(s["predicted_alpha_std"]), rel=1e-3)
        assert m["alpha_uncertainty_epistemic_cv"] > 10 * m["alpha_uncertainty_cv"]
        assert m["alpha_uncertainty_aleatoric"] == pytest.approx(alea, rel=1e-6)
        # Observation noise carries the overwhelming majority of the variance.
        assert m["alpha_uncertainty_epistemic_var_share"] < 0.10

    def test_metrics_are_none_not_zero_when_the_field_is_absent(self):
        entries = [
            {"ticker": f"T{i}", "predicted_direction": "UP",
             "prediction_confidence": 0.1, "predicted_alpha": 0.01 * (i - 6),
             "p_up": 0.55, "p_down": 0.45}
            for i in range(12)
        ]
        m = self._metrics(entries)
        assert m["n_alpha_uncertainty_present"] == 0
        assert m["alpha_uncertainty_cv"] is None
        assert m["alpha_uncertainty_epistemic_cv"] is None
        assert m["alpha_uncertainty_epistemic_var_share"] is None

    def test_the_uncertainty_metrics_never_drive_the_verdict(self):
        """A dead uncertainty channel degrades SIZING; it must not halt a day."""
        s = SESSION_DEGENERATE
        alea = math.sqrt(1.0 / s["alpha_"])
        epis = _replay(s)
        entries = [self._entry(i, t, e, alea)
                   for i, (t, e) in enumerate(zip(s["predicted_alpha_std"], epis))]
        from model.output_distribution_gate import validate_live_batch_invariant_health

        result = validate_live_batch_invariant_health(entries)
        assert result.metrics["alpha_uncertainty_epistemic_cv"] < 0.001
        assert result.passed, (
            "a degenerate α̂-uncertainty channel must not fail the output "
            "distribution gate — the tradable alpha vector is unaffected"
        )


class TestDetectorReplayOverStoredSessions:
    """The number that says the channel is working, checked against history."""

    def _alerts(self, session, epis):
        from monitoring import drift_detector as dd

        alea = math.sqrt(1.0 / session["alpha_"])
        today = [
            {"ticker": f"T{i}", "predicted_direction": "UP" if i % 2 else "DOWN",
             "prediction_confidence": 0.3 + 0.01 * i,
             "predicted_alpha": 0.01 * (i - 10),
             "predicted_alpha_std": t,
             "predicted_alpha_std_epistemic": e,
             "predicted_alpha_std_aleatoric": alea}
            for i, (t, e) in enumerate(zip(session["predicted_alpha_std"], epis))
        ]
        return dd.alpha_uncertainty_alerts(today, session["date"])

    def _codes(self, alerts):
        return {a["code"] for a in alerts}

    def test_healthy_champion_session_does_not_fire(self):
        s = SESSION_HEALTHY
        alerts = self._alerts(s, _replay(s))
        assert "alpha_uncertainty_degeneration" not in self._codes(alerts)

    def test_degenerate_champion_session_fires(self):
        s = SESSION_DEGENERATE
        alerts = self._alerts(s, _replay(s))
        codes = self._codes(alerts)
        assert "alpha_uncertainty_degeneration" in codes
        alert = next(a for a in alerts
                     if a["code"] == "alpha_uncertainty_degeneration")
        assert alert["severity"] == "WARN"
        assert alert["value"] < alert["threshold"]

    def test_a_threshold_on_the_emitted_total_could_not_separate_them(self):
        """Why the detector reads the epistemic half and not the total.

        The healthy session's total-CV (0.0018) is LOWER than the degenerate
        session's (0.00007) is small — both sit deep inside the same band, so no
        floor on the total distinguishes them without firing on healthy days.
        """
        healthy_total_cv = _cv(SESSION_HEALTHY["predicted_alpha_std"])
        degenerate_total_cv = _cv(SESSION_DEGENERATE["predicted_alpha_std"])
        assert healthy_total_cv < 0.008
        assert degenerate_total_cv < 0.008

        healthy_epi_cv = _cv(_replay(SESSION_HEALTHY))
        degenerate_epi_cv = _cv(_replay(SESSION_DEGENERATE))
        from monitoring.drift_detector import ALPHA_UNCERTAINTY_MIN_CV

        assert degenerate_epi_cv < ALPHA_UNCERTAINTY_MIN_CV < healthy_epi_cv
        # And with real margin on both sides, not a hairline.
        assert healthy_epi_cv > 2.5 * ALPHA_UNCERTAINTY_MIN_CV
        assert degenerate_epi_cv < ALPHA_UNCERTAINTY_MIN_CV / 10

    def test_pre_decomposition_artifacts_are_skipped_not_judged(self):
        """Artifacts written before this shipped carry no `_epistemic` field.
        Judging them on the total would fire on every historical day."""
        from monitoring import drift_detector as dd

        today = [
            {"ticker": f"T{i}", "predicted_direction": "UP" if i % 2 else "DOWN",
             "prediction_confidence": 0.3 + 0.01 * i,
             "predicted_alpha": 0.01 * (i - 10),
             "predicted_alpha_std": t}
            for i, t in enumerate(SESSION_HEALTHY["predicted_alpha_std"])
        ]
        alerts = dd.alpha_uncertainty_alerts(today, SESSION_HEALTHY["date"])
        assert alerts == []
