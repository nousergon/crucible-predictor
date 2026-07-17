"""config#856 — the weekly training email is slimmed to a headline + a console
deep-link (pull-for-state, push-on-transition).

The per-step content email is replaced by the console Predictor Training page
(crucible-dashboard ``views/36_Predictor_Training.py``, ``url_path=
"predictor-training"``). The email keeps only the operator-actionable headline
(model + IC-gate verdict + promotion status + the shadow-calibrator readiness
row + data warnings) and deep-links to ``…/predictor-training?date=YYYY-MM-DD``;
the bulky tables (walk-forward folds, feature-importance bars, gain-vs-SHAP,
feature health, calibration, multi-horizon) move to the console page. The
underlying ``training_summary_{date}.json`` artifact is UNCHANGED (no data loss).

These pin the slug contract with the dashboard and the slim shape.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import train_handler as th


def _base_result():
    return {
        "model_version": "v3.0-meta",
        "passes_ic_gate": True,
        "promoted": False,
        "auto_promote_enabled": False,
        "promotion_gate_detail": {"promoted_blocker_reason": "challenger_first"},
        "val_ic": 0.05,
        "test_ic": 0.06,
        "mse_ic": 0.06,
        "rank_ic": 0.04,
        "ic_ir": 1.2,
        "ic_positive_20": 14,
        "meta_model_oos_ic": 0.061,
        "n_train": 12000,
        "elapsed_s": 300,
        "walk_forward": {
            "volatility_median_ic": 0.32,
            "momentum_median_ic": -0.001,
            "passes_wf": True,
            "folds": [
                {"fold": 1, "test_start": "2026-01-01", "test_end": "2026-01-21",
                 "vol_ic": 0.3, "mom_ic": 0.0},
                {"fold": 2, "test_start": "2026-02-01", "test_end": "2026-02-21",
                 "vol_ic": 0.34, "mom_ic": -0.01},
            ],
        },
        "shadow_calibration": {
            "fitted": True, "method": "isotonic", "n_samples": 5000,
            "ece_before": 0.08, "ece_after": 0.02,
            "output_distribution_gate_passed": True,
            "stratified_per_regime_passed": True,
        },
        "noise_candidates": ["feat_x", "feat_y"],
        "feature_importance_top10": [{"feature": "mom_21d", "gain": 12.3}],
    }


def _capture(monkeypatch, result, date_str="2026-06-13"):
    captured: dict = {}

    def _fake_send(subject, plain, *, recipients=None, html=None, sender=None, region=None):
        captured["subject"] = subject
        captured["plain"] = plain
        captured["html"] = html
        return True

    import config as cfg
    import krepis.email_sender as es
    monkeypatch.setattr(es, "send_email", _fake_send)
    monkeypatch.setattr(cfg, "EMAIL_SENDER", "x@y.z", raising=False)
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", ["a@b.c"], raising=False)
    ok = th.send_training_email(result, date_str)
    assert ok is True
    return captured


def test_slug_matches_dashboard_contract():
    # The dashboard app.py pins url_path="predictor-training" and
    # tests/test_predictor_training_page.py guards it. Drift here breaks the
    # emailed deep-link.
    assert th.TRAINING_PAGE_SLUG == "predictor-training"


def test_email_carries_console_deep_link(monkeypatch):
    cap = _capture(monkeypatch, _base_result())
    for body in (cap["plain"], cap["html"]):
        assert "predictor-training?date=2026-06-13" in body


def test_email_keeps_operator_actionable_headline(monkeypatch):
    cap = _capture(monkeypatch, _base_result())
    # IC-gate verdict + promotion status stay in the email.
    assert "PASS" in cap["subject"]
    assert "Registered challenger" in cap["plain"]
    # Shadow-calibrator readiness row is retained (kept in-email per config#856).
    assert "Shadow Calibrator" in cap["html"]
    # Data warnings surface inline.
    assert "Noise candidates" in cap["plain"]
    # One-line walk-forward verdict, but NOT the per-fold table.
    assert "Walk-forward" in cap["plain"]


def test_email_drops_bulky_tables(monkeypatch):
    cap = _capture(monkeypatch, _base_result())
    html = cap["html"]
    # The bulky inline tables that moved to the console page must NOT be in the
    # email any more (their distinctive headers).
    assert "Gain vs SHAP" not in html
    assert "Feature Health" not in html
    assert "Multi-Horizon Models" not in html
    # No per-fold walk-forward table columns in the email.
    assert "Test Start" not in html
