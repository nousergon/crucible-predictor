"""config#2882 — the base training email surfaces data_coverage_degraded.

send_training_email() renders the run's own IC/promotion verdict; a training
run that completed on measurably-incomplete ArcticDB coverage must not look
indistinguishable from a healthy run in that SAME email. These tests call
the real formatter (krepis' send_email mocked out) with a minimal-but-
realistic meta-model result dict to confirm the warning renders (and that a
healthy run renders nothing extra / doesn't crash on missing keys).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import train_handler as th


def _base_result(**overrides):
    result = {
        "model_version": "v3.0-meta",
        "promoted": False,
        "promoted_mode": None,
        "passes_ic_gate": True,
        "auto_promote_enabled": False,
        "val_ic": 0.05,
        "test_ic": 0.05,
        "mse_ic": 0.05,
        "ic_ir": 0.0,
        "elapsed_s": 120.0,
        "n_train": 1000,
        "ic_positive_20": 12,
        "feature_importance_top10": [],
        "meta_model_ic": 0.05,
        "meta_model_oos_ic": 0.04,
        "walk_forward": {},
        "calibration": {},
        "shadow_calibration": {},
        "promotion_gate_detail": {"promoted_blocker_reason": "challenger_first"},
    }
    result.update(overrides)
    return result


@pytest.fixture
def mocked_email(monkeypatch):
    monkeypatch.setattr(cfg, "EMAIL_SENDER", "sender@example.com", raising=False)
    monkeypatch.setattr(cfg, "EMAIL_RECIPIENTS", ["ops@example.com"], raising=False)
    captured = {}

    def _fake_send_email(subject, plain_body, *, recipients, html, sender, region):
        captured["subject"] = subject
        captured["plain_body"] = plain_body
        captured["html"] = html
        return True

    with patch("krepis.email_sender.send_email", _fake_send_email):
        yield captured


def test_degraded_coverage_renders_warning_in_both_bodies(mocked_email):
    result = _base_result(
        data_coverage_degraded=True,
        arctic_coverage={
            "n_written": 850, "n_expected": 900, "n_failed": 50,
            "coverage_ratio": 850 / 900, "failed_universe": [], "failed_macro": [],
        },
    )
    ok = th.send_training_email(result, "2026-07-18")
    assert ok is True
    assert "DATA COVERAGE DEGRADED" in mocked_email["plain_body"]
    assert "config#2882" in mocked_email["plain_body"]
    assert "DATA COVERAGE DEGRADED" in mocked_email["html"]


def test_healthy_coverage_renders_no_warning(mocked_email):
    result = _base_result(
        data_coverage_degraded=False,
        arctic_coverage={
            "n_written": 900, "n_expected": 900, "n_failed": 0,
            "coverage_ratio": 1.0, "failed_universe": [], "failed_macro": [],
        },
    )
    th.send_training_email(result, "2026-07-18")
    assert "DATA COVERAGE DEGRADED" not in mocked_email["plain_body"]
    assert "DATA COVERAGE DEGRADED" not in mocked_email["html"]


def test_missing_coverage_fields_does_not_crash(mocked_email):
    # A result dict from before config#2882 (or arctic_coverage=None) must
    # still render cleanly — no KeyError on the new fields.
    result = _base_result()
    ok = th.send_training_email(result, "2026-07-18")
    assert ok is True
    assert "DATA COVERAGE DEGRADED" not in mocked_email["plain_body"]
