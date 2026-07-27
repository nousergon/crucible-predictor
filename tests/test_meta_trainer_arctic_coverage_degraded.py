"""Tests for ``training.meta_trainer._data_coverage_degraded`` (config#2882).

The single chokepoint that decides whether a training run's ArcticDB
read-coverage counts as "degraded" for the manifest/result-dict
``data_coverage_degraded`` field — the signal ``model_zoo.select_winner``
threads onto a candidate's leaderboard entry so a challenger that trained on
a measurably-incomplete dataset is visible before/after a promotion, per the
issue's acceptance criterion 2(b).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training.meta_trainer import _data_coverage_degraded


def _coverage(ratio):
    return {
        "n_written": int(ratio * 900), "n_expected": 900,
        "n_failed": 900 - int(ratio * 900),
        "coverage_ratio": ratio, "failed_universe": [], "failed_macro": [],
    }


def test_none_input_returns_none():
    # No coverage supplied (e.g. an injected-data test harness bypassing
    # ArcticDB entirely) — "unknown", not silently "healthy".
    assert _data_coverage_degraded(None) is None


def test_full_coverage_not_degraded(monkeypatch):
    monkeypatch.setattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", 0.98, raising=False)
    assert _data_coverage_degraded(_coverage(1.0)) is False


def test_just_above_degraded_floor_not_degraded(monkeypatch):
    monkeypatch.setattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", 0.98, raising=False)
    assert _data_coverage_degraded(_coverage(0.99)) is False


def test_just_below_degraded_floor_is_degraded(monkeypatch):
    """config#2882 acceptance criterion 3 — N tickers requested, M < N read
    (M > 0): the degraded flag must correctly reflect a partial loss even
    when it's mild enough to be well above the train_handler.py hard floor."""
    monkeypatch.setattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", 0.98, raising=False)
    assert _data_coverage_degraded(_coverage(0.97)) is True


def test_severe_partial_loss_is_degraded(monkeypatch):
    monkeypatch.setattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", 0.98, raising=False)
    # 850/900 read (the issue's own worked example) — well below the floor.
    assert _data_coverage_degraded(_coverage(850 / 900)) is True


def test_threshold_is_env_configurable(monkeypatch):
    monkeypatch.setattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", 0.5, raising=False)
    # 0.7 would be degraded under the default 0.98 floor but is healthy
    # under an operator-loosened 0.5 floor.
    assert _data_coverage_degraded(_coverage(0.7)) is False


def test_missing_config_attr_defaults_to_098(monkeypatch):
    monkeypatch.delattr(cfg, "ARCTIC_DEGRADED_COVERAGE_RATIO", raising=False)
    assert _data_coverage_degraded(_coverage(0.99)) is False
    assert _data_coverage_degraded(_coverage(0.97)) is True
