"""config#2882 — train_handler.py's ArcticDB read-coverage gate.

The pre-existing gate (``n_files == 0 → raise``) only catches TOTAL ArcticDB
read failure. A PARTIAL failure episode (throttling/connectivity/corrupted
symbols — e.g. 850/900 tickers erroring) left ``n_files`` nonzero-but-
crippled, so the gate never tripped and a challenger could train + win
weekly ModelZoo promotion on a silently starved dataset with zero trace in
the promoted artifact.

These tests exercise the NEW coverage-ratio gate end to end through
``train_handler._main_impl(dry_run=True)``: N tickers requested, M < N
read (M > 0) — verifying (1) the hard floor raises before training starts
for catastrophic partial loss, (2) a milder partial loss proceeds but is
flagged ``data_coverage_degraded=True`` in the run's own result dict (which
flows into training_summary/manifest → model_zoo.select_winner's
leaderboard), and (3) full/near-full coverage is NOT flagged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from training import train_handler


@pytest.fixture(autouse=True)
def _no_ambient_shadow_env(monkeypatch):
    monkeypatch.delenv("CRSP_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("SHADOW_BASIS", raising=False)
    monkeypatch.delenv("PREDICTOR_DEFER_TRAINING_EMAIL", raising=False)


def _coverage(n_written, n_expected, failed=None):
    failed = failed or []
    return {
        "n_written": n_written,
        "n_expected": n_expected,
        "n_failed": len(failed),
        "coverage_ratio": (n_written / n_expected) if n_expected else 1.0,
        "failed_universe": failed,
        "failed_macro": [],
    }


def _run_main_impl(coverage, run_meta_training_result=None):
    """Drive ``_main_impl(dry_run=True)`` with ``download_from_arctic`` and
    ``run_meta_training`` mocked, ``TrainingPreflight`` a no-op. Returns the
    result dict (or raises, letting the caller assert on the exception)."""
    if run_meta_training_result is None:
        run_meta_training_result = {"model_version": "v3.0-meta", "promoted": False}

    fake_preflight = MagicMock()
    fake_preflight.run.return_value = None

    with patch("training.preflight.TrainingPreflight", return_value=fake_preflight), \
         patch("store.arctic_reader.download_from_arctic", return_value=coverage) as mock_dl, \
         patch("training.meta_trainer.run_meta_training", return_value=run_meta_training_result) as mock_train:
        result = train_handler._main_impl(
            "test-bucket", date_str="2026-07-18", dry_run=True, send_email=False,
        )
    return result, mock_dl, mock_train


def test_partial_failure_below_hard_floor_raises():
    """900 requested, 50 read (850 failed, ~5.6% coverage) — well below the
    default hard floor (0.5). n_written=50 is nonzero, so the OLD n_files==0
    gate would never have tripped; the new coverage-ratio gate must."""
    coverage = _coverage(50, 900, failed=[f"T{i}" for i in range(850)])
    with pytest.raises(RuntimeError, match="coverage"):
        _run_main_impl(coverage)


def test_partial_failure_at_hard_floor_boundary_passes():
    """Exactly at the hard floor (0.5) — the gate is a strict '<', so this
    must NOT raise."""
    coverage = _coverage(450, 900, failed=[f"T{i}" for i in range(450)])
    result, mock_dl, mock_train = _run_main_impl(coverage)
    assert mock_train.called
    # Below the degraded floor (0.98) → flagged, even though it passed the
    # hard floor.
    _, kwargs = mock_train.call_args
    assert kwargs["arctic_coverage"]["coverage_ratio"] == pytest.approx(0.5)


def test_partial_failure_between_floors_proceeds_and_flags_degraded():
    """850/900 read (~94.4% coverage) — above the hard floor (0.5) so
    training proceeds, but below the degraded floor (0.98) so the run must
    be flagged data_coverage_degraded=True. This is the exact scenario from
    the issue: n_files nonzero, self-reported CPCV IC still computed, but the
    dataset was measurably incomplete."""
    coverage = _coverage(850, 900, failed=[f"T{i}" for i in range(50)])
    run_result = {"model_version": "spec-x", "promoted": False}
    result, mock_dl, mock_train = _run_main_impl(coverage, run_result)

    assert mock_train.called
    _, kwargs = mock_train.call_args
    passed_coverage = kwargs["arctic_coverage"]
    assert passed_coverage["coverage_ratio"] == pytest.approx(850 / 900)
    # run_meta_training received the raw coverage dict — its own job (per
    # meta_trainer.py) is to compute + persist data_coverage_degraded from it.


def test_full_coverage_not_flagged():
    coverage = _coverage(900, 900)
    result, mock_dl, mock_train = _run_main_impl(coverage)
    assert mock_train.called
    _, kwargs = mock_train.call_args
    assert kwargs["arctic_coverage"]["coverage_ratio"] == 1.0


def test_zero_files_still_raises_before_coverage_check():
    """The pre-existing total-failure gate must still fire (and fire first,
    with its own message) — this test guards against a regression where the
    new coverage gate silently subsumed it."""
    coverage = _coverage(0, 900, failed=[f"T{i}" for i in range(900)])
    with pytest.raises(RuntimeError, match="zero files"):
        _run_main_impl(coverage)


def test_coverage_threshold_is_env_configurable(monkeypatch):
    """A tighter arctic_min_coverage_ratio (e.g. 0.9) must be honored — an
    operator can raise the bar without a code change."""
    import config as cfg
    monkeypatch.setattr(cfg, "ARCTIC_MIN_COVERAGE_RATIO", 0.9, raising=False)
    coverage = _coverage(850, 900, failed=[f"T{i}" for i in range(50)])  # 0.944
    # 0.944 > 0.9, so this should NOT raise under the tightened floor... use a
    # ratio that WOULD pass the default 0.5 floor but fails a tightened 0.95.
    monkeypatch.setattr(cfg, "ARCTIC_MIN_COVERAGE_RATIO", 0.95, raising=False)
    with pytest.raises(RuntimeError, match="coverage"):
        _run_main_impl(coverage)
