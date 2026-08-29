"""training/served_slice_dispersion.py — oos_rows key scoping + freshness
(alpha-engine-config-I9378).

WHY THIS FILE EXISTS. `predictor/diagnostics/oos_rows/{date}.parquet` carried
no arm identity, so every live-basis training spec in a rotation (the
champion architecture AND every specialist, e.g. `m-spec-60d`) resolved to
the SAME key. A later same-day specialist run silently overwrote the
champion run's diagnostic — measured 2026-08-14/-21/-28: the object at the
champion's own dated key was 150000x34 spanning 2025-07-10..2026-03-09, a
panel whose last date is 5.5 months before the run that "wrote" it.

This is `served_slice_dispersion.read_oos_panel`'s own consumer, and it feeds
the promotion-time behavioral veto (`training/model_zoo.py::select`) — the
exact class of "a record answering for a different arm" the champion-
challenger-policy calls out (§7.5).
"""
from __future__ import annotations

import io as _io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import served_slice_dispersion as ssd


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = _io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _panel(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "x": range(len(dates))})


class _FakeS3:
    """Minimal stand-in for the ``s3`` client ``read_oos_panel`` takes
    directly (it is not a boto3.client() caller — ``_get_bytes`` is passed
    the client object)."""

    def __init__(self, key_to_bytes: dict):
        self._key_to_bytes = key_to_bytes

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def get_object(self, Bucket, Key):
        if Key not in self._key_to_bytes:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": _io.BytesIO(self._key_to_bytes[Key])}


# ── Key scoping ──────────────────────────────────────────────────────────

def test_model_version_scopes_the_key():
    fresh_dates = [f"2026-08-{d:02d}" for d in range(20, 28)]
    key = f"{ssd._OOS_ROWS_PREFIX}v3.0-meta-2026-08-28-01cf7e1a/2026-08-28.parquet"
    s3 = _FakeS3({key: _parquet_bytes(_panel(fresh_dates))})
    panel, panel_key = ssd.read_oos_panel(
        s3, "bucket", "2026-08-28", "v3.0-meta-2026-08-28-01cf7e1a",
    )
    assert panel_key == key
    assert len(panel) == len(fresh_dates)


def test_champion_and_specialist_keys_for_the_same_date_are_disjoint():
    """The exact bug, reproduced: two live-basis specs, same rotation date,
    must never resolve to the same key."""
    champion_key = f"{ssd._OOS_ROWS_PREFIX}v3.0-meta-2026-08-28-01cf7e1a/2026-08-28.parquet"
    specialist_key = f"{ssd._OOS_ROWS_PREFIX}m-spec-60d-2026-08-28-50bb8bdf/2026-08-28.parquet"
    assert champion_key != specialist_key

    fresh_dates = [f"2026-08-{d:02d}" for d in range(20, 28)]
    s3 = _FakeS3({
        champion_key: _parquet_bytes(_panel(fresh_dates)),
        specialist_key: _parquet_bytes(_panel(fresh_dates).assign(specialist_only=1)),
    })
    champ_panel, _ = ssd.read_oos_panel(
        s3, "bucket", "2026-08-28", "v3.0-meta-2026-08-28-01cf7e1a",
    )
    spec_panel, _ = ssd.read_oos_panel(
        s3, "bucket", "2026-08-28", "m-spec-60d-2026-08-28-50bb8bdf",
    )
    assert "specialist_only" not in champ_panel.columns
    assert "specialist_only" in spec_panel.columns


def test_no_model_version_falls_back_to_the_legacy_unscoped_prefix():
    """Reading a panel written before this fix landed — the pre-fix bare
    prefix must still resolve."""
    fresh_dates = [f"2026-08-{d:02d}" for d in range(20, 28)]
    key = f"{ssd._OOS_ROWS_PREFIX}2026-08-28.parquet"
    s3 = _FakeS3({key: _parquet_bytes(_panel(fresh_dates))})
    panel, panel_key = ssd.read_oos_panel(s3, "bucket", "2026-08-28", None)
    assert panel_key == key


# ── Freshness verification ──────────────────────────────────────────────

def test_a_5_month_stale_panel_is_refused_not_silently_answered():
    """The measured incident: 150000x34 spanning 2025-07-10..2026-03-09 read
    against the 2026-08-28 champion run — must raise, never return."""
    stale_dates = [f"2026-0{m}-01" for m in range(1, 4)]  # 2026-01..2026-03
    key = f"{ssd._OOS_ROWS_PREFIX}v3.0-meta/2026-08-28.parquet"
    s3 = _FakeS3({key: _parquet_bytes(_panel(stale_dates))})
    with pytest.raises(RuntimeError, match="no readable"):
        ssd.read_oos_panel(s3, "bucket", "2026-08-28", "v3.0-meta")


def test_verify_oos_panel_freshness_raises_directly_on_a_stale_panel():
    stale_dates = [f"2026-0{m}-01" for m in range(1, 4)]
    with pytest.raises(RuntimeError, match="STALE"):
        ssd._verify_oos_panel_freshness(
            _panel(stale_dates), "some/key.parquet", "2026-08-28",
        )


def test_verify_oos_panel_freshness_passes_a_fresh_panel():
    fresh_dates = [f"2026-08-{d:02d}" for d in range(20, 28)]
    ssd._verify_oos_panel_freshness(
        _panel(fresh_dates), "some/key.parquet", "2026-08-28",
    )  # must not raise


def test_verify_oos_panel_freshness_is_a_noop_without_an_expected_date():
    """No expected date to compare against — never a false-positive raise on
    an unmeasured claim (champion-challenger-policy §5.1)."""
    stale_dates = [f"2026-0{m}-01" for m in range(1, 4)]
    ssd._verify_oos_panel_freshness(
        _panel(stale_dates), "some/key.parquet", None,
    )  # must not raise


def test_served_slice_metrics_reports_uncomputable_not_a_pass_on_a_stale_panel():
    """The RAISE from read_oos_panel must surface as `uncomputable`, never
    propagate as an unhandled crash of a training run — the non-blocking
    posture champion-challenger-policy §5.1 requires of every gate here."""
    stale_dates = [f"2026-0{m}-01" for m in range(1, 4)]
    key = f"{ssd._OOS_ROWS_PREFIX}v3.0-meta/2026-08-28.parquet"
    s3 = _FakeS3({key: _parquet_bytes(_panel(stale_dates))})
    out = ssd.served_slice_metrics(
        s3, "bucket", ["v1", "v2"], date_str="2026-08-28", model_version="v3.0-meta",
    )
    assert out["status"] == "uncomputable"
    assert "STALE" in out["reason"] or "no readable" in out["reason"]
