"""A model-zoo challenger run never writes champion-serving state.

alpha-engine-config-I11478. The weekly rotation trains the champion
architecture and then every model-zoo spec through the same
``train_handler.main``. On rehearsal 2026-09-23 all three runs wrote the same
``predictor/weights/meta/feature_drift_reference.json`` (6 features, then 5,
then 4 — the last challenger won), and the spec runs also overwrote
``health/predictor_training``, ``data_manifest/predictor_training/{date}`` and
the triple-barrier gate's ``variant_gates/`` key.

These tests drive the real write helpers against a recording fake S3 in
challenger mode and assert that nothing under the champion's serving prefix,
and none of the shared training records, is written.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from monitoring.feature_drift import (
    FEATURE_DRIFT_REFERENCE_FILENAME,
    FEATURE_DRIFT_REFERENCE_KEY,
)
from training import train_handler
from training.io_spec import TrainingIOSpec
from training.meta_trainer import _write_feature_drift_reference

_LIVE_PREFIX = "predictor/weights/meta/"
_REPO = Path(__file__).resolve().parent.parent

_REFERENCE = {
    "features": ["momentum_score", "expected_move"],
    "n_samples": 3,
    "trained_date": "2026-09-23",
}


class _RecordingS3:
    def __init__(self):
        self.puts: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.puts[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {}


def _challenger_io(label: str = "spec-sota-combine") -> TrainingIOSpec:
    io = train_handler._io_for_run_role(TrainingIOSpec.live(), label)
    return io.for_run(date_str="2026-09-23", model_version=label)


def _champion_io() -> TrainingIOSpec:
    io = train_handler._io_for_run_role(TrainingIOSpec.live(), None)
    return io.for_run(date_str="2026-09-23", model_version="v3.0-meta")


# ── The run role is decided once, from the spec namespace ────────────────


def test_a_spec_namespace_marks_the_run_a_challenger(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_VERSION_LABEL", "spec-sota-combine", raising=False)
    spec_ns = train_handler._spec_namespace()
    assert spec_ns == "spec-sota-combine"
    io = train_handler._io_for_run_role(TrainingIOSpec.live(), spec_ns)
    assert io.is_challenger_spec
    assert io.challenger_spec == "spec-sota-combine"
    assert io.write_side_artifacts is False
    # Still a real challenger: it trains, registers and may be promoted by
    # selection. Only its SHARED writes are off.
    assert io.register_in_zoo is True
    assert io.allow_live_promote is True


def test_the_champion_architecture_run_keeps_its_shared_writes(monkeypatch):
    monkeypatch.setattr(
        cfg, "MODEL_VERSION_LABEL", train_handler._cfg_default_label(cfg), raising=False,
    )
    assert train_handler._spec_namespace() is None
    io = train_handler._io_for_run_role(TrainingIOSpec.live(), None)
    assert not io.is_challenger_spec
    assert io.write_side_artifacts is True


def test_a_shadow_run_is_not_relabelled_a_challenger():
    shadow = TrainingIOSpec.shadow("crsp")
    assert train_handler._io_for_run_role(shadow, "spec-x") is shadow


def test_for_challenger_spec_refuses_an_empty_namespace():
    with pytest.raises(ValueError):
        TrainingIOSpec.live().for_challenger_spec("")


def test_for_run_keeps_the_challenger_role():
    io = _challenger_io()
    assert io.is_challenger_spec and io.write_side_artifacts is False
    assert io.weights_prefix == (
        "predictor/weights/meta_staging/2026-09-23/spec-sota-combine/"
    )


# ── Feature-drift reference ──────────────────────────────────────────────


def test_challenger_writes_its_drift_reference_only_into_its_own_staging_prefix():
    io = _challenger_io()
    s3 = _RecordingS3()
    written = _write_feature_drift_reference(
        s3, "bkt", _REFERENCE, staging_prefix=io.weights_prefix, io=io,
    )
    assert written == [f"{io.weights_prefix}{FEATURE_DRIFT_REFERENCE_FILENAME}"]
    assert list(s3.puts) == written
    assert not [k for k in s3.puts if k.startswith(_LIVE_PREFIX)], (
        "a model-zoo challenger wrote under the champion's serving prefix — "
        "alpha-engine-config-I11478"
    )


def test_champion_architecture_run_also_refreshes_the_live_reference():
    io = _champion_io()
    s3 = _RecordingS3()
    _write_feature_drift_reference(
        s3, "bkt", _REFERENCE, staging_prefix=io.weights_prefix, io=io,
    )
    assert set(s3.puts) == {
        f"{io.weights_prefix}{FEATURE_DRIFT_REFERENCE_FILENAME}",
        FEATURE_DRIFT_REFERENCE_KEY,
    }
    assert json.loads(s3.puts[FEATURE_DRIFT_REFERENCE_KEY]) == _REFERENCE


def test_shadow_run_never_writes_the_live_reference():
    io = TrainingIOSpec.shadow("crsp")
    s3 = _RecordingS3()
    _write_feature_drift_reference(
        s3, "bkt", _REFERENCE, staging_prefix=io.weights_prefix, io=io,
    )
    assert list(s3.puts) == [f"predictor/weights_shadow/crsp/{FEATURE_DRIFT_REFERENCE_FILENAME}"]


def test_the_staged_reference_lands_at_the_live_key_on_promotion():
    """The staging filename is the live key's basename, so the registry bundle
    (a flat snapshot of the staging prefix) carries it and promote_to_champion,
    which copies every bundle file to ``{live_prefix}{filename}``, lands it at
    exactly the key inference reads."""
    assert FEATURE_DRIFT_REFERENCE_KEY == f"{_LIVE_PREFIX}{FEATURE_DRIFT_REFERENCE_FILENAME}"


def test_no_reference_means_no_write():
    s3 = _RecordingS3()
    assert _write_feature_drift_reference(
        s3, "bkt", None, staging_prefix="p/", io=_champion_io(),
    ) == []
    assert s3.puts == {}


def test_training_never_puts_the_live_reference_key_directly():
    """RED pre-fix: meta_trainer called save_training_reference(_drift_ref,
    bucket=bucket) with the live default key on every run."""
    src = (_REPO / "training/meta_trainer.py").read_text()
    assert "save_training_reference(_drift_ref, bucket=bucket)" not in src


# ── Health record + data manifest ────────────────────────────────────────


def _record_shared_writes(monkeypatch) -> list[str]:
    calls: list[str] = []
    import data_manifest
    import nousergon_lib.health as health

    monkeypatch.setattr(
        health, "write_health", lambda **kw: calls.append(f"health/{kw['module_name']}"),
    )
    monkeypatch.setattr(
        data_manifest, "write_data_manifest",
        lambda **kw: calls.append(f"data_manifest/{kw['module_name']}/{kw['run_date']}"),
    )
    return calls


def test_challenger_does_not_write_the_training_health_or_data_manifest(monkeypatch):
    calls = _record_shared_writes(monkeypatch)
    train_handler._write_run_health_and_manifest({}, "bkt", "2026-09-23", _challenger_io())
    assert calls == []


def test_champion_architecture_writes_the_training_health_and_data_manifest(monkeypatch):
    calls = _record_shared_writes(monkeypatch)
    train_handler._write_run_health_and_manifest({}, "bkt", "2026-09-23", _champion_io())
    assert calls == [
        "health/predictor_training",
        "data_manifest/predictor_training/2026-09-23",
    ]


# ── Every shared write in the handler tail is gated on the one switch ────


def test_each_shared_side_artifact_is_gated_on_write_side_artifacts():
    src = (_REPO / "training/train_handler.py").read_text()
    for marker in (
        "skipping risk_model_persist",
        "skipping triple-barrier cutover gate",
    ):
        idx = src.index(marker)
        assert "if not dry_run and not io.write_side_artifacts:" in src[idx - 400: idx]
    assert "io = _io_for_run_role(io, spec_ns)" in src
