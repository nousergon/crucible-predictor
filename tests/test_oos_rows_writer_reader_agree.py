"""The OOS panel select reads is the one training wrote.

alpha-engine-config-I11477. On rehearsal 2026-09-23 training wrote

    predictor/diagnostics/oos_rows/v3.0-meta/2026-09-23.parquet

keyed by the training manifest's ``version`` (the arm label), while model-zoo
select looked for

    predictor/diagnostics/oos_rows/v3.0-meta-2026-09-23-d7f8f864/2026-09-23.parquet

keyed by the champion-arch's registry ``version_id``. No writer has ever
produced that key, so the I9061 served-slice dispersion was UNCOMPUTABLE on
every rotation. Both sides now resolve the arm through
``training.io_spec.oos_rows_arm`` and build the key through
``TrainingIOSpec.oos_rows_key``; these tests build the writer's key and the
reader's key from that one function and require them to be equal.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz
from training import served_slice_dispersion as ssd
from training.io_spec import TrainingIOSpec, oos_rows_arm
from tests.test_model_zoo import _FakeS3
from tests.test_promotion_served_slice_veto import INCUMBENT_VID, _manifest

_REPO = Path(__file__).resolve().parent.parent
DATE = "2026-09-23"
CHAMP_ARCH_VID = "v3.0-meta-2026-09-23-d7f8f864"


class _PanelS3:
    def __init__(self, objects: dict):
        self._objects = objects

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self._objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self._objects[Key])}


def _panel_bytes() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"date": ["2026-09-21", "2026-09-22", DATE], "x": [1, 2, 3]}).to_parquet(
        buf, index=False)
    return buf.getvalue()


def _writer_key(manifest: dict) -> str:
    """Exactly what meta_trainer does before its put_object."""
    return TrainingIOSpec.live().oos_rows_key(DATE, oos_rows_arm(manifest))


def _select_and_capture(monkeypatch, champ_manifest: dict, trained_rec: dict) -> dict:
    captured: dict = {}

    def _fake_served_slice(s3, bucket, vids, **kw):
        captured["called"] = True
        captured.update(kw)
        return {"status": "uncomputable", "reason": "stub", "panel_key": None,
                "metrics": {}, "errors": {}}

    monkeypatch.setattr(ssd, "served_slice_metrics", _fake_served_slice)
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: {"forward_days": 21, "served_version": INCUMBENT_VID},
        f"predictor/registry/{INCUMBENT_VID}/manifest.json": _manifest(mean_ic=0.13),
        f"predictor/registry/{CHAMP_ARCH_VID}/manifest.json": champ_manifest,
    })
    board = mz.select_winner(s3, "bkt", trained=[trained_rec], margin=0.0, date_str=DATE)
    captured["board"] = board
    return captured


def test_select_reads_the_key_training_wrote(monkeypatch):
    champ_manifest = dict(_manifest(mean_ic=0.0776), version="v3.0-meta")
    got = _select_and_capture(
        monkeypatch, champ_manifest,
        {"spec_id": "champion-arch", "version_id": CHAMP_ARCH_VID, "model_version": "v3.0-meta"},
    )
    assert got["model_version"] == "v3.0-meta"

    written = _writer_key(champ_manifest)
    assert written == f"predictor/diagnostics/oos_rows/v3.0-meta/{DATE}.parquet"
    panel, read_key = ssd.read_oos_panel(
        _PanelS3({written: _panel_bytes()}), "bkt", DATE, got["model_version"],
    )
    assert read_key == written
    assert len(panel) == 3


def test_the_registry_version_id_is_never_the_arm(monkeypatch):
    """RED pre-fix: select passed champ_arch_vid, so the reader looked under
    ``oos_rows/v3.0-meta-2026-09-23-d7f8f864/``, which no writer produces."""
    champ_manifest = dict(_manifest(mean_ic=0.0776), version="v3.0-meta")
    got = _select_and_capture(
        monkeypatch, champ_manifest,
        {"spec_id": "champion-arch", "version_id": CHAMP_ARCH_VID, "model_version": "v3.0-meta"},
    )
    assert got["model_version"] != CHAMP_ARCH_VID


def test_the_latest_alias_agrees_too():
    manifest = {"version": "spec-sota-combine"}
    writer = TrainingIOSpec.live().oos_rows_latest_key(oos_rows_arm(manifest))
    s3 = _PanelS3({writer: _panel_bytes()})
    _, read_key = ssd.read_oos_panel(s3, "bkt", None, oos_rows_arm(manifest))
    assert read_key == writer == "predictor/diagnostics/oos_rows/spec-sota-combine/latest.parquet"


def test_an_unlabelled_champion_arch_is_uncomputable_not_a_guess(monkeypatch):
    got = _select_and_capture(
        monkeypatch, _manifest(mean_ic=0.0776),  # no "version"
        {"spec_id": "champion-arch", "version_id": CHAMP_ARCH_VID},
    )
    assert "called" not in got, "must not read an unscoped/legacy panel"
    block = got["board"]["served_slice_dispersion"]
    assert block["status"] == "uncomputable"
    assert "I11477" in block["reason"]


def test_oos_rows_arm_reads_only_the_manifest_version():
    assert oos_rows_arm({"version": "v3.0-meta", "version_id": "x"}) == "v3.0-meta"
    assert oos_rows_arm({}) is None
    assert oos_rows_arm(None) is None


def test_trainer_resolves_its_arm_through_the_shared_function():
    src = (_REPO / "training/meta_trainer.py").read_text()
    assert "oos_rows_arm(manifest)" in src
    assert 'manifest.get("version") or getattr(' not in src
