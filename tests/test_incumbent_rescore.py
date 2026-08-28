"""The incumbent, scored on TODAY's folds instead of the ones it trained on.

alpha-engine-config-I9024 section 2.

crucible-predictor-PR570 (I9018) fixed WHERE the incumbent's CPCV is read from,
killing the ``x >= x`` tautology. It did not fix WHEN that number was computed:
the registry bundle carries the score the incumbent earned during its own
training run, weeks ago, on that vintage's folds, while every candidate in the
rotation is scored on today's. The comparison was cross-vintage, and unfair in
both directions.

champion-challenger-policy section 7.4: each test names what the pre-fix tree did
instead. On origin/main ``training/incumbent_rescore.py`` does not exist, no
manifest carries ``incumbent_rescore``, and ``serving_ic_basis`` is absent from
every leaderboard — so every assertion here is RED there.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from model.meta_model import MetaModel
from training import incumbent_rescore as ir
from training import model_zoo as mz
from tests.test_model_zoo import _FakeS3

INCUMBENT_VID = "v3.0-meta-2026-08-14-119e069b"
FEATURES = ["f_a", "f_b", "f_c"]


class _BinaryFakeS3:
    """S3 stub that serves raw bytes as well as JSON, so a real pickled
    ``meta_model.pkl`` can live in the fixture bundle."""

    def __init__(self, objects):
        self.objects = dict(objects)
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg names
        if Key not in self.objects:
            raise KeyError(Key)
        val = self.objects[Key]
        body = val if isinstance(val, bytes) else json.dumps(val).encode()
        return {"Body": io.BytesIO(body)}

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.puts[Key] = Body


def _panel(n_dates=30, n_names=40, seed=0):
    """A feature matrix whose label is a clean linear function of ``f_a``."""
    rng = np.random.default_rng(seed)
    n = n_dates * n_names
    X = rng.normal(size=(n, len(FEATURES)))
    y = X[:, 0] * 0.02 + rng.normal(0, 0.002, size=n)
    dates = np.repeat([f"2026-06-{1 + d:02d}" for d in range(n_dates)], n_names)
    return X, y, dates


def _pickled_model(X, y, feature_names=FEATURES):
    import tempfile
    from pathlib import Path

    m = MetaModel(alpha=1.0)
    m.fit(X, y, feature_names=list(feature_names))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "meta_model.pkl"
        m.save(p)
        return p.read_bytes()


def _bundle_s3(model_bytes, *, feature_names=FEATURES, vid=INCUMBENT_VID,
               served_version=INCUMBENT_VID):
    return _BinaryFakeS3({
        cfg.META_MANIFEST_KEY: {"served_version": served_version},
        f"predictor/registry/{vid}/meta_model.pkl": model_bytes,
        f"predictor/registry/{vid}/feature_list.json": {
            "l2_features": list(feature_names)},
    })


@pytest.fixture(autouse=True)
def _bundle_verification_passes(monkeypatch):
    """``verify_bundle`` needs a real ``_lineage.json`` with per-file ETags,
    which a fixture cannot produce. Its own refusal path has a dedicated test
    below; everywhere else it is stubbed to succeed."""
    import model.registry as reg
    monkeypatch.setattr(reg, "verify_bundle", lambda *a, **k: {"ok": True})


def _rescore(s3, X, y, dates, *, train_features=FEATURES):
    return ir.rescore_incumbent_on_current_vintage(
        s3, "bkt", meta_X=X, meta_y=y, row_dates=list(dates),
        train_meta_features=list(train_features),
        forward_days=21, embargo_days=0, n_groups=6, k_test=2,
    )


# ═════════════════════════════════════════════════════════════════════════════
# The incumbent is EVALUATED on this vintage — never refitted to it
# ═════════════════════════════════════════════════════════════════════════════

def test_the_incumbent_is_scored_on_the_current_vintages_folds():
    X, y, dates = _panel()
    out = _rescore(_bundle_s3(_pickled_model(X, y)), X, y, dates)
    assert out["status"] == "ok"
    assert out["incumbent_version_id"] == INCUMBENT_VID
    assert out["fit_mode"] == "frozen_predict_only"
    assert out["mean_ic"] is not None
    assert out["n_combos"] and out["n_combos"] > 0
    assert out["n_groups"] == 6 and out["k_test"] == 2
    assert out["forward_days"] == 21
    assert out["n_rows"] == X.shape[0]
    # A model fit on THIS signal scores positively on it.
    assert out["mean_ic"] > 0.1, out["mean_ic"]


def test_the_weights_are_FROZEN_a_refit_would_score_differently():
    """The decisive test. The bundled model is fit on INVERTED labels, so its
    predictions are anti-correlated with this vintage's ``meta_y``.

    Predict-only ⇒ a NEGATIVE mean IC. Any refit inside the fold loop — which is
    what ``cpcv_meta_oos_ic``'s ``fit_predict_fn`` does for every candidate —
    would recover the signal and score positive. Asserting the sign is asserting
    that the incumbent's own weights, not its architecture, were measured.
    """
    X, y, dates = _panel(seed=3)
    inverted = _pickled_model(X, -y)
    out = _rescore(_bundle_s3(inverted), X, y, dates)
    assert out["status"] == "ok"
    assert out["mean_ic"] < -0.1, out["mean_ic"]


def test_a_feature_reorder_is_not_a_mismatch():
    """The ridge consumes columns positionally, so the same feature set in a
    different order is reindexed rather than refused — otherwise a cosmetic
    change to META_FEATURES would silently disable the re-score forever."""
    X, y, dates = _panel(seed=4)
    reordered = [FEATURES[2], FEATURES[0], FEATURES[1]]
    perm = [FEATURES.index(f) for f in reordered]
    model_bytes = _pickled_model(X[:, perm], y, feature_names=reordered)
    out = _rescore(_bundle_s3(model_bytes, feature_names=reordered), X, y, dates)
    assert out["status"] == "ok"
    assert out["mean_ic"] > 0.1, out["mean_ic"]


# ═════════════════════════════════════════════════════════════════════════════
# Every failure is LABELLED and NON-FATAL
# ═════════════════════════════════════════════════════════════════════════════

def test_a_feature_contract_mismatch_reports_and_does_NOT_raise():
    """Any feature addition creates one. Raising would break every rotation
    after any feature change — a worse failure than an unfair comparison — and
    zero-filling the difference is alpha-engine-config-I5949.
    """
    X, y, dates = _panel(seed=5)
    old_features = ["f_a", "f_b"]            # the incumbent predates f_c
    model_bytes = _pickled_model(X[:, :2], y, feature_names=old_features)
    out = _rescore(_bundle_s3(model_bytes, feature_names=old_features), X, y, dates)
    assert out["status"] == "feature_contract_mismatch"
    assert out["mean_ic"] is None
    assert "f_c" in out["reason"]
    assert "I5949" in out["reason"]
    assert out["incumbent_version_id"] == INCUMBENT_VID


def test_no_incumbent_is_its_own_status_not_an_error():
    X, y, dates = _panel(seed=6)
    s3 = _BinaryFakeS3({cfg.META_MANIFEST_KEY: {"date": "2026-06-13"}})
    out = _rescore(s3, X, y, dates)
    assert out["status"] == "no_incumbent"
    assert out["mean_ic"] is None


def test_an_unverifiable_bundle_is_refused_before_its_weights_are_scored(monkeypatch):
    """alpha-engine-config-I9028 — a drifted bundle must not become the number a
    promotion decision is made against."""
    import model.registry as reg

    def _boom(*a, **k):
        raise reg.RegistryError("etag mismatch: meta_model.pkl")

    monkeypatch.setattr(reg, "verify_bundle", _boom)
    X, y, dates = _panel(seed=7)
    out = _rescore(_bundle_s3(_pickled_model(X, y)), X, y, dates)
    assert out["status"] == "bundle_unverifiable"
    assert out["mean_ic"] is None
    assert "I9028" in out["reason"]


def test_a_missing_bundle_is_an_error_not_a_raise():
    X, y, dates = _panel(seed=8)
    s3 = _BinaryFakeS3({cfg.META_MANIFEST_KEY: {"served_version": INCUMBENT_VID}})
    out = _rescore(s3, X, y, dates)
    assert out["status"] == "error"
    assert out["mean_ic"] is None


# ═════════════════════════════════════════════════════════════════════════════
# select_winner consumes it — and SAYS which comparison happened
# ═════════════════════════════════════════════════════════════════════════════

def _mk(mean_ic, *, rescore=None, forward_days=21):
    m = {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": {"mean_ic": mean_ic, "n_combos": 44},
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": True},
            "overfit": {"passes_overfit_gate": True, "dsr": 1.0},
        },
    }
    if rescore is not None:
        m["incumbent_rescore"] = rescore
    return m


def _board(monkeypatch, *, arch_rescore, stored_ic=0.30, arch_ic=0.20):
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    s3 = _FakeS3({
        cfg.META_MANIFEST_KEY: {"forward_days": 21, "served_version": INCUMBENT_VID},
        f"predictor/registry/{INCUMBENT_VID}/manifest.json": _mk(stored_ic),
        "predictor/registry/arch-v/manifest.json": _mk(arch_ic, rescore=arch_rescore),
    })
    return mz.select_winner(
        s3, "bkt",
        trained=[{"spec_id": "champion-arch", "version_id": "arch-v",
                  "model_version": "v3.0-meta"}],
        margin=0.0, date_str="2026-08-21",
    )


def test_the_rescored_number_becomes_serving_ic_and_is_labelled(monkeypatch):
    """RED pre-fix: ``serving_ic`` was the bundle's stored 0.30 and the
    champion-arch at 0.20 could never refresh, however well it scored on data
    the incumbent has since decayed against."""
    board = _board(monkeypatch, stored_ic=0.30, arch_ic=0.20, arch_rescore={
        "status": "ok", "mean_ic": 0.05,
        "incumbent_version_id": INCUMBENT_VID, "fit_mode": "frozen_predict_only",
    })
    sc = board["serving_champion"]
    assert sc["serving_ic_basis"] == "rescored_current_vintage"
    assert sc["serving_ic_rescored"] == 0.05
    assert sc["serving_ic_stored"] == 0.30
    assert sc["cpcv_mean_ic"] == 0.05              # the number decided on
    assert sc["rescore_source_version_id"] == "arch-v"
    # BOTH numbers survive, and their gap is the model-decay signal.
    assert sc["serving_ic_decay"] == pytest.approx(0.25)
    # …and the comparison the refresh actually won is the honest one.
    assert board["champion_arch_refresh_version_id"] == "arch-v"


def test_a_missing_rescore_falls_back_to_the_stored_number_and_SAYS_so(monkeypatch):
    board = _board(monkeypatch, stored_ic=0.30, arch_ic=0.20, arch_rescore=None)
    sc = board["serving_champion"]
    assert sc["serving_ic_basis"] == "stored_training_vintage"
    assert sc["serving_ic_rescored"] is None
    assert sc["serving_ic_stored"] == 0.30
    assert sc["cpcv_mean_ic"] == 0.30
    assert sc["serving_ic_decay"] is None
    assert "no training run" in sc["serving_ic_basis_reason"]
    # The cross-vintage comparison is the one that happened, and 0.20 < 0.30.
    assert board["champion_arch_refresh_version_id"] is None


def test_a_feature_mismatch_fallback_carries_its_reason_to_the_leaderboard(monkeypatch):
    board = _board(monkeypatch, arch_rescore={
        "status": "feature_contract_mismatch",
        "reason": "incumbent was fit on a different feature set",
        "mean_ic": None, "incumbent_version_id": INCUMBENT_VID,
    })
    sc = board["serving_champion"]
    assert sc["serving_ic_basis"] == "stored_training_vintage"
    assert "feature_contract_mismatch" in sc["serving_ic_basis_reason"]
    assert "different feature set" in sc["serving_ic_basis_reason"]


def test_a_rescore_of_a_DIFFERENT_incumbent_is_refused(monkeypatch):
    """A run that raced a promotion re-scored a model that is no longer serving.
    Using it would compare the candidate against something that is not the
    incumbent."""
    board = _board(monkeypatch, arch_rescore={
        "status": "ok", "mean_ic": 0.01,
        "incumbent_version_id": "some-other-version",
    })
    sc = board["serving_champion"]
    assert sc["serving_ic_basis"] == "stored_training_vintage"
    assert "wrong_incumbent" in sc["serving_ic_basis_reason"]
    assert sc["cpcv_mean_ic"] == 0.30
