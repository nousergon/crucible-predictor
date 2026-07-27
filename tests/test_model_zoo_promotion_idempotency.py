"""Exactly-once model-zoo promotion — idempotency marker (config#2252).

Watch re-runs of the weekly SF re-execute the predictor branch, re-entering the
model-zoo rotation/select entrypoints for the SAME run_date — which re-promoted
and re-sent the identical promotion digest email. The fix: a durable per-run_date
marker at ``predictor/model_zoo/promotions/{run_date}.json``.

Pins:
  - a successfully-finalized rotation WRITES the marker (promoted vid, prior
    champion, decision summary) — for both cutover (promoted) and observe
    (promoted=None) outcomes;
  - a re-run for the same run_date with a MATCHING marker is a verified no-op:
    no selection writes, no re-promote, NO digest email, no inert alert;
  - a marker whose recorded champion MISMATCHES the live registry champion
    RAISES PromotionStateDivergenceError (divergence needs eyes, never a
    silent re-apply) — as does an unreadable (non-not-found) marker;
  - a marker-WRITE failure RAISES (a promotion that can't record itself must
    not silently allow future duplicates);
  - a FAILED promote does NOT write the marker (a re-run must retry it);
  - run_rotation_and_select no-ops BEFORE re-training the rotation.

Reuses the _FakeS3 stub + _mk_manifest helper shape from test_model_zoo.py /
test_model_zoo_parallel.py.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from training import model_zoo as mz

_SPECS = [
    {"id": "resid", "status": "active", "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True}},
    {"id": "h60", "status": "active", "model_version_label": "spec-60d",
     "overrides": {"FORWARD_DAYS": 60}},
]

_DATE = "2026-07-11"
_MARKER_KEY = f"{mz._PROMOTIONS_PREFIX}/{_DATE}.json"


class _FakeS3:
    """Minimal S3 stub mirroring test_model_zoo_parallel.py: get_object reads
    ``objects``, put_object records to ``puts``; missing keys raise KeyError."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = json.dumps(self.objects[Key]).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body


class _MarkerPutBoomS3(_FakeS3):
    """put_object fails ONLY for the promotion-marker key — leaderboard puts
    (which are swallowed as best-effort) still succeed, isolating the pin to
    the marker-write fail-loud contract."""

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        if Key.startswith(mz._PROMOTIONS_PREFIX):
            raise RuntimeError("s3 down for marker put")
        super().put_object(Bucket, Key, Body, ContentType=ContentType)


def _mk_manifest(forward_days, cpcv_ic, gate_pass):
    return {
        "forward_days": forward_days,
        "meta_model_oos_ic_cpcv": {"mean_ic": cpcv_ic} if cpcv_ic is not None else {},
        "meta_model_promotion_stats": {
            "downside": {"passes_downside_gate": gate_pass},
            "overfit": {"passes_overfit_gate": gate_pass},
        },
    }


def _mk_marker(*, promoted, champion_after, mode="cutover"):
    return {
        "schema_version": 1,
        "run_date": _DATE,
        "mode": mode,
        "promoted": promoted,
        "promoted_kind": "challenger" if promoted else None,
        "winner_version_id": promoted,
        "prior_champion_version_id": None,
        "champion_version_id_after": champion_after,
        "written_at_utc": "2026-07-11T10:00:00+00:00",
    }


def _select_fixture(monkeypatch, *, s3=None, objects=None, auto_promote=True,
                    champion_vid=None, promote_boom=False):
    """Registry fixture mirroring test_model_zoo_parallel.py: one challenger
    (resid-v) registered for _DATE; ``champion_vid`` controls what
    list_versions(stage='champion') reports as the live champion. Returns
    (board_fn, promotes, s3, sent, inert) where board_fn runs the select."""
    import model.registry as reg
    monkeypatch.setattr(cfg, "FORWARD_DAYS", 21, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_PROMOTE_MIN_IC", 0.0, raising=False)
    base_objects = {
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.20, True),
    }
    base_objects.update(objects or {})
    if s3 is None:
        s3 = _FakeS3(base_objects)
    versions = [{"version_id": "resid-v", "model_version": "spec-resid", "date": _DATE}]
    champs = [{"version_id": champion_vid}] if champion_vid else []
    monkeypatch.setattr(reg, "list_versions",
                        lambda s3c, b, stage=None: champs if stage == "champion" else versions)
    promotes = []

    def _promote(s3c, b, vid, **k):
        if promote_boom:
            raise RuntimeError("registry bundle incomplete")
        promotes.append(vid)

    monkeypatch.setattr(reg, "promote_to_champion", _promote)
    sent = []
    monkeypatch.setattr(mz, "send_zoo_digest_email",
                        lambda lb, b, d, **k: sent.append(d) or True)
    inert = []
    monkeypatch.setattr(mz, "_alert_inert_rotation", lambda *a, **k: inert.append(k))
    monkeypatch.setattr(mz, "_emit_challengers_trained_metric", lambda n: None)
    monkeypatch.setattr(mz, "_alert_promotion", lambda *a, **k: None)
    monkeypatch.setattr(mz, "_alert_observe_recommendation", lambda *a, **k: None)

    def _run():
        return mz.run_select_only("bkt", date_str=_DATE, s3=s3, specs=_SPECS,
                                  auto_promote_winner=auto_promote)

    return _run, promotes, s3, sent, inert


# ── marker written on a successful finalize ──────────────────────────────────


def test_marker_written_on_promotion_apply(monkeypatch):
    run, promotes, s3, sent, _ = _select_fixture(monkeypatch, auto_promote=True)
    board = run()
    assert board["promoted"] == "resid-v"
    assert promotes == ["resid-v"]
    assert _MARKER_KEY in s3.puts, "promotion marker not written on apply"
    marker = json.loads(s3.puts[_MARKER_KEY])
    assert marker["schema_version"] == 1
    assert marker["run_date"] == _DATE
    assert marker["promoted"] == "resid-v"
    assert marker["promoted_kind"] == "challenger"
    assert marker["champion_version_id_after"] == "resid-v"
    assert marker["registry_bundle_prefix"] == "predictor/registry/resid-v/"
    assert marker["live_weights_prefix"] == "predictor/weights/meta/"
    assert marker["decision_summary"]["n_candidates"] >= 1
    assert marker["written_at_utc"]
    # First run still sends exactly ONE digest.
    assert sent == [_DATE]


def test_marker_written_on_observe_finalize_with_no_promotion(monkeypatch):
    # Observe mode (auto-promote OFF): nothing promoted, but the finalize —
    # including its ONE digest email — still happened; the marker (promoted=
    # None, champion unchanged) makes the re-run email-free too.
    run, promotes, s3, _, _ = _select_fixture(
        monkeypatch, auto_promote=False, champion_vid="champ-0")
    board = run()
    assert board["promoted"] is None
    assert promotes == []
    marker = json.loads(s3.puts[_MARKER_KEY])
    assert marker["promoted"] is None
    assert marker["mode"] == "observe"
    assert marker["champion_version_id_after"] == "champ-0"
    assert marker["registry_bundle_prefix"] is None


# ── re-run with a matching marker → verified no-op, email suppressed ─────────


def test_noop_with_matching_marker_suppresses_email_and_writes(monkeypatch):
    run, promotes, s3, sent, inert = _select_fixture(
        monkeypatch, auto_promote=True, champion_vid="resid-v",
        objects={_MARKER_KEY: _mk_marker(promoted="resid-v", champion_after="resid-v")},
    )
    board = run()
    assert board["idempotent_noop"] is True
    assert board["promoted"] == "resid-v"          # reported FROM the marker
    assert board["digest_email"] == "suppressed_idempotent_noop"
    assert sent == [], "digest email must be suppressed on the idempotent re-run"
    assert promotes == [], "promotion must not be re-applied"
    assert s3.puts == {}, "no writes (leaderboard/trial-log/marker) on a no-op"
    assert inert == [], "empty no-op candidates must not fire the inert alert"


def test_noop_matching_marker_observe_mode_none_champion(monkeypatch):
    # promoted=None + no registry champion at all: None == None is a MATCH.
    run, _, s3, sent, _ = _select_fixture(
        monkeypatch, auto_promote=False, champion_vid=None,
        objects={_MARKER_KEY: _mk_marker(promoted=None, champion_after=None,
                                         mode="observe")},
    )
    board = run()
    assert board["idempotent_noop"] is True
    assert sent == []
    assert s3.puts == {}


# ── marker/state divergence → fail loud ──────────────────────────────────────


def test_mismatched_marker_raises_divergence(monkeypatch):
    # Marker says resid-v became champion, but the registry now serves other-v
    # — somebody changed state since the recorded run. RAISE, never re-apply.
    run, promotes, _, sent, _ = _select_fixture(
        monkeypatch, auto_promote=True, champion_vid="other-v",
        objects={_MARKER_KEY: _mk_marker(promoted="resid-v", champion_after="resid-v")},
    )
    with pytest.raises(mz.PromotionStateDivergenceError, match="diverged"):
        run()
    assert sent == [] and promotes == []


def test_unreadable_marker_raises_not_proceeds(monkeypatch):
    # A non-not-found read failure (e.g. AccessDenied) must RAISE — proceeding
    # blind could duplicate the promotion email.
    class _DeniedS3(_FakeS3):
        def get_object(self, Bucket, Key):  # noqa: N803
            if Key == _MARKER_KEY:
                exc = RuntimeError("denied")
                exc.response = {"Error": {"Code": "AccessDenied"}}
                raise exc
            return super().get_object(Bucket, Key)

    run, _, _, sent, _ = _select_fixture(
        monkeypatch, auto_promote=True, s3=_DeniedS3({
            cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
            "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.20, True),
        }),
    )
    with pytest.raises(mz.PromotionStateDivergenceError, match="could not be read"):
        run()
    assert sent == []


# ── marker-write failure → fail loud ─────────────────────────────────────────


def test_marker_write_failure_raises(monkeypatch):
    boom_s3 = _MarkerPutBoomS3({
        cfg.META_MANIFEST_KEY: _mk_manifest(21, 0.10, True),
        cfg.META_FEATURE_LIST_KEY: {"features": ["a"]},
        "predictor/registry/resid-v/manifest.json": _mk_manifest(21, 0.20, True),
    })
    run, promotes, s3, sent, _ = _select_fixture(
        monkeypatch, auto_promote=True, s3=boom_s3)
    with pytest.raises(RuntimeError, match="s3 down for marker put"):
        run()
    # The promote itself DID land before the marker write blew up (fail loud,
    # not fail silent) and the digest still reported the run via the finally.
    assert promotes == ["resid-v"]
    assert sent == [_DATE]
    assert _MARKER_KEY not in s3.puts


# ── failed promote → NO marker (re-run must retry) ───────────────────────────


def test_no_marker_written_when_promote_fails(monkeypatch):
    run, promotes, s3, sent, _ = _select_fixture(
        monkeypatch, auto_promote=True, promote_boom=True)
    promote_failed = []
    monkeypatch.setattr(mz, "_alert_promote_failure",
                        lambda b, d, vid, kind, exc: promote_failed.append((vid, kind)))
    board = run()
    assert board["promoted"] is None
    assert "promote_error" in board
    assert promotes == []
    assert _MARKER_KEY not in s3.puts, (
        "a FAILED promote must not seal the run_date — a re-run has to retry")
    assert sent == [_DATE]                       # the failure run still reports
    # config#2870: a promote_to_champion exception must not go silent — the
    # only prior trace was board["promote_error"], persisted to S3 and never
    # surfaced to an operator.
    assert promote_failed == [("resid-v", "challenger")]


def test_promote_failure_alert_not_fired_on_success(monkeypatch):
    run, promotes, _, _, _ = _select_fixture(monkeypatch, auto_promote=True)
    promote_failed = []
    monkeypatch.setattr(mz, "_alert_promote_failure",
                        lambda b, d, vid, kind, exc: promote_failed.append((vid, kind)))
    board = run()
    assert board["promoted"] == "resid-v"
    assert promotes == ["resid-v"]
    assert promote_failed == []


# ── run_rotation_and_select no-ops BEFORE re-training ────────────────────────


def test_rotation_entrypoint_noops_before_retraining(monkeypatch):
    import model.registry as reg
    monkeypatch.setattr(reg, "list_versions",
                        lambda s3c, b, stage=None: (
                            [{"version_id": "resid-v"}] if stage == "champion" else []))
    trained = []
    monkeypatch.setattr(mz, "train_weekly_rotation",
                        lambda *a, **k: trained.append(a) or {})
    sent = []
    monkeypatch.setattr(mz, "send_zoo_digest_email",
                        lambda lb, b, d, **k: sent.append(d) or True)
    s3 = _FakeS3({
        _MARKER_KEY: _mk_marker(promoted="resid-v", champion_after="resid-v"),
    })
    board = mz.run_rotation_and_select("bkt", date_str=_DATE, s3=s3, specs=_SPECS)
    assert board["idempotent_noop"] is True
    assert trained == [], "the rotation must NOT retrain on an idempotent re-run"
    assert sent == []
    assert s3.puts == {}
