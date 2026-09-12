"""alpha-engine-config-I10259 — the pre-registered out-of-sample commitment
window as an ACTUATOR (the promoter refuses) and a DETECTOR (the served bundle
is graded daily), not as prose in a file nothing reads.

Root cause these pin: `ALPHA_EXPERIMENT_COMMITMENT.yaml` declared
`rotation: suspended` on 2026-09-10 and, on the FIRST weekly rotation after the
freeze (2026-09-12), `ModelZooSelect` auto-promoted
`spec-sota-combine-2026-09-11-753cfbed` straight over the pinned version. A
`git grep -i commitment` over training/, inference/ and model/ returned zero
hits: nothing in this repo had ever read the ruling.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402
from training import commitment as commit  # noqa: E402

FROZEN_VID = "v3.0-meta-2026-08-14-119e069b"
PROMOTED_VID = "spec-sota-combine-2026-09-11-753cfbed"

# The shape of the file on alpha-engine-config main (2026-09-12).
_LIVE_YAML = f"""
freeze_date: 2026-09-10
window_length_sessions: 252
frozen_scope: slot
frozen_version_ids:
  - {FROZEN_VID}
frozen_arm_id: "M:champion-arch:4db81ad0d630"
frozen_arm_ids:
  - {FROZEN_VID}
rotation: suspended
on_void:
  console_render: "VOID — restarted <date>"
  silent_rebase: false
  transition_pages: true
"""


def _write(tmp_path, text: str, name: str = "ALPHA_EXPERIMENT_COMMITMENT.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ── training/commitment.py — the predicate ───────────────────────────────────


class TestPredicate:
    def test_live_file_is_an_active_window_on_and_after_the_freeze_date(self, tmp_path):
        doc = commit.load_commitment(_write(tmp_path, _LIVE_YAML))
        assert commit.frozen_version_ids(doc) == [FROZEN_VID]
        assert commit.window_active(doc, "2026-09-10") is True
        assert commit.window_active(doc, "2026-09-12") is True
        assert commit.window_active(doc, dt.date(2026, 9, 12)) is True

    def test_inactive_before_the_freeze_date(self, tmp_path):
        doc = commit.load_commitment(_write(tmp_path, _LIVE_YAML))
        assert commit.window_active(doc, "2026-09-09") is False

    def test_rotation_active_is_not_a_window(self, tmp_path):
        doc = commit.load_commitment(
            _write(tmp_path, _LIVE_YAML.replace("rotation: suspended", "rotation: active"))
        )
        assert commit.window_active(doc, "2026-09-12") is False

    def test_no_freeze_date_is_not_a_window(self, tmp_path):
        doc = commit.load_commitment(
            _write(tmp_path, _LIVE_YAML.replace("freeze_date: 2026-09-10", ""))
        )
        assert commit.window_active(doc, "2026-09-12") is False

    def test_void_state_is_not_an_active_window(self, tmp_path):
        doc = commit.load_commitment(_write(tmp_path, _LIVE_YAML + "\nstate: VOID\n"))
        assert commit.window_active(doc, "2026-09-12") is False

    def test_absent_state_field_means_not_void(self, tmp_path):
        doc = commit.load_commitment(_write(tmp_path, _LIVE_YAML))
        assert commit.window_state(doc) is None
        assert commit.window_active(doc, "2026-09-12") is True

    def test_absent_file_is_none_not_a_raise(self, tmp_path):
        assert commit.load_commitment(str(tmp_path / "nope.yaml")) is None

    def test_present_but_unparseable_raises_rather_than_reading_as_absent(self, tmp_path):
        path = _write(tmp_path, "freeze_date: [unclosed\n")
        with pytest.raises(commit.CommitmentUnreadable):
            commit.load_commitment(path)

    def test_empty_file_raises(self, tmp_path):
        with pytest.raises(commit.CommitmentUnreadable):
            commit.load_commitment(_write(tmp_path, "\n"))

    def test_none_commitment_is_never_active(self):
        assert commit.window_active(None, "2026-09-12") is False
        assert commit.frozen_version_ids(None) == []

    def test_legacy_frozen_arm_ids_alone_still_reads(self, tmp_path):
        doc = commit.load_commitment(_write(tmp_path, f"""
freeze_date: 2026-09-10
rotation: suspended
frozen_arm_ids:
  - {FROZEN_VID}
"""))
        assert commit.frozen_version_ids(doc) == [FROZEN_VID]

    def test_path_resolution_prefers_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", "/x/y.yaml",
                            raising=False)
        assert commit.commitment_path() == "/x/y.yaml"

    def test_config_default_points_at_the_box_checkout(self):
        # The select step git-pulls this checkout immediately before running.
        assert cfg.ALPHA_EXPERIMENT_COMMITMENT_PATH.endswith(
            "ALPHA_EXPERIMENT_COMMITMENT.yaml"
        )


# ── training/model_zoo.py — the actuator ─────────────────────────────────────


@pytest.fixture()
def _mz(monkeypatch):
    from training import model_zoo as mz

    alerts: list[dict] = []
    monkeypatch.setattr(
        mz, "_alert_commitment_refusal",
        lambda date_str, vid, kind, reason, *, dedup_suffix: alerts.append(
            {"vid": vid, "kind": kind, "reason": reason, "dedup": dedup_suffix}
        ),
    )
    return mz, alerts


class TestActuator:
    def test_refuses_the_2026_09_12_promotion_and_names_the_file(self, _mz, tmp_path,
                                                                monkeypatch):
        mz, alerts = _mz
        path = _write(tmp_path, _LIVE_YAML)
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", path, raising=False)

        out = mz._commitment_refusal("2026-09-12", PROMOTED_VID, "arena-pointer")

        assert out is not None
        assert out["would_have_promoted"] == PROMOTED_VID
        assert out["would_have_been_kind"] == "arena-pointer"
        assert path in out["reason"]
        assert FROZEN_VID in out["reason"]
        assert out["commitment"]["frozen_version_ids"] == [FROZEN_VID]
        assert out["commitment"]["rotation"] == "suspended"
        assert len(alerts) == 1
        assert alerts[0]["dedup"] == PROMOTED_VID

    def test_missing_file_with_auto_promote_on_fails_closed(self, _mz, tmp_path,
                                                            monkeypatch):
        mz, alerts = _mz
        missing = str(tmp_path / "absent.yaml")
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", missing,
                            raising=False)

        out = mz._commitment_refusal("2026-09-12", PROMOTED_VID, "arena-pointer")

        assert out is not None, "a promoter that cannot read its ruling must not promote"
        assert out["commitment"]["status"] == "missing"
        assert missing in out["reason"]
        assert alerts and alerts[0]["dedup"] == "missing"

    def test_unparseable_file_fails_closed(self, _mz, tmp_path, monkeypatch):
        mz, alerts = _mz
        path = _write(tmp_path, "freeze_date: [unclosed\n")
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", path, raising=False)

        out = mz._commitment_refusal("2026-09-12", PROMOTED_VID, "arena-pointer")

        assert out is not None
        assert out["commitment"]["status"] == "unreadable"
        assert alerts and alerts[0]["dedup"] == "unreadable"

    def test_rotation_active_lets_the_promotion_through(self, _mz, tmp_path, monkeypatch):
        mz, alerts = _mz
        path = _write(tmp_path, _LIVE_YAML.replace("rotation: suspended", "rotation: active"))
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", path, raising=False)
        assert mz._commitment_refusal("2026-09-12", PROMOTED_VID, "arena-pointer") is None
        assert alerts == []

    def test_no_freeze_date_lets_the_promotion_through(self, _mz, tmp_path, monkeypatch):
        mz, alerts = _mz
        path = _write(tmp_path, _LIVE_YAML.replace("freeze_date: 2026-09-10", ""))
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", path, raising=False)
        assert mz._commitment_refusal("2026-09-12", PROMOTED_VID, "arena-pointer") is None
        assert alerts == []

    def test_promoting_the_pinned_version_is_not_a_refusal(self, _mz, tmp_path,
                                                           monkeypatch):
        mz, alerts = _mz
        path = _write(tmp_path, _LIVE_YAML)
        monkeypatch.setattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", path, raising=False)
        assert mz._commitment_refusal("2026-09-12", FROZEN_VID, "refit") is None

    def test_the_gate_is_wired_upstream_of_promote_to_champion(self):
        """The predicate existing is not the fix — being CALLED before the
        promote is. Source-order assertion: a refactor that moves the promote
        above the gate fails here rather than in production."""
        import inspect

        from training import model_zoo as mz

        src = inspect.getsource(mz.select_and_finalize)
        assert "_commitment_refusal(" in src, (
            "the commitment gate is not called from the cutover — this is exactly "
            "the 2026-09-12 state (a ruling nothing reads)"
        )
        assert src.index("_commitment_refusal(") < src.index("promote_to_champion"), (
            "the commitment gate must be evaluated BEFORE promote_to_champion"
        )

    def test_the_exit_gate_demote_is_deliberately_not_commitment_gated(self):
        """The realized-edge EXIT gate (I8175) is the ONE path allowed to move
        the pointer inside the window: the commitment file states that "a
        red-line demotion deliberately and loudly VOIDS this window. There is no
        exempt transition." Gating it would turn the kill switch off for a year.
        Pinned so a later sweep does not "fix" the omission."""
        import inspect

        from training import model_zoo as mz

        src = inspect.getsource(mz._apply_realized_edge_demote)
        assert "_commitment_refusal(" not in src


# ── inference/stages/commitment_guard.py — the detector ──────────────────────


class _FakeS3:
    """get_object reads from ``objects``; a missing key raises the boto3-shaped
    NoSuchKey the stage's ``_is_missing`` classifies."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _NoSuchKey(Key)
        body = self.objects[Key]
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        return {"Body": io.BytesIO(body)}

    def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
        self.puts[Key] = Body
        self.objects[Key] = Body


class _NoSuchKey(Exception):
    def __init__(self, key):
        super().__init__(key)
        self.response = {"Error": {"Code": "NoSuchKey"}}


@pytest.fixture()
def _guard(monkeypatch):
    from inference.stages import commitment_guard as cg

    alerts: list[dict] = []
    monkeypatch.setattr(
        cg, "_alert",
        lambda message, *, severity, dedup_key: alerts.append(
            {"message": message, "severity": severity, "dedup_key": dedup_key}
        ),
    )
    return cg, alerts


def _ctx(**kw):
    from inference.pipeline import PipelineContext

    kw.setdefault("date_str", "2026-09-12")
    kw.setdefault("bucket", "alpha-engine-research")
    return PipelineContext(**kw)


def _objects(*, served, commitment_yaml=_LIVE_YAML, extra=None):
    objs = {
        cfg.ALPHA_EXPERIMENT_COMMITMENT_S3_KEY: commitment_yaml.encode(),
        cfg.META_MANIFEST_KEY: {"served_version": served},
    }
    objs.update(extra or {})
    return objs


def _install(monkeypatch, cg, s3):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: s3)


class TestDetector:
    def test_writes_void_and_pages_critical_on_a_mismatch(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(served=PROMOTED_VID))
        _install(monkeypatch, cg, s3)

        cg.run(_ctx())

        key = cfg.COMMITMENT_WINDOW_STATE_KEY
        assert key in s3.puts
        rec = json.loads(s3.puts[key])
        assert rec == {
            "schema_version": 1,
            "state": "VOID",
            "detected_at": rec["detected_at"],
            "served_version": PROMOTED_VID,
            "frozen_version_ids": [FROZEN_VID],
            "freeze_date": "2026-09-10",
            "execution": rec["execution"],
        }
        assert rec["execution"]["date"] == "2026-09-12"
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["dedup_key"] == f"predictor_commitment_window_void_{PROMOTED_VID}"
        assert FROZEN_VID in alerts[0]["message"]

    def test_idempotent_on_a_repeat_day(self, _guard, monkeypatch):
        cg, alerts = _guard
        prior = {
            "schema_version": 1, "state": "VOID",
            "detected_at": "2026-09-12T13:00:00+00:00",
            "served_version": PROMOTED_VID, "frozen_version_ids": [FROZEN_VID],
            "freeze_date": "2026-09-10", "execution": {"date": "2026-09-12"},
        }
        s3 = _FakeS3(_objects(
            served=PROMOTED_VID,
            extra={cfg.COMMITMENT_WINDOW_STATE_KEY: prior},
        ))
        _install(monkeypatch, cg, s3)

        cg.run(_ctx(date_str="2026-09-15"))

        assert cfg.COMMITMENT_WINDOW_STATE_KEY not in s3.puts, (
            "rewriting the record daily erases detected_at, the one field that "
            "dates the breach"
        )
        # Still reports — idempotent on the ARTIFACT, never silent on the condition.
        assert alerts and alerts[0]["severity"] == "critical"

    def test_a_different_served_version_rewrites_the_record(self, _guard, monkeypatch):
        cg, _ = _guard
        prior = {"state": "VOID", "served_version": "some-older-vid",
                 "detected_at": "2026-09-12T13:00:00+00:00"}
        s3 = _FakeS3(_objects(
            served=PROMOTED_VID,
            extra={cfg.COMMITMENT_WINDOW_STATE_KEY: prior},
        ))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx())
        assert cfg.COMMITMENT_WINDOW_STATE_KEY in s3.puts

    def test_intact_window_writes_nothing_and_pages_nothing(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(served=FROZEN_VID))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx())
        assert s3.puts == {}
        assert alerts == []

    def test_absent_commitment_file_warns_rather_than_passing(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3({cfg.META_MANIFEST_KEY: {"served_version": PROMOTED_VID}})
        _install(monkeypatch, cg, s3)

        cg.run(_ctx())

        assert s3.puts == {}
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["dedup_key"] == "predictor_commitment_file_absent"
        assert "UNREPORTED" in alerts[0]["message"]

    def test_no_active_window_is_a_clean_no_op(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(
            served=PROMOTED_VID,
            commitment_yaml=_LIVE_YAML.replace("rotation: suspended", "rotation: active"),
        ))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx())
        assert s3.puts == {}
        assert alerts == []

    def test_manifest_without_served_version_warns(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(served=None))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx())
        assert s3.puts == {}
        assert alerts and alerts[0]["severity"] == "warning"
        assert alerts[0]["dedup_key"] == "predictor_commitment_served_version_missing"

    def test_dry_run_and_local_are_no_ops(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(served=PROMOTED_VID))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx(dry_run=True))
        cg.run(_ctx(local=True))
        assert s3.puts == {}
        assert alerts == []

    def test_shadow_context_is_a_no_op(self, _guard, monkeypatch):
        cg, alerts = _guard
        s3 = _FakeS3(_objects(served=PROMOTED_VID))
        _install(monkeypatch, cg, s3)
        cg.run(_ctx(weights_prefix_override="predictor/registry/some-vid/"))
        assert s3.puts == {}
        assert alerts == []

    def test_a_read_failure_reports_rather_than_raising_into_the_pipeline(
        self, _guard, monkeypatch
    ):
        cg, alerts = _guard

        class _Boom:
            def get_object(self, **kw):
                raise RuntimeError("throttled")

        _install(monkeypatch, cg, _Boom())
        cg.run(_ctx())  # must not raise — trading is never halted by this stage
        assert alerts and alerts[0]["dedup_key"] == "predictor_commitment_guard_failed"

    def test_stage_is_registered_non_critical_in_the_pipeline(self):
        from inference.pipeline import STAGES

        rows = [s for s in STAGES if s[0] == "commitment_guard"]
        assert rows, "the detector is not registered — nothing would run it"
        name, module, critical = rows[0]
        assert module == "inference.stages.commitment_guard"
        assert critical is False, "a commitment measurement must never halt trading"
        names = [s[0] for s in STAGES]
        assert names.index("commitment_guard") > names.index("load_model")
        assert names.index("commitment_guard") < names.index("write_output")
