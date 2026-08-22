"""alpha-engine-config-I8175 / -I8195 — the model-zoo EXIT and ENTRANCE gates and
the one state they share.

The two issues were split as tracker items, never as designs: 8175 is the exit
(demote a champion shown to have no realized edge), 8195 is the entrance (stop
admitting one), and both answer *what does the system do when no arm is good
enough?* If the demote-TO state and the no-arm-clears state can differ, the system
can demote into a state its own promotion gate would have refused. These tests pin
that they cannot.

They also pin the correctness that makes the exit gate act on the right number:
the legacy realized read is an unattributed SLOT aggregate over a 42-day window
while the champion rotates weekly, so under a 21d horizon the matured subset never
contains a single prediction from the version currently serving. Acting on it
would demote a champion for its predecessors' results.
"""
from __future__ import annotations

import json

import pytest

import config as cfg
import training.model_zoo_gates as gates
from analysis.observe_leaderboard import (
    _attributed_realized_rank_ic,
    _realized_rank_ic_by_version,
)


# ── the single shared configuration point ───────────────────────────────────

def _configure(monkeypatch, *, state=None, floor=None, consecutive=None, enable=True,
               scope="version"):
    monkeypatch.setattr(
        cfg, "MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE", scope, raising=False
    )
    monkeypatch.setattr(cfg, "MODEL_ZOO_NO_GOOD_ARM_STATE", state, raising=False)
    monkeypatch.setattr(cfg, "MODEL_ZOO_REALIZED_EDGE_FLOOR", floor, raising=False)
    monkeypatch.setattr(
        cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE", consecutive, raising=False
    )
    monkeypatch.setattr(
        cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE", enable, raising=False
    )


def test_no_good_arm_state_is_unset_by_default_and_never_guessed():
    """The shipped default must be UNSET. An auto-demote with an invented fallback
    moves live capital on a value nobody ruled."""
    assert getattr(cfg, "MODEL_ZOO_NO_GOOD_ARM_STATE", "MISSING") is None
    assert getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_FLOOR", "MISSING") is None
    assert getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE", "MISSING") is None
    assert getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE", "MISSING") is False
    assert getattr(cfg, "MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE", "MISSING") is None


def test_resolve_no_good_arm_state_raises_when_unset(monkeypatch):
    _configure(monkeypatch, state=None)
    with pytest.raises(gates.NoGoodArmStateUnset) as exc:
        gates.resolve_no_good_arm_state()
    assert "reserved operator decision" in str(exc.value)


def test_resolve_no_good_arm_state_rejects_an_unknown_state(monkeypatch):
    _configure(monkeypatch, state="go_short")
    with pytest.raises(gates.NoGoodArmStateUnset):
        gates.resolve_no_good_arm_state()


@pytest.mark.parametrize("state", gates.NO_GOOD_ARM_STATES)
def test_resolve_no_good_arm_state_accepts_each_legal_state(monkeypatch, state):
    _configure(monkeypatch, state=state)
    assert gates.resolve_no_good_arm_state() == state


def test_realized_edge_floor_and_hysteresis_raise_when_unset(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=None, consecutive=None)
    with pytest.raises(gates.RealizedEdgeFloorUnset):
        gates.resolve_realized_edge_floor()
    monkeypatch.setattr(cfg, "MODEL_ZOO_REALIZED_EDGE_FLOOR", 0.0, raising=False)
    with pytest.raises(gates.RealizedEdgeFloorUnset):
        gates.resolve_demote_consecutive()


# ── the EXIT gate ───────────────────────────────────────────────────────────

def _payload(ic, *, vid="v-serving", status="attributed", n=40, day="2026-08-21",
             line_ic="__same__"):
    return {
        "trading_day": day,
        "serving_champion_attributed": {
            "version_id": vid,
            "realized_rank_ic": ic,
            "n_matured_outcomes": n,
            "attribution_status": status,
        },
        "serving_line_attributed": {
            "line": "v3.0-meta",
            "realized_rank_ic": ic if line_ic == "__same__" else line_ic,
            "n_matured_outcomes": n,
            "attribution_status": status,
        },
    }


def test_exit_gate_is_disabled_by_default(monkeypatch):
    _configure(monkeypatch, enable=False)
    v = gates.evaluate_realized_edge_exit([_payload(-0.9)])
    assert v["status"] == "disabled"


def test_armed_gate_with_unset_state_raises_rather_than_defaulting(monkeypatch):
    """An armed gate whose fallback nobody ruled must FAIL, not pick one."""
    _configure(monkeypatch, state=None, floor=0.0, consecutive=2, enable=True)
    with pytest.raises(gates.NoGoodArmStateUnset):
        gates.evaluate_realized_edge_exit([_payload(-0.9)])


def test_armed_gate_refuses_to_arm_on_a_state_with_no_actuator(monkeypatch):
    """`flat` and `reduced_size` need an executor-side contract this repo does not
    have. The gate refuses to ARM rather than firing and then failing to act."""
    for state in ("flat", "reduced_size"):
        _configure(monkeypatch, state=state, floor=0.0, consecutive=1, enable=True)
        with pytest.raises(gates.NoGoodArmActuatorMissing) as exc:
            gates.evaluate_realized_edge_exit([_payload(-0.9)])
        assert "does not size positions" in str(exc.value)


def test_unmeasurable_is_never_rendered_as_a_pass(monkeypatch):
    """champion-challenger-policy §5.1 — you cannot gate on a statistic you did
    not measure, and an uncomputed gate reported as a PASS is the defect."""
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=1)
    v = gates.evaluate_realized_edge_exit(
        [_payload(None, status="no_matured_outcomes", n=0)]
    )
    assert v["status"] == "unmeasurable"
    assert v["status"] != "pass"


def test_short_history_is_unmeasurable_not_a_demote(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=3)
    v = gates.evaluate_realized_edge_exit([_payload(-0.4), _payload(-0.3, day="d2")])
    assert v["status"] == "unmeasurable"


def test_attributed_edge_above_the_floor_passes(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=2)
    v = gates.evaluate_realized_edge_exit([_payload(0.12), _payload(-0.4, day="d2")])
    assert v["status"] == "pass"


def test_one_bad_reading_arms_but_does_not_demote(monkeypatch):
    """Hysteresis (champion-challenger-policy §5.2) — a sitting champion is not
    demoted on a single reading, or the loop oscillates on noise."""
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=3)
    v = gates.evaluate_realized_edge_exit(
        [_payload(-0.20), _payload(0.05, day="d2"), _payload(-0.10, day="d3")]
    )
    assert v["status"] == "arming"
    assert v["consecutive_below"] == 2


def test_consecutive_readings_below_the_floor_fire_the_demote(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=2)
    v = gates.evaluate_realized_edge_exit([_payload(-0.20), _payload(-0.10, day="d2")])
    assert v["status"] == "demote"
    assert v["demote_to"] == "hold_incumbent"


# ── the ENTRANCE gate, and consistency with the exit ────────────────────────

def test_zero_promote_min_ic_is_recorded_as_a_sign_check_not_a_bar(monkeypatch):
    """`promote_min_ic: 0.0` recorded faithfully still left a gated and an ungated
    promotion indistinguishable in the artifact. It must say which it is."""
    _configure(monkeypatch, state=None, floor=None, consecutive=None, enable=False)
    rec = gates.evaluate_absolute_bar(
        promote_min_ic=0.0, candidate_cpcv_ic=0.3056,
        slot_realized_rank_ic=None, slot_attribution_status=None,
    )
    assert rec["candidate_bar"]["is_gating"] is False
    assert rec["slot_bar"]["status"] == "unset"
    assert rec["gated"] is False


def test_absolute_bar_recorder_never_raises_on_unruled_config(monkeypatch):
    """It runs on the artifact-write path; the ACTING gates raise, this records."""
    _configure(monkeypatch, state=None, floor=None, consecutive=None, enable=False)
    rec = gates.evaluate_absolute_bar(
        promote_min_ic=0.0, candidate_cpcv_ic=0.1,
        slot_realized_rank_ic=-0.26, slot_attribution_status="attributed",
    )
    assert rec["slot_bar"]["is_gating"] is False


def test_unmeasurable_slot_does_not_gate_the_entrance(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=1)
    rec = gates.evaluate_absolute_bar(
        promote_min_ic=0.0, candidate_cpcv_ic=0.1,
        slot_realized_rank_ic=None, slot_attribution_status="no_matured_outcomes",
    )
    assert rec["slot_bar"]["status"] == "unmeasurable"
    assert rec["slot_bar"]["is_gating"] is False


@pytest.mark.parametrize("state", sorted(gates.ACTUATOR_IMPLEMENTED))
def test_entrance_and_exit_resolve_THE_SAME_state(monkeypatch, state):
    """The consistency invariant both issues turn on. The exit's demote-TO state
    and the entrance's no-arm-clears state are ONE symbol; if they could differ the
    system could demote into something promotion would have refused."""
    _configure(monkeypatch, state=state, floor=0.0, consecutive=1, enable=True)

    exit_verdict = gates.evaluate_realized_edge_exit([_payload(-0.26)])
    entrance = gates.evaluate_absolute_bar(
        promote_min_ic=0.0, candidate_cpcv_ic=0.30,
        slot_realized_rank_ic=-0.26, slot_attribution_status="attributed",
    )

    assert exit_verdict["status"] == "demote"
    assert entrance["slot_bar"]["clears"] is False
    # Same state, and the same floor produced both verdicts from the same number.
    assert exit_verdict["demote_to"] == entrance["slot_bar"]["no_good_arm_state"] == state
    assert exit_verdict["floor"] == entrance["slot_bar"]["floor"]


# ── attribution: the number the exit gate is allowed to read ────────────────

def _pairs(vid, ic_sign, n, *, matured=True, date="2026-07-20"):
    """n pairs for `vid` whose predicted/realized ranks agree (+) or invert (-)."""
    out = []
    for i in range(n):
        out.append({
            "date": date,
            "ticker": f"T{i}",
            "predicted_alpha": float(i),
            "realized_alpha": (float(i) * ic_sign) if matured else None,
            "champion_version_id": vid,
        })
    return out


def test_unstamped_predictions_are_reported_not_folded_into_the_serving_version():
    """Artifacts written before the stamp shipped must NOT have their aggregate
    silently attributed to whoever is serving now."""
    pairs = _pairs(None, -1.0, 30)
    res = _attributed_realized_rank_ic(pairs, "v3.0-meta-2026-08-14-119e069b")
    assert res["attribution_status"] == "unstamped_predictions"
    assert res["realized_rank_ic"] is None


def test_a_freshly_promoted_champion_with_no_matured_outcomes_is_not_a_verdict():
    """THE regression guard (alpha-engine-config-I8175). Measured 2026-08-22: the
    -0.2626 that raised the issue belongs to spec-residual-mom-2026-07-17, while
    the serving v3.0-meta-2026-08-14 had ZERO matured outcomes of its own. Under
    weekly rotation against a 21d horizon that is the EXPECTED state, and the exit
    gate must not read it as no edge.

    Without the attributed read this returns the aggregate's strongly negative
    number under the serving champion's name — a demote for a predecessor's
    results (champion-challenger-policy §7.5)."""
    predecessor = _pairs("spec-residual-mom-2026-07-17", -1.0, 30, date="2026-07-20")
    serving = _pairs(
        "v3.0-meta-2026-08-14", 1.0, 20, matured=False, date="2026-08-18"
    )
    pairs = predecessor + serving

    res = _attributed_realized_rank_ic(pairs, "v3.0-meta-2026-08-14")
    assert res["attribution_status"] == "no_matured_outcomes"
    assert res["realized_rank_ic"] is None, (
        "the serving champion has no matured outcomes — reporting the predecessor's "
        "negative aggregate under its name is the defect this guards"
    )

    # PRE-FIX behaviour, pinned so the guard is shown to fail without the change
    # (champion-challenger-policy §7.4): the unattributed aggregate — exactly what
    # `champion.realized_rank_ic` reported and what a naive demote gate would have
    # consumed — is strongly NEGATIVE over the same pairs.
    from analysis.observe_leaderboard import _realized_rank_ic as _aggregate

    aggregate_ic, aggregate_n, _ = _aggregate(pairs)
    assert aggregate_ic is not None and aggregate_ic < -0.5
    assert aggregate_n == 30, "every matured pair belongs to the PREDECESSOR"

    # And the per-version breakdown makes the real owner of that number visible.
    by_version = {r["version_id"]: r for r in _realized_rank_ic_by_version(pairs)}
    assert by_version["spec-residual-mom-2026-07-17"]["realized_rank_ic"] < 0
    assert by_version["v3.0-meta-2026-08-14"]["realized_rank_ic"] is None


def test_a_serving_champion_with_its_own_matured_outcomes_is_attributed():
    pairs = _pairs("v-serving", 1.0, 30) + _pairs("v-old", -1.0, 30)
    res = _attributed_realized_rank_ic(pairs, "v-serving")
    assert res["attribution_status"] == "attributed"
    assert res["realized_rank_ic"] > 0


def test_unresolvable_serving_version_is_version_unknown():
    res = _attributed_realized_rank_ic(_pairs("v-serving", 1.0, 30), None)
    assert res["attribution_status"] == "version_unknown"
    assert res["realized_rank_ic"] is None


# ── the stamp that makes attribution possible at all ───────────────────────

def test_predictions_artifact_carries_the_champion_version_id(monkeypatch):
    """alpha-engine-config-I8175 — `model_version` is an architecture literal
    ("meta-v3.0-8models") identical across every rotation, so nothing downstream
    could tell which arm produced a batch. The registry version_id must be on the
    artifact; without it the realized read can only pool the whole slot, and the
    exit gate would act on a predecessor's number."""
    import boto3

    import inference.stages.write_output as wo

    written: dict = {}

    monkeypatch.setattr(
        wo, "_resolve_champion_version_id",
        lambda bucket: "v3.0-meta-2026-08-21-7d3d1cce",
    )
    monkeypatch.setattr(
        wo, "_s3_put_json",
        lambda s3, bucket, key, body: written.setdefault(key, body),
    )
    monkeypatch.setattr(boto3, "client", lambda *a, **k: object())

    wo.write_predictions(
        [{"ticker": "AAA", "predicted_alpha": 0.01, "prediction_confidence": 0.9,
          "p_up": 0.6}],
        "2026-08-21", "test-bucket", {"model_version": "meta-v3.0-8models"},
    )

    raw = written["predictor/predictions/2026-08-21.json"]
    envelope = json.loads(raw) if isinstance(raw, str) else raw
    assert envelope["champion_version_id"] == "v3.0-meta-2026-08-21-7d3d1cce"
    # The architecture literal is still there and still stale — which is exactly
    # why the version_id had to be added rather than the literal reinterpreted.
    assert envelope["model_version"] == "meta-v3.0-8models"


def test_champion_version_id_resolver_returns_none_not_a_stale_literal(monkeypatch):
    """An unreadable registry yields honest absence. A hardcoded architecture
    literal in its place is the §7.5 defect this replaced."""
    import inference.stages.write_output as wo

    monkeypatch.setattr(
        "model.registry.list_versions",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("registry down")),
    )
    assert wo._resolve_champion_version_id("test-bucket") is None


# ── attribution SCOPE: the cadence arithmetic that decides whether the gate can
#    fire at all ───────────────────────────────────────────────────────────────

def test_armed_gate_with_unset_attribution_scope_raises(monkeypatch):
    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=1,
               scope=None)
    with pytest.raises(gates.RealizedEdgeFloorUnset) as exc:
        gates.evaluate_realized_edge_exit([_payload(-0.26)])
    assert "VERSION or the serving LINE" in str(exc.value)


def test_version_scope_reads_the_version_number_line_scope_reads_the_line(monkeypatch):
    """The two scopes must be able to disagree, or the config point is decorative."""
    payloads = [_payload(None, status="no_matured_outcomes", n=0, line_ic=-0.26)]

    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=1,
               scope="version")
    v = gates.evaluate_realized_edge_exit(payloads)
    assert v["status"] == "unmeasurable", (
        "the serving VERSION has no matured outcomes — under weekly rotation "
        "against a 21d horizon this is the permanent state, and a version-scoped "
        "gate correctly never fires rather than firing on someone else's number"
    )

    _configure(monkeypatch, state="hold_incumbent", floor=0.0, consecutive=1,
               scope="line")
    v = gates.evaluate_realized_edge_exit(payloads)
    assert v["status"] == "demote"
    assert v["attribution_scope"] == "line"


def test_champion_line_of_strips_the_dated_version_suffix():
    from analysis.observe_leaderboard import champion_line_of

    assert champion_line_of("v3.0-meta-2026-08-14-119e069b") == "v3.0-meta"
    assert champion_line_of("spec-residual-mom-2026-07-17-f478ece3") == "spec-residual-mom"
    assert champion_line_of(None) is None


def test_line_attribution_pools_every_version_on_the_line():
    """alpha-engine-config-I8175 — measured 2026-08-22, every v3.0-meta champion
    from 2026-07-24 onward had zero matured outcomes of its own while the LINE had
    been serving continuously. The line read is what makes the gate firable."""
    from analysis.observe_leaderboard import _attributed_line_realized_rank_ic

    pairs = (
        _pairs("v3.0-meta-2026-07-24-aaa", 1.0, 15, date="2026-07-24")
        + _pairs("v3.0-meta-2026-07-31-bbb", 1.0, 15, date="2026-07-31")
        + _pairs("spec-residual-mom-2026-07-17-ccc", -1.0, 30, date="2026-07-17")
    )
    res = _attributed_line_realized_rank_ic(pairs, "v3.0-meta-2026-08-14-119e069b")
    assert res["attribution_status"] == "attributed"
    assert res["line"] == "v3.0-meta"
    assert res["n_versions"] == 2
    assert res["realized_rank_ic"] > 0, "the spec-residual-mom line must not leak in"
