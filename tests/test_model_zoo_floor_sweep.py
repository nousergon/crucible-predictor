"""alpha-engine-config-I8195 deliverable 1 — the observe-mode floor sweep.

The entrance and exit gates (crucible-predictor-PR548) are built, correct, and
inert: five reserved values stand between them and arming. Four are a genuine
operator decision. The fifth, `MODEL_ZOO_REALIZED_EDGE_FLOOR`, is a NUMBER, and
-I8195 deliverable 1 is explicit that nominating one without measurement is
"a guess wearing a number".

This module tests the measurement that answers it: on every rotation, what BOTH
gates WOULD have decided across a range of candidate floors, both attribution
scopes and each hysteresis count — from the same attributed readings the armed
gates read.

The load-bearing property is the LAST test: it cannot act.
"""
from __future__ import annotations

import inspect

import pytest

from training.model_zoo_gates import (
    ATTRIBUTION_SCOPES,
    CONSECUTIVE_SWEEP,
    FLOOR_SWEEP,
    evaluate_floor_sweep,
)


def _payload(trading_day: str, *, version_ic=None, line_ic=None) -> dict:
    p: dict = {"trading_day": trading_day}
    if version_ic is not None:
        p["serving_champion_attributed"] = {
            "version_id": f"v-{trading_day}",
            "realized_rank_ic": version_ic,
            "n_matured_outcomes": 120,
            "attribution_status": "attributed",
        }
    if line_ic is not None:
        p["serving_line_attributed"] = {
            "line": "v3.0-meta",
            "realized_rank_ic": line_ic,
            "n_matured_outcomes": 225,
            "attribution_status": "attributed",
        }
    return p


# ── Shape ────────────────────────────────────────────────────────────────────


def test_sweep_covers_both_scopes_and_declares_observe_mode():
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=-0.26)])
    assert out["mode"] == "observe"
    assert out["acts"] is False
    assert set(out["scopes"]) == set(ATTRIBUTION_SCOPES)
    assert out["floors_swept"] == list(FLOOR_SWEEP)
    assert out["consecutive_swept"] == list(CONSECUTIVE_SWEEP)


def test_unmeasurable_scope_is_never_a_pass():
    """The version scope has no attributed reading in practice (weekly rotation
    against a 21d horizon). It must report unmeasurable, not clear."""
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=-0.26)])
    version = out["scopes"]["version"]
    assert version["status"] == "unmeasurable"
    assert version["grid"] == []
    assert "never rendered as a pass" in version["reason"]
    # And the measured scope IS measured, so the two are distinguishable.
    assert out["scopes"]["line"]["status"] == "measured"


def test_reserved_values_outstanding_is_recorded():
    """The artifact must answer 'what is this waiting on?' without a reader
    opening config.py."""
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=-0.26)])
    outstanding = out["reserved_values_outstanding"]
    assert "MODEL_ZOO_REALIZED_EDGE_FLOOR" in outstanding
    assert "MODEL_ZOO_NO_GOOD_ARM_STATE" in outstanding
    assert "MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE" in outstanding


# ── The counterfactuals ──────────────────────────────────────────────────────


def test_a_clearly_negative_reading_trips_every_floor_at_hysteresis_one():
    """The live 2026-08-21 line reading, -0.2626, sits below every swept floor."""
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=-0.2626)])
    for row in out["scopes"]["line"]["grid"]:
        assert row["entrance_would_refuse"] is True, row["floor"]
        assert row["exit_would_demote"]["1"] is True, row["floor"]


def test_a_positive_reading_trips_only_the_floors_above_it():
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=0.03)])
    grid = {r["floor"]: r for r in out["scopes"]["line"]["grid"]}
    assert grid[-0.05]["entrance_would_refuse"] is False
    assert grid[0.0]["entrance_would_refuse"] is False
    assert grid[0.05]["entrance_would_refuse"] is True


def test_hysteresis_separates_a_one_off_from_a_run():
    """One bad reading followed by two good ones: demotes at N=1, not at N=2.

    This is the whole reason §5.2 hysteresis exists, and it is the single most
    consequential thing the operator is choosing.
    """
    history = [
        _payload("2026-08-21", line_ic=-0.20),  # newest
        _payload("2026-08-14", line_ic=+0.08),
        _payload("2026-08-07", line_ic=+0.11),
    ]
    grid = {r["floor"]: r for r in evaluate_floor_sweep(history)["scopes"]["line"]["grid"]}
    row = grid[-0.05]
    assert row["exit_would_demote"]["1"] is True
    assert row["exit_would_demote"]["2"] is False
    assert row["exit_would_demote"]["3"] is False


def test_a_sustained_run_demotes_at_every_hysteresis_count():
    history = [
        _payload("2026-08-21", line_ic=-0.20),
        _payload("2026-08-14", line_ic=-0.18),
        _payload("2026-08-07", line_ic=-0.31),
    ]
    grid = {r["floor"]: r for r in evaluate_floor_sweep(history)["scopes"]["line"]["grid"]}
    row = grid[-0.05]
    assert row["exit_would_demote"]["1"] is True
    assert row["exit_would_demote"]["2"] is True
    assert row["exit_would_demote"]["3"] is True


def test_short_history_reports_unmeasurable_not_false():
    """One reading cannot answer 'would N=3 have fired'. Saying False would
    read as 'the gate checked and declined' — champion-challenger-policy §7.2."""
    grid = {
        r["floor"]: r
        for r in evaluate_floor_sweep(
            [_payload("2026-08-21", line_ic=-0.30)]
        )["scopes"]["line"]["grid"]
    }
    row = grid[-0.05]
    assert row["exit_would_demote"]["1"] is True
    assert row["exit_would_demote"]["2"] == "unmeasurable"
    assert row["exit_would_demote"]["3"] == "unmeasurable"


def test_empty_history_does_not_raise():
    out = evaluate_floor_sweep([])
    assert out["scopes"]["line"]["status"] == "unmeasurable"
    assert out["acts"] is False


def test_malformed_payloads_do_not_raise():
    out = evaluate_floor_sweep([None, {}, {"serving_line_attributed": None}])
    assert out["scopes"]["line"]["status"] == "unmeasurable"


def test_a_null_realized_ic_is_not_treated_as_zero():
    """An attributed block present but carrying a null IC is unmeasurable, not
    a reading of 0.0 — which would clear every negative floor."""
    payload = {
        "trading_day": "2026-08-21",
        "serving_line_attributed": {
            "line": "v3.0-meta",
            "realized_rank_ic": None,
            "attribution_status": "no_matured_outcomes",
        },
    }
    out = evaluate_floor_sweep([payload])
    assert out["scopes"]["line"]["status"] == "unmeasurable"


# ── The load-bearing property ────────────────────────────────────────────────


def test_floor_sweep_cannot_act():
    """No path from the sweep to a promotion, a demotion, or a weights write.

    An observe-mode recorder that can act is not observe mode. Asserted on the
    source because the property is the ABSENCE of a call, which a behavioural
    test can only sample.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(evaluate_floor_sweep)))

    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)

    # Every callable the sweep is allowed to reach. An addition to this set is a
    # deliberate act, reviewed as one — which is the point.
    allowed = {
        "max", "len", "float", "bool", "list", "str", "isinstance", "get",
        "resolver", "demote_gate_enabled",
        "resolve_realized_edge_floor", "resolve_demote_consecutive",
        "resolve_no_good_arm_state", "resolve_attribution_scope",
        "append",
    }
    assert called <= allowed, (
        f"evaluate_floor_sweep calls {sorted(called - allowed)} — an "
        "observe-mode counterfactual recorder must have no path to an action "
        "(alpha-engine-config-I8195). If the new call is genuinely inert, add "
        "it to `allowed` in this test so the addition is reviewed."
    )
    # No S3 / boto handle reaches it at all: it takes exactly one argument.
    sig = inspect.signature(evaluate_floor_sweep)
    assert list(sig.parameters) == ["monitor_history"], (
        "the sweep must not accept an s3 client or a bucket — it has nothing "
        "to write and nothing to read beyond the history it is handed"
    )


def test_floor_sweep_never_resolves_a_reserved_value_for_use():
    """It may ASK whether a reserved value is set (to report what is
    outstanding), but it must not read one INTO the counterfactual — otherwise
    the sweep silently narrows to the ruled value and stops being a sweep."""
    src = inspect.getsource(evaluate_floor_sweep)
    # The only permitted use is inside the outstanding-values probe, which
    # discards the return value.
    assert "resolver()" in src
    for banned in (
        "floor = resolve_realized_edge_floor()",
        "need = resolve_demote_consecutive()",
        "scope = resolve_attribution_scope()",
    ):
        assert banned not in src


def test_sweep_is_independent_of_the_master_switch(monkeypatch):
    """The sweep must produce its curve on exactly the runs where the gate is
    NOT armed — that is when the ruling is still outstanding."""
    import config as cfg

    monkeypatch.setattr(cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE", False, raising=False)
    out = evaluate_floor_sweep([_payload("2026-08-21", line_ic=-0.26)])
    assert out["gate_armed"] is False
    assert out["scopes"]["line"]["status"] == "measured"
    assert len(out["scopes"]["line"]["grid"]) == len(FLOOR_SWEEP)
