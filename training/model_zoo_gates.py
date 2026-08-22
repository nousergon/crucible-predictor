"""training/model_zoo_gates.py — the model-zoo ENTRANCE and EXIT gates, and the
ONE state they share.

alpha-engine-config-I8175 (the exit: demote a champion shown to have no realized
edge) and alpha-engine-config-I8195 (the entrance: stop admitting one) answer the
same question — **what does the system do when NO arm is good enough?** They were
split as tracker items, never as designs: if the demote-TO state and the
no-arm-clears state are chosen independently, the system can demote into a state
its own promotion gate would have refused.

This module is how they are held consistent, by construction rather than by
review:

* ``resolve_no_good_arm_state()`` is the SINGLE configuration point. Both gates
  call it. There is no second symbol either one can drift onto.
* ``resolve_realized_edge_floor()`` is the SINGLE numeric bar, expressed in the
  SAME quantity (realized 21d rank-IC) for both directions. The exit demotes when
  the slot's attributed realized edge sits at or below the floor; the entrance
  refuses to admit a new arm into live capital while the slot sits at or below
  that same floor. One number, both directions.

**Nothing here defaults.** Every value is un-defaulted and raises a named error
when a gate that needs it runs unset. That is deliberate and it is the whole
point: choosing between flat / reduced size / hold-incumbent / last-known-good is
a live-trading-risk decision reserved to the operator (`principles.md` §3.2), and
a gate that silently guesses is worse than no gate — it moves capital on a value
nobody ruled.

Why the bar is on REALIZED edge and not on ``cpcv_mean_ic``
-----------------------------------------------------------
Measured 2026-08-22 over the whole `predictor/model_zoo/` history, with each
champion era attributed to the version that actually served it: promotion-time
``cpcv_mean_ic`` has **no measurable relationship** with the later realized 21d
rank-IC (n=6 attributed eras, Spearman +0.03, Pearson +0.06, exact permutation
p=0.90). A ``promote_min_ic`` raised to any positive number would therefore be a
threshold with no evidence behind it. The quantity that does describe whether the
slot is earning its capital is the realized rank-IC — so that is what both gates
read. See alpha-engine-config-I8195.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402

log = logging.getLogger(__name__)

# The exhaustive set of states the system may occupy when no arm is good enough.
# Adding one is a policy change, not a config change — it needs a PR here.
NO_GOOD_ARM_STATES: tuple[str, ...] = (
    # Keep serving the incumbent and keep measuring. Lowest operational blast
    # radius; accepts that a champion with no demonstrated edge keeps trading.
    "hold_incumbent",
    # Stop emitting tradable alpha — the predictor publishes predictions marked
    # non-actionable and the executor holds no new risk on them.
    "flat",
    # Keep serving, at reduced gross. Requires an executor-side contract: the
    # predictor does not size positions.
    "reduced_size",
    # Revert the live weights to the last champion version whose own attributed
    # realized edge cleared the floor.
    "last_known_good",
)


# Which of the four states this repo can actually EXECUTE today. `hold_incumbent`
# and `last_known_good` are predictor-local: they move (or deliberately do not
# move) the live weights pointer, which is this repo's own artifact.
#
# `flat` and `reduced_size` are NOT predictor-local. The predictor emits
# `predicted_alpha`; it does not size positions and it cannot stand a book down.
# Both require a contract the executor consumes, and that contract does not exist
# — so arming the gate on either would produce a gate that fires and then cannot
# act. It refuses to arm instead, loudly, BEFORE any capital moves. Building the
# executor-side contract is the tracked follow-up if the operator rules one of
# them (alpha-engine-config-I8175).
ACTUATOR_IMPLEMENTED: frozenset = frozenset({"hold_incumbent", "last_known_good"})


class NoGoodArmStateUnset(RuntimeError):
    """A model-zoo gate that must decide what to do when no arm is good enough
    ran with ``cfg.MODEL_ZOO_NO_GOOD_ARM_STATE`` unset or unrecognised.

    Raised, never defaulted (alpha-engine-config-I8175). The fallback state is a
    reserved operator decision; guessing it would move live capital on a value
    nobody ruled."""


class NoGoodArmActuatorMissing(RuntimeError):
    """The configured no-good-arm state is recognised but this repo has no
    actuator for it — ``flat`` and ``reduced_size`` need an executor-side contract
    that does not exist yet. Raised at ARM time, never at fire time: a gate that
    fires and then cannot act is worse than a gate that refuses to arm."""


class RealizedEdgeFloorUnset(RuntimeError):
    """A model-zoo gate ran with ``cfg.MODEL_ZOO_REALIZED_EDGE_FLOOR`` (or its
    hysteresis count) unset. Same contract as ``NoGoodArmStateUnset``: the bar is
    a reserved operator decision and is never invented at runtime."""


def resolve_no_good_arm_state() -> str:
    """THE no-good-arm state, shared by the entrance and the exit gate.

    Returns one of ``NO_GOOD_ARM_STATES``. Raises ``NoGoodArmStateUnset`` when the
    config point is absent or not a recognised state — deliberately loud, so a
    gate can never act on an un-ruled fallback.
    """
    raw = getattr(cfg, "MODEL_ZOO_NO_GOOD_ARM_STATE", None)
    if raw is None or str(raw).strip() == "":
        raise NoGoodArmStateUnset(
            "alpha-engine-config-I8175: MODEL_ZOO_NO_GOOD_ARM_STATE is UNSET. A "
            "model-zoo gate needs to know what the system does when no arm is good "
            "enough, and that is a reserved operator decision — flat, reduced size, "
            "hold the incumbent, and last-known-good are materially different risk "
            f"postures. Set it to one of {list(NO_GOOD_ARM_STATES)} in "
            "config/predictor.yaml (model_zoo_no_good_arm_state) before enabling "
            "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE."
        )
    state = str(raw).strip()
    if state not in NO_GOOD_ARM_STATES:
        raise NoGoodArmStateUnset(
            f"alpha-engine-config-I8175: MODEL_ZOO_NO_GOOD_ARM_STATE={state!r} is not "
            f"a recognised state. Legal values: {list(NO_GOOD_ARM_STATES)}."
        )
    return state


def resolve_realized_edge_floor() -> float:
    """THE realized 21d rank-IC floor, shared by the entrance and the exit gate.

    Raises ``RealizedEdgeFloorUnset`` when absent or non-numeric.
    """
    raw = getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_FLOOR", None)
    if raw is None or str(raw).strip() == "":
        raise RealizedEdgeFloorUnset(
            "alpha-engine-config-I8195: MODEL_ZOO_REALIZED_EDGE_FLOOR is UNSET. Both "
            "the entrance and the exit gate read this one bar; it is a reserved "
            "operator decision and is never invented at runtime. Measured 2026-08-22: "
            "promotion-time cpcv_mean_ic does not predict realized 21d rank-IC "
            "(n=6 attributed champion eras, Spearman +0.03, permutation p=0.90), so "
            "the bar belongs on realized edge and its value needs a ruling."
        )
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise RealizedEdgeFloorUnset(
            f"alpha-engine-config-I8195: MODEL_ZOO_REALIZED_EDGE_FLOOR={raw!r} is not "
            "numeric."
        ) from exc


def resolve_demote_consecutive() -> int:
    """Hysteresis count for the EXIT gate (`champion-challenger-policy` §5.2):
    consecutive attributed readings at/below the floor before a demote fires.

    Raises ``RealizedEdgeFloorUnset`` when unset. Demoting a sitting champion on a
    single reading oscillates on noise and destroys the measurement continuity §3
    depends on.
    """
    raw = getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE", None)
    if raw is None or str(raw).strip() == "":
        raise RealizedEdgeFloorUnset(
            "alpha-engine-config-I8175: MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE is "
            "UNSET. champion-challenger-policy §5.2 requires hysteresis before "
            "demoting a sitting champion; the count is a reserved operator decision."
        )
    try:
        n = int(raw)
    except (TypeError, ValueError) as exc:
        raise RealizedEdgeFloorUnset(
            f"alpha-engine-config-I8175: MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE="
            f"{raw!r} is not an integer."
        ) from exc
    if n < 1:
        raise RealizedEdgeFloorUnset(
            "alpha-engine-config-I8175: MODEL_ZOO_REALIZED_EDGE_DEMOTE_CONSECUTIVE "
            f"must be >= 1, got {n}."
        )
    return n


ATTRIBUTION_SCOPES: tuple[str, ...] = ("version", "line")

#: The monitor payload field each scope reads.
_SCOPE_FIELD = {
    "version": "serving_champion_attributed",
    "line": "serving_line_attributed",
}


def resolve_attribution_scope() -> str:
    """WHICH attributed realized read the EXIT gate acts on.

    ``version`` — the serving champion's own matured predictions. Correct in the
    narrow sense and, measured 2026-08-22, essentially never available: the
    champion rotates weekly while a 21-trading-day window takes ~30 calendar days
    to close, so a version's first matured outcome lands four or five rotations
    after it stopped serving. A version-scoped gate reports ``unmeasurable``
    forever — honest, but it never fires.

    ``line`` — every version sharing the serving champion's ``model_version``
    prefix. The unit that has actually been trading capital continuously
    (``v3.0-meta`` since 2026-07-24), and the coarser unit a gate can realistically
    read. It grades an architecture rather than one retrain.

    Un-defaulted and reserved: this is a choice about what the system is allowed to
    demote FOR, and neither answer is free.
    """
    raw = getattr(cfg, "MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE", None)
    if raw is None or str(raw).strip() == "":
        raise RealizedEdgeFloorUnset(
            "alpha-engine-config-I8175: MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE is UNSET. "
            "The exit gate must be told whether it grades the serving VERSION or the "
            "serving LINE. Measured 2026-08-22: weekly rotation against a 21d horizon "
            "means the serving version has matured outcomes essentially never, so a "
            "version-scoped gate would never fire — a gate that reads as coverage "
            "while doing nothing (champion-challenger-policy §7.4). Legal values: "
            f"{list(ATTRIBUTION_SCOPES)}."
        )
    scope = str(raw).strip()
    if scope not in ATTRIBUTION_SCOPES:
        raise RealizedEdgeFloorUnset(
            f"alpha-engine-config-I8175: MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE={scope!r} "
            f"is not recognised. Legal values: {list(ATTRIBUTION_SCOPES)}."
        )
    return scope


def demote_gate_enabled() -> bool:
    """Whether the realized-edge AUTO-DEMOTE gate (L4539) is armed."""
    return bool(getattr(cfg, "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE", False))


def evaluate_realized_edge_exit(monitor_history: list[dict]) -> dict:
    """The EXIT gate (alpha-engine-config-I8175 / L4539).

    ``monitor_history`` is newest-first: each entry is a champion realized-edge
    monitor payload as written by ``analysis/observe_leaderboard.py``. The gate
    reads the ATTRIBUTED read only — the realized rank-IC of the serving champion's
    OWN predictions — never the unattributed slot aggregate.

    That distinction is the whole correctness of this gate. The legacy aggregate
    pools 42 days of live predictions with no version filter while the champion
    rotates weekly, and only pairs whose 21d forward window has closed are scored —
    so the matured subset is structurally the OLDEST ~3 weeks of the window and
    contains **zero** predictions from the version currently serving. Measured
    2026-08-22: the -0.2626 that raised alpha-engine-config-I8175 belongs to
    ``spec-residual-mom-2026-07-17-f478ece3`` (era mean -0.31), not to the serving
    ``v3.0-meta-2026-08-14-119e069b``, which has never had a matured outcome.
    Acting on the aggregate would demote a champion for its predecessors'
    performance — `champion-challenger-policy` §7.5, a record asserting something
    false about its own origin.

    Returns a verdict dict, ALWAYS. Never raises for a data reason; raises only
    when the gate is armed and its reserved configuration is unset (see the
    module docstring — that is a defect in deployment, not in the data).

    Verdict ``status``:
      ``disabled``      — the gate is not armed; no action.
      ``unmeasurable``  — no attributed reading yet. NON-BLOCKING and it does NOT
                          demote (`champion-challenger-policy` §5.1: you cannot
                          gate on a statistic you did not measure), but it is
                          reported loud, never rendered as a pass.
      ``pass``          — attributed realized edge is above the floor.
      ``arming``        — at/below the floor, but the hysteresis run is short.
      ``demote``        — hysteresis satisfied; ``demote_to`` names the state.
    """
    if not demote_gate_enabled():
        return {
            "status": "disabled",
            "reason": (
                "MODEL_ZOO_REALIZED_EDGE_DEMOTE_ENABLE is off — the realized-edge "
                "auto-demote gate (L4539) is not armed. The champion keeps serving "
                "and the noise-watch alert remains the operator surface."
            ),
        }

    # Armed ⇒ every reserved value must be present. Raise rather than guess.
    floor = resolve_realized_edge_floor()
    need = resolve_demote_consecutive()
    state = resolve_no_good_arm_state()
    scope = resolve_attribution_scope()
    if state not in ACTUATOR_IMPLEMENTED:
        raise NoGoodArmActuatorMissing(
            f"alpha-engine-config-I8175: the realized-edge auto-demote gate is armed "
            f"with MODEL_ZOO_NO_GOOD_ARM_STATE={state!r}, but this repo has no "
            f"actuator for it. The predictor emits predicted_alpha; it does not size "
            f"positions, so {state!r} needs a contract the executor consumes and that "
            f"contract does not exist. Implemented states: "
            f"{sorted(ACTUATOR_IMPLEMENTED)}. Refusing to arm rather than firing a "
            f"gate that cannot act."
        )

    field = _SCOPE_FIELD[scope]
    readings: list[dict] = []
    for payload in monitor_history[:need]:
        attributed = (payload or {}).get(field) or {}
        readings.append({
            "trading_day": (payload or {}).get("trading_day"),
            "scope": scope,
            "version_id": attributed.get("version_id") or attributed.get("line"),
            "realized_rank_ic": attributed.get("realized_rank_ic"),
            "n_matured_outcomes": attributed.get("n_matured_outcomes"),
            "attribution_status": attributed.get("attribution_status"),
        })

    usable = [r for r in readings if isinstance(r.get("realized_rank_ic"), (int, float))]
    if not usable:
        return {
            "status": "unmeasurable",
            "floor": floor,
            "consecutive_required": need,
            "attribution_scope": scope,
            "no_good_arm_state": state,
            "readings": readings,
            "reason": (
                "NO attributed realized-edge reading for the serving champion — the "
                "gate cannot fire on a statistic it did not measure "
                "(champion-challenger-policy §5.1). This is reported, never rendered "
                "as a pass. Most likely cause: predictions in the matured window "
                "carry no champion_version_id, or the serving champion has not yet "
                "had a 21d forward window close."
            ),
        }

    if len(usable) < need:
        return {
            "status": "unmeasurable",
            "floor": floor,
            "consecutive_required": need,
            "n_attributed_readings": len(usable),
            "attribution_scope": scope,
            "no_good_arm_state": state,
            "readings": readings,
            "reason": (
                f"only {len(usable)} attributed reading(s) available; hysteresis "
                f"needs {need}. Not a pass — an unmeasurable gate."
            ),
        }

    below = [r for r in usable if float(r["realized_rank_ic"]) <= floor]
    if len(below) < need:
        latest = float(usable[0]["realized_rank_ic"])
        if latest > floor:
            return {
                "status": "pass",
                "floor": floor,
                "consecutive_required": need,
                "realized_rank_ic": latest,
                "version_id": usable[0].get("version_id"),
                "attribution_scope": scope,
                "no_good_arm_state": state,
                "readings": readings,
                "reason": (
                    f"serving champion's attributed realized 21d rank-IC {latest:+.4f} "
                    f"> floor {floor:+.4f}."
                ),
            }
        return {
            "status": "arming",
            "floor": floor,
            "consecutive_required": need,
            "consecutive_below": len(below),
            "realized_rank_ic": latest,
            "version_id": usable[0].get("version_id"),
            "attribution_scope": scope,
            "no_good_arm_state": state,
            "readings": readings,
            "reason": (
                f"attributed realized rank-IC {latest:+.4f} <= floor {floor:+.4f}, but "
                f"only {len(below)} of the required {need} consecutive readings are "
                "below it — hysteresis not satisfied (champion-challenger-policy §5.2)."
            ),
        }

    return {
        "status": "demote",
        "floor": floor,
        "consecutive_required": need,
        "consecutive_below": len(below),
        "realized_rank_ic": float(usable[0]["realized_rank_ic"]),
        "version_id": usable[0].get("version_id"),
        "attribution_scope": scope,
        "demote_to": state,
        "no_good_arm_state": state,
        "readings": readings,
        "reason": (
            f"EXIT GATE FIRED: the serving champion's own attributed realized 21d "
            f"rank-IC has been <= the floor {floor:+.4f} on {len(below)} consecutive "
            f"readings (hysteresis {need}). Demote-to state: {state}."
        ),
    }


def evaluate_absolute_bar(
    *,
    promote_min_ic: float,
    candidate_cpcv_ic: float | None,
    slot_realized_rank_ic: float | None,
    slot_attribution_status: str | None = None,
) -> dict:
    """The ENTRANCE gate record (alpha-engine-config-I8195).

    Two components, both recorded on every promotion artifact so a future reader
    can tell a GATED promotion from an UNGATED one — which today is impossible,
    because ``promote_min_ic: 0.0`` renders identically whether it was ruled or
    merely never set:

    1. ``candidate_bar`` — the existing ``promote_min_ic`` floor on the
       candidate's own leak-free CPCV mean IC. Recorded with ``is_gating``, which
       is False at 0.0: a floor of zero is a sign check, not a bar, and calling it
       a bar is how an ungated promotion reads as gated.
    2. ``slot_bar`` — the SHARED realized-edge floor. This is what makes the
       entrance and the exit the same decision: a new arm is not admitted to live
       capital while the slot's own attributed realized edge sits at or below the
       floor the exit gate would demote for. Without it the two disagree and the
       system can demote into a state promotion would have refused.

    Never raises: when the shared floor is unruled the slot component is recorded
    as ``unset`` and does not gate. Recording an unruled bar as a pass is the
    defect this field exists to prevent.
    """
    candidate_clears = (
        None if candidate_cpcv_ic is None else bool(candidate_cpcv_ic > promote_min_ic)
    )
    record: dict = {
        "candidate_bar": {
            "field": "cpcv_mean_ic",
            "promote_min_ic": promote_min_ic,
            "candidate_value": candidate_cpcv_ic,
            "clears": candidate_clears,
            # A zero floor is a positivity check. Naming it honestly is the whole
            # point of this record (alpha-engine-config-I8195).
            "is_gating": bool(promote_min_ic > 0.0),
            "note": (
                "Measured 2026-08-22: promotion-time cpcv_mean_ic has no measurable "
                "relationship with later realized 21d rank-IC (n=6 attributed "
                "champion eras, Spearman +0.03, permutation p=0.90). A positive value "
                "here is not supported by the model_zoo history — see -I8195."
            ),
        }
    }

    try:
        floor = resolve_realized_edge_floor()
    except RealizedEdgeFloorUnset as exc:
        record["slot_bar"] = {
            "field": "serving_champion_attributed.realized_rank_ic",
            "status": "unset",
            "is_gating": False,
            "reason": str(exc),
        }
        record["gated"] = bool(promote_min_ic > 0.0)
        return record

    if not isinstance(slot_realized_rank_ic, (int, float)):
        record["slot_bar"] = {
            "field": "serving_champion_attributed.realized_rank_ic",
            "floor": floor,
            "status": "unmeasurable",
            "attribution_status": slot_attribution_status,
            "is_gating": False,
            "reason": (
                "no attributed realized-edge reading for the serving champion — "
                "cannot gate on a statistic that was not measured "
                "(champion-challenger-policy §5.1). Reported, never a pass."
            ),
        }
        record["gated"] = bool(promote_min_ic > 0.0)
        return record

    clears = bool(float(slot_realized_rank_ic) > floor)
    # The state is only meaningful when the bar is NOT cleared. Resolve it
    # defensively: this recorder must never raise (it runs on the artifact-write
    # path), while the ACTING gates above still raise on an unruled state.
    state_when_below = None
    if not clears:
        try:
            state_when_below = resolve_no_good_arm_state()
        except NoGoodArmStateUnset:
            state_when_below = "unset"
    record["slot_bar"] = {
        "field": "serving_champion_attributed.realized_rank_ic",
        "floor": floor,
        "slot_value": float(slot_realized_rank_ic),
        "status": "pass" if clears else "below_floor",
        "clears": clears,
        "is_gating": True,
        "no_good_arm_state": state_when_below,
        "reason": (
            f"slot attributed realized 21d rank-IC {float(slot_realized_rank_ic):+.4f} "
            f"{'>' if clears else '<='} floor {floor:+.4f}"
        ),
    }
    record["gated"] = True
    return record
