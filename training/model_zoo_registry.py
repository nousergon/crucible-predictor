"""training/model_zoo_registry.py — the M-slot arm register (I9313).

One resolution point for "which arms exist, which may serve, and why not".

Before this module the predictor's model slot had **four** places that each
decided membership independently, and they disagreed:

  1. ``model_zoo.train_all_active``      — trains ``status == "active"``
  2. ``model_zoo.select_rotation_specs`` — schedules ``status == "active"``
  3. ``model_zoo.select_winner``         — refuses ``fwd != champ_fwd`` with a
                                           bare ``non_canonical_horizon`` string
  4. ``model_zoo._selection_pbo``        — separately drops the same arms as
                                           ``dropped_misaligned_specs``

The live consequence, measured on the 2026-08-28 leaderboard: ``horizon-60d``
and ``horizon-90d`` were **trained and scored every rotation at a full
training run each**, and were then refused at (3) and (4) for a reason no
artifact explained. Two of four challengers could not win a promotion
regardless of score — a structural exclusion that read, on the leaderboard,
as an ordinary per-candidate verdict.

`champion-challenger-policy.md` already names both halves of the fix:

  §4 — "Same benchmark and horizon across every arm in a slot (fleet default:
       21-day, market-relative vs SPY, log-domain)."
  §4 — an arm that cannot legitimately compete "is `inapplicable`, not
       vacuous, and is refused BEFORE it is ever scored."
  §2 — "before adding an arm, name its slot. If it does not fit an existing
       slot, it is a new slot and needs its own scoring path — not a
       borrowed one."
  §10 — "Every slot names, in its registry, the metric, horizon, benchmark,
       count-matching width, and hysteresis margins it uses."

So: applicability is resolved ONCE, here, from a declared slot descriptor;
an inapplicable arm is refused before training rather than after scoring; and
the reason is a recorded property carried onto the leaderboard, not a string
invented at the point of refusal.

## The multi-horizon decision (recorded, per the task's requirement)

``horizon-60d`` and ``horizon-90d`` are **declared out of the M slot**. They
are not horizon-normalized into it. The reasoning, stated so it is not
relitigated:

* A horizon-normalized statistic (IC/sqrt(h), or an annualized IC-IR) would
  make the *numbers* comparable. It would not make the *arms* comparable,
  because the slot's champion is what ``model.registry.promote_to_champion``
  publishes to the live meta weights prefix, and the executor's exit and risk
  rules implement a 21-day hold. Promoting a 60d arm
  on a normalized score would change what the executor consumes to a horizon
  it does not implement. That is replacing the slot's contract, not winning
  the slot — §2's "it is a new slot and needs its own scoring path".
* The horizon question is still worth asking, and is already answered more
  cheaply: ``analysis/horizon_battery.py`` computes 5d/10d/21d/60d/90d IC with
  non-overlapping windows and bootstrap CIs off the OOS rows the champion run
  already persists, at ~zero marginal cost. Two full weekly training runs buy
  a worse version of a measurement we take offline for free.
* Keeping them active-but-ineligible is the status quo being removed: real
  weekly compute spent on arms that cannot win, occupying 2 of the 4
  ``model_zoo_weekly_budget`` slots that 21-day challengers could hold.

**Retirement itself is an operator decision** (§6) and is NOT taken here.
What this module does is narrower and is a machine fact: an arm whose horizon
is not the slot's canonical horizon is `inapplicable` to the M slot, and an
inapplicable arm is not scheduled, not trained, and not scored. Flipping
``status: retired`` in the live ``predictor.yaml`` (and deleting the spec, per
§6's "retired code is deleted, not left dormant") is Brian's call; until he
makes it, the arms cost nothing and the leaderboard states why.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

__all__ = [
    "SLOT",
    "SlotDescriptor",
    "Arm",
    "resolve_arms",
    "applicable_spec_ids",
    "arm_applicability",
    "as_leaderboard_block",
]


@dataclass(frozen=True)
class SlotDescriptor:
    """The M slot's declared comparison contract (policy §10).

    Every field here is a fact CI can check against code, which is why the
    policy deliberately does not enumerate them itself.
    """

    slot: str
    decision: str
    champion_feeds: str
    metric: str
    canonical_horizon_days: int
    benchmark: str
    label_domain: str
    promote_margin_cfg: str
    promote_min_ic_cfg: str
    # Why a non-canonical horizon is refused rather than normalized. Recorded
    # on every leaderboard so the exclusion is never again a bare string.
    horizon_policy: str
    horizon_policy_reason: str
    horizon_policy_alternative_measure: str


SLOT = SlotDescriptor(
    slot="M",
    decision="which trained model emits predicted_alpha",
    champion_feeds="model.registry.promote_to_champion -> the live meta weights prefix",
    metric="cpcv_mean_ic",
    # Resolved against cfg.FORWARD_DAYS at call time rather than pinned here,
    # so a deliberate slot-wide horizon change moves ONE knob and every arm
    # follows it. The literal is the documented fleet default and the
    # fallback when cfg is unavailable.
    canonical_horizon_days=21,
    benchmark="market-relative vs SPY (sector ETF where mapped)",
    label_domain="log-return",
    promote_margin_cfg="MODEL_ZOO_PROMOTE_MARGIN",
    promote_min_ic_cfg="MODEL_ZOO_PROMOTE_MIN_IC",
    horizon_policy="refuse_non_canonical",
    horizon_policy_reason=(
        "champion-challenger-policy §4 requires one horizon across every arm "
        "in a slot. A horizon-normalized statistic would make the numbers "
        "comparable but not the arms: the M-slot champion is what "
        "model.registry.promote_to_champion publishes to the live meta "
        "weights prefix, and the executor implements a 21-day hold, "
        "so promoting a 60d or 90d arm would change the contract the slot "
        "serves rather than win the slot (§2 — a different decision is a "
        "different slot with its own scoring path). Arms at a non-canonical "
        "horizon are therefore INAPPLICABLE and are refused before they are "
        "trained, not after they are scored."
    ),
    horizon_policy_alternative_measure=(
        "analysis/horizon_battery.py — 5d/10d/21d/60d/90d IC with "
        "non-overlapping windows and bootstrap CIs, computed offline from the "
        "OOS rows the champion run already persists. The horizon question "
        "keeps being measured; it stops costing two full training runs a week."
    ),
)


@dataclass(frozen=True)
class Arm:
    """One registered arm and its resolved applicability to the slot."""

    spec_id: str
    model_version_label: str
    status: str                 # active | retired  (operator-declared)
    horizon_days: int
    applicability: str          # applicable | inapplicable | retired
    reason: str                 # "" when applicable
    retired_date: "str | None" = None
    priority: int = 0
    # I9319 — the spec's allowlisted override knobs, carried so the arena's
    # RECIPE hash can be formed from the register rather than by re-reading
    # ambient config at the hashing site. They are the arm's features and
    # hyperparameters: changing one makes it a different arm
    # (champion-challenger-policy §3.1). Deliberately NOT in ``as_dict`` — the
    # leaderboard block is an audit surface for membership, not a config dump.
    overrides: "dict" = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        """Whether the rotation should spend a training run on this arm.

        The cost half of the decision: an arm that cannot win the slot is not
        scheduled. Refused BEFORE training (policy §4), not after scoring.
        """
        return self.applicability == "applicable"

    def as_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "model_version_label": self.model_version_label,
            "status": self.status,
            "horizon_days": self.horizon_days,
            "applicability": self.applicability,
            "reason": self.reason or None,
            "retired_date": self.retired_date,
            "trainable": self.trainable,
        }


def _resolve_label(spec: dict) -> str:
    """The registry ``model_version`` a spec's trained versions carry.

    Mirrors ``model_zoo.train_spec`` exactly: a ``MODEL_VERSION_LABEL`` in the
    spec's own overrides wins, else the declared ``model_version_label``, else
    ``spec-<id>``.
    """
    ov = spec.get("overrides") or {}
    if "MODEL_VERSION_LABEL" in ov:
        return ov["MODEL_VERSION_LABEL"]
    return spec.get("model_version_label", f"spec-{spec.get('id')}")


def _spec_horizon(spec: dict, canonical: int) -> int:
    """A spec's forward horizon: its ``FORWARD_DAYS`` override, else the slot's.

    A spec with no override trains at the slot horizon by construction, which
    is why the absence of the key means canonical rather than unknown.
    """
    ov = spec.get("overrides") or {}
    try:
        return int(ov.get("FORWARD_DAYS", canonical))
    except (TypeError, ValueError):
        return canonical


def resolve_arms(
    specs: "list | None" = None,
    *,
    canonical_horizon: "int | None" = None,
) -> list[Arm]:
    """Resolve every declared spec into an ``Arm`` with its applicability.

    THE single membership decision for the M slot. ``model_zoo``'s scheduler,
    trainer, winner-selection and PBO all read this rather than each
    re-deriving "is this arm in the slot?" from ``status`` or from a horizon
    comparison of their own.
    """
    if specs is None:
        import config as cfg
        specs = getattr(cfg, "MODEL_SPECS", []) or []
    if canonical_horizon is None:
        try:
            import config as cfg
            canonical_horizon = int(
                getattr(cfg, "FORWARD_DAYS", SLOT.canonical_horizon_days)
            )
        except Exception:  # noqa: BLE001 — config import is not this module's job
            # Recorded, not swallowed: falling back to the documented fleet
            # default is correct, and saying so is the difference between a
            # default and an accident.
            log.warning(
                "model_zoo_registry: could not read cfg.FORWARD_DAYS; using "
                "the declared slot default %d as the canonical horizon.",
                SLOT.canonical_horizon_days, exc_info=True,
            )
            canonical_horizon = SLOT.canonical_horizon_days

    arms: list[Arm] = []
    for spec in specs:
        sid = spec.get("id")
        if not sid:
            continue
        status = spec.get("status", "active")
        horizon = _spec_horizon(spec, canonical_horizon)

        if status != "active":
            applicability, reason = "retired", (
                f"{sid} is declared status={status!r} in the spec register — "
                "an operator decision (champion-challenger-policy §6). It is "
                "not scheduled, trained or scored."
            )
        elif horizon != canonical_horizon:
            applicability, reason = "inapplicable", (
                f"{sid} trains at a {horizon}-day forward horizon; the M slot's "
                f"canonical horizon is {canonical_horizon} days. "
                f"{SLOT.horizon_policy_reason} The horizon question is instead "
                f"measured by {SLOT.horizon_policy_alternative_measure}"
            )
        else:
            applicability, reason = "applicable", ""

        arms.append(Arm(
            spec_id=sid,
            model_version_label=_resolve_label(spec),
            status=status,
            horizon_days=horizon,
            applicability=applicability,
            reason=reason,
            retired_date=spec.get("retired_date"),
            priority=int(spec.get("priority", 0) or 0),
            overrides=dict(spec.get("overrides") or {}),
        ))
    return arms


def applicable_spec_ids(
    specs: "list | None" = None, *, canonical_horizon: "int | None" = None,
) -> list[str]:
    """Spec ids the rotation may schedule and train. Sorted for determinism."""
    return sorted(
        a.spec_id for a in resolve_arms(specs, canonical_horizon=canonical_horizon)
        if a.trainable
    )


def arm_applicability(
    spec_id: "str | None",
    specs: "list | None" = None,
    *,
    canonical_horizon: "int | None" = None,
) -> dict:
    """The recorded applicability property for one spec id.

    Returns ``{"applicability", "reason", "horizon_days"}``. An id that is not
    in the register at all is reported ``unregistered`` — never ``applicable``
    by default, because an unregistered arm writing shadow artifacts is the
    ``thinktank_coverage`` defect the policy names in §3.
    """
    for a in resolve_arms(specs, canonical_horizon=canonical_horizon):
        if a.spec_id == spec_id:
            return {
                "applicability": a.applicability,
                "reason": a.reason or None,
                "horizon_days": a.horizon_days,
            }
    return {
        "applicability": "unregistered",
        "reason": (
            f"{spec_id!r} is not in MODEL_SPECS. An arm that produces output "
            "without a register row is not scored and its data rots unnoticed "
            "(champion-challenger-policy §3)."
        ),
        "horizon_days": None,
    }


def as_leaderboard_block(
    specs: "list | None" = None, *, canonical_horizon: "int | None" = None,
) -> dict:
    """The slot + arms block every leaderboard carries.

    This is the audit surface: it states the slot's comparison contract, the
    horizon ruling and its reason, and every registered arm with the
    applicability that decided whether it ran at all. A reader of the artifact
    alone can tell an arm that LOST from an arm that was never in the race.
    """
    arms = resolve_arms(specs, canonical_horizon=canonical_horizon)
    resolved_horizon = canonical_horizon
    if resolved_horizon is None:
        try:
            import config as cfg
            resolved_horizon = int(
                getattr(cfg, "FORWARD_DAYS", SLOT.canonical_horizon_days)
            )
        except Exception:  # noqa: BLE001
            log.warning(
                "model_zoo_registry: cfg.FORWARD_DAYS unreadable for the "
                "leaderboard block; recording the declared slot default.",
                exc_info=True,
            )
            resolved_horizon = SLOT.canonical_horizon_days
    return {
        "slot": SLOT.slot,
        "decision": SLOT.decision,
        "champion_feeds": SLOT.champion_feeds,
        "metric": SLOT.metric,
        "canonical_horizon_days": resolved_horizon,
        "benchmark": SLOT.benchmark,
        "label_domain": SLOT.label_domain,
        "horizon_policy": SLOT.horizon_policy,
        "horizon_policy_reason": SLOT.horizon_policy_reason,
        "horizon_policy_alternative_measure": SLOT.horizon_policy_alternative_measure,
        "arms": [a.as_dict() for a in arms],
        "n_applicable": sum(1 for a in arms if a.trainable),
        "n_inapplicable": sum(1 for a in arms if a.applicability == "inapplicable"),
        "n_retired": sum(1 for a in arms if a.applicability == "retired"),
    }
