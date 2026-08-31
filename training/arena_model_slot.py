"""training/arena_model_slot.py — the model (M) slot, wired to the shared arena.

alpha-engine-config-I9319 (the wiring) and -I9322 (the statistic it ranks on).

``nousergon_lib.arena`` is the fleet's single implementation of
``champion-challenger-policy.md`` §§3–6 — the arm register, the score ladder,
longest-common-window pairing, the anytime-valid confidence sequence, the
Condorcet pairwise ranking, the pointer decision and the cap-with-grace
retirement rule. **This module implements none of that and must never start
to** (policy §10, ``shared-code-policy.md``). It is an ADAPTER: it answers the
four questions the engine deliberately refuses to answer for a slot, and hands
them over.

    1. Which arms exist?          → :func:`build_register`
    2. What is each arm's score?  → :func:`build_series`
    3. Did each arm train soundly? → :func:`training_statuses`
    4. May each arm SERVE?         → :func:`serving_preconditions`

Everything else — including who wins — belongs to ``run_cycle``.

The recipe boundary (the `complexity:high` fork, decided here)
==============================================================

Policy §3.1: *"an arm fixes its features, hyperparameters, training-window rule
and refit cadence at registration, and its fitted weights refresh on the
schedule the recipe itself declares."* The predictor's raw material does not
come pre-cut that way, so the boundary is a decision. It is:

    **One arm per declared zoo SPEC, plus one arm for the base champion
    architecture. A registry bundle is a REFIT of an arm, never an arm.**

The argument, in the order the facts decide it:

* ``version_id = f"{model_version}-{date}-{fp}"`` where ``fp`` hashes the
  contract files' S3 ETags (``model/registry.py::snapshot_to_registry``). It
  therefore changes on **every fit**, identically-specified or not. A register
  keyed on ``version_id`` would mint a brand-new arm with an empty series every
  Saturday — which is precisely the failure §3.1 abolishes, and precisely why
  ``promoted_kind: champion-arch-refresh`` existed and why 100% of predictor
  transitions used it. ``version_id`` names a fitted BUNDLE.
* What is actually fixed across those refits is the spec: its
  ``model_version_label``, its ``overrides`` (the allowlisted knobs in
  ``model_zoo._ALLOWED_OVERRIDES`` — the features and hyperparameters), and its
  forward horizon. Change any of them and the spec hash changes, which mints a
  new arm id that cannot inherit the old series. That is §3.1 enforced by
  construction rather than requested in prose.
* The base champion architecture (``cfg.MODEL_VERSION_LABEL``) is not a
  declared spec but registers itself into the same pool every Saturday
  (``model_zoo._resolve_base_champion_version``). Under this boundary it is an
  ordinary arm whose recipe is "the champion architecture, no overrides", and
  its weekly retrain is a refit. **This is what makes ``champion-arch-refresh``
  structurally unwritable rather than merely discouraged**: there is no longer
  a category of event for it to be. Refreshing the champion arm's weights is
  the arm doing its job, gated only by §5.3's serving preconditions; a
  *different* recipe winning the pointer is a promotion like any other.

Two things the boundary needs that the repository does not yet persist, and
what was done about each:

``refit cadence``
    Not stored per arm anywhere — cadence is rotation-level config
    (``MODEL_ZOO_WEEKLY_BUDGET`` round-robins the pool). It is therefore
    **resolved from live config and hashed into the recipe explicitly**
    (:func:`recipe_spec`), so a deliberate cadence change mints new arms rather
    than silently re-shaping the meaning of an existing series. That is the
    conservative direction: §3.1 names cadence as part of the recipe, so a
    cadence change must not inherit a record earned under a different one.

``the spec itself``
    ``overrides`` live only in ``config.MODEL_SPECS`` at rotation time and are
    never written onto a manifest, so a register derived only from S3 could not
    reconstruct a recipe. The register is therefore derived from the LIVE spec
    register (``training/model_zoo_registry.resolve_arms``) and **persisted as
    an append-only event log** at ``arena/model/register.json``. Ambient config
    seeds it once per arm; the log is the record from then on. A spec edited in
    place turns up as a new arm id superseding the old one — visibly, in the
    log — instead of quietly re-labelling a track record.

What it ranks on (I9322, RULED)
===============================

**Realized market-relative rank-IC**, per arm, per date. Not promotion-time
CPCV IC, which was measured on 2026-08-22 to have no relationship with the
outcome it is supposed to predict — n=6 attributed champion eras, Spearman
+0.03, permutation p=0.90 (``predictor/model_zoo/leaderboard/2026-08-28.json``,
``absolute_bar.candidate_bar.note``). CPCV IC is retained as a per-arm
DIAGNOSTIC on the emitted cycle and never as the ranking basis.

The producer already exists and is reused rather than re-implemented:
``analysis/variant_cutover_gate.compute_realized_alpha_for_pairs`` fills the
sector-ETF-relative (SPY fallback) forward log-return per (date, ticker), and
``analysis/observe_leaderboard.per_date_rank_ic_by_version`` reduces those pairs
to a per-version, per-date cross-sectional Spearman. This module's only job is
to fold the per-VERSION series onto the per-ARM series the register defines —
which is the whole point of the recipe boundary: a refit continues the line.

The accepted cost, ruled: a newly registered arm has no decision-grade score
until its first 21-trading-day forward window matures. The confidence sequence
reports that honestly as a wide interval and the incumbent holds. That is
correct behaviour, not a stall, and it is why **no minimum-week or
minimum-cohort bar appears anywhere below** (policy §5.0).
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

__all__ = [
    "SLOT", "SLOT_KIND", "BENCHMARK", "ARENA_CONFIG", "DIFF_CLIP_RATIONALE",
    "REGISTER_KEY", "CYCLE_PREFIX", "BASE_ARCH_ARM",
    "recipe_spec", "build_register", "build_series", "training_statuses",
    "serving_preconditions", "run_model_slot_cycle", "emit_cycle",
    "ArenaSlotUnservable",
]

SLOT = "M"
SLOT_KIND = "model"

#: The M slot is a RANKING stage, not a selection stage, so
#: ``SELECTION_SLOT_KINDS`` does not fix its benchmark and the choice is this
#: slot's to declare (policy §4). Every arm scores the SAME universe on the
#: SAME day, so "the population it selected from" carries no information here —
#: it is identical for every arm and would difference away to zero. What the
#: executor consumes is a market-relative alpha ranking, so that is what the
#: arms are graded on: the realized forward log-return of each name minus its
#: sector ETF's (SPY where the ticker has no sector mapping), exactly as
#: ``compute_realized_alpha_for_pairs`` computes it. Grading against the raw
#: return instead would rank arms on how much market beta they happened to
#: carry, which is the 2026-08-17 SPY inversion wearing different clothes.
BENCHMARK = "market-relative-21d-sector-neutral-alpha"

#: Durable per-cycle artifact and the persisted register.
CYCLE_PREFIX = "arena/model"
REGISTER_KEY = f"{CYCLE_PREFIX}/register.json"

#: The base champion-architecture arm. Not a declared spec (it registers itself
#: from the ``--full-only`` training state), but an arm in every other sense.
BASE_ARCH_ARM = "champion-arch"

DIFF_CLIP = 0.30

DIFF_CLIP_RATIONALE = (
    "The per-date score is a cross-sectional Spearman rank-IC over ~900 names, "
    "so it lives in [-1, +1] and a PAIRED per-date difference between two arms "
    "lives in [-2, +2]. The clip is the declared sub-Gaussian scale of the "
    "anytime-valid sequence (policy §5.0: 'the scale must be declared, not "
    "discovered'), so it must bound essentially every observed difference "
    "without being so loose that the interval never narrows. 0.30 is chosen "
    "from the observed range: single-arm attributed realized 21d rank-IC "
    "readings on this slot sit within roughly +/-0.15 (the Gu-Kelly-Xiu "
    "realistic monthly-OOS band is 0.03-0.07, and the fleet's own realized-edge "
    "floor sits at ~0.0), and two arms scoring the SAME universe on the SAME "
    "day share the whole market factor, so their paired difference is "
    "materially tighter than either level. 0.30 is therefore ~2x the largest "
    "single-arm level ever recorded here and comfortably above any paired "
    "difference. It is not a free parameter: every cycle records "
    "`diff_clip_diagnostics.clip_rate` — the fraction of paired per-date "
    "differences at or beyond the clip. A clip rate that stops being ~0 means "
    "the declared scale is wrong and the bound is no longer honest, which is a "
    "finding, not a tuning opportunity."
)


def _arena():
    """Import the engine lazily so a missing extra fails at the call, loudly."""
    from nousergon_lib.arena import engine
    return engine


def _arena_config():
    engine = _arena()
    return engine.ArenaConfig(
        slot=SLOT,
        slot_kind=SLOT_KIND,
        benchmark=BENCHMARK,
        alpha=0.05,
        diff_clip=DIFF_CLIP,
        variance_mode="declared",
        # A well-formedness check ONLY: a paired window of zero dates forms no
        # statistic at all. It is NOT an evidence bar and may never be used as
        # one — the confidence sequence is the evidence bar (policy §5.0).
        min_paired_dates=1,
        cap=5,
        grace_weeks=4,
        min_active_arms=3,
        retired_trailing_cycles=8,
        # The four-week grace period IS the evidence bar for retirement; the
        # sequence is the evidence bar for serving (policy §6.2).
        retire_evidence="point",
        max_ladder_weeks=None,
    )


class _LazyConfig:
    """``ARENA_CONFIG`` without importing the engine at module import time."""

    _cached = None

    def __getattr__(self, name):
        if _LazyConfig._cached is None:
            _LazyConfig._cached = _arena_config()
        return getattr(_LazyConfig._cached, name)

    def resolve(self):
        if _LazyConfig._cached is None:
            _LazyConfig._cached = _arena_config()
        return _LazyConfig._cached


ARENA_CONFIG = _LazyConfig()


class ArenaSlotUnservable(RuntimeError):
    """No arm may serve this cycle. Policy §5.3: the slot fails LOUD.

    Distinct from ``TrainingIntegrityError`` (§3), which is about the training
    substrate. This one says every arm trained fine and none of them is fit to
    be consumed — continuing to serve a known-unfit arm is never the safer
    option, and rendering that as a quiet hold is §7.2's dominant bug class.
    """


# ── 1. the register ─────────────────────────────────────────────────────────

def recipe_spec(arm, *, cadence: str) -> dict:
    """The immutable RECIPE for one arm. Its hash becomes the arm id.

    Every field here is something a change to which makes the arm a DIFFERENT
    arm that must win its place again (policy §3.1). Nothing that varies
    between refits of the same recipe may appear — no dates, no ``version_id``,
    no fitted statistic.
    """
    overrides = dict(getattr(arm, "overrides", None) or {})
    return {
        "slot": SLOT,
        "spec_id": arm.spec_id,
        "model_version_label": arm.model_version_label,
        "forward_days": arm.horizon_days,
        # Sorted so a dict-ordering change can never mint a phantom arm.
        "overrides": {k: overrides[k] for k in sorted(overrides)},
        "training_window_rule": "expanding-window-from-TRAIN_START_DATE",
        "refit_cadence": cadence,
    }


def _cadence(n_applicable: int, budget: int, spec_id: str) -> str:
    """The refit cadence a recipe declares, resolved from live rotation config.

    The champion architecture retrains every Saturday unconditionally (the
    ``--full-only`` state). Every other arm is refit round-robin by
    ``MODEL_ZOO_WEEKLY_BUDGET``, so its cadence is a function of the pool size
    and the budget — both live facts, both hashed in, so that changing either
    is recorded as a new recipe rather than silently re-shaping the meaning of
    an existing track record.
    """
    if spec_id == BASE_ARCH_ARM:
        return "weekly"
    if budget <= 0 or n_applicable <= 0:
        return "unscheduled"
    import math
    return f"round-robin:every-{max(1, math.ceil(n_applicable / budget))}-weeks"


class _BaseArchArm:
    """The base champion architecture, shaped like a ``model_zoo_registry.Arm``."""

    applicability = "applicable"
    status = "active"
    reason = ""
    retired_date = None
    overrides: dict = {}

    def __init__(self, label: str, horizon_days: int):
        self.spec_id = BASE_ARCH_ARM
        self.model_version_label = label
        self.horizon_days = horizon_days

    @property
    def trainable(self) -> bool:
        return True


def slot_arms(specs=None, *, canonical_horizon: int | None = None) -> list:
    """Every arm in the M slot: the applicable declared specs plus the base arch.

    Derived from the LIVE spec register — ``model_zoo_registry.resolve_arms``,
    the one membership decision (I9313) — never from a literal. An arm the
    register calls ``inapplicable`` (a non-canonical horizon) is not in this
    slot at all and is excluded here with the register's own recorded reason;
    an arm the register calls ``retired`` IS registered, because a retired arm
    keeps being scored for its trailing window (policy §6.3).
    """
    from training import model_zoo_registry as reg

    arms = reg.resolve_arms(specs, canonical_horizon=canonical_horizon)
    out = [a for a in arms if a.applicability in ("applicable", "retired")]
    for a in arms:
        if a.applicability == "inapplicable":
            log.info("arena[M]: %s excluded from the slot — %s", a.spec_id, a.reason)

    horizon = canonical_horizon
    if horizon is None:
        horizon = int(getattr(_cfg(), "FORWARD_DAYS", reg.SLOT.canonical_horizon_days))
    label = getattr(_cfg(), "MODEL_VERSION_LABEL", "v3.0-meta")
    out.append(_BaseArchArm(label, horizon))
    return out


def _cfg():
    import config as cfg
    return cfg


def load_register(s3, bucket: str):
    """The persisted append-only event log, or an empty register.

    An unreadable-but-present log RAISES: silently starting a fresh register
    would reset every arm's series and re-create the exact defect §3.1 exists
    to prevent. Only a genuinely absent log is a legitimate empty start.
    """
    from nousergon_lib.arena.arms import ArmRegister

    try:
        body = s3.get_object(Bucket=bucket, Key=REGISTER_KEY)["Body"].read()
    except KeyError:
        log.warning(
            "arena[M]: no register at s3://%s/%s — BOOTSTRAP: registering "
            "every live arm fresh. Every subsequent cycle folds this log.",
            bucket, REGISTER_KEY,
        )
        return ArmRegister()
    except Exception as exc:  # noqa: BLE001 — absence is the bootstrap case
        code = getattr(getattr(exc, "response", None), "get", lambda *_: {})("Error") or {}
        if isinstance(code, dict) and code.get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
            log.warning(
                "arena[M]: no register at s3://%s/%s — BOOTSTRAP: registering "
                "every live arm fresh. Every subsequent cycle folds this log.",
                bucket, REGISTER_KEY,
            )
            return ArmRegister()
        log.warning(
            "arena[M]: register read failed (%s) — treated as absent only if "
            "the key does not exist; otherwise this raises.", exc,
        )
        raise
    return ArmRegister.from_dicts(json.loads(body))


def build_register(
    s3, bucket: str, *, as_of: str, specs=None, canonical_horizon: int | None = None,
    versions: list | None = None, register=None,
):
    """Fold the live slot roster and the registry's refit history into the log.

    Returns ``(register, arm_id_by_label)``. Pure with respect to S3 when
    ``register`` and ``versions`` are both supplied.

    Three kinds of append, and nothing else is ever written:

    * a **registration** for an arm whose recipe hash is not yet in the log. If
      an arm with the same name is already registered under a different hash,
      the new one carries ``supersedes`` to it and starts a fresh series —
      §3.1, enforced by the id rather than requested in prose.
    * a **refit** for every ``predictor/registry/{version_id}/`` bundle dated
      after the arm's last recorded event. Changes no id and resets no series.
    * a **retirement** for an arm the live spec register declares retired.

    Retirements the ENGINE decides (the cap-with-grace rule) are applied by the
    caller from the cycle's verdicts, never here.
    """
    cfg = _cfg()
    arms = slot_arms(specs, canonical_horizon=canonical_horizon)
    n_applicable = sum(1 for a in arms if a.applicability == "applicable")
    budget = int(getattr(cfg, "MODEL_ZOO_WEEKLY_BUDGET", 0) or 0)

    if register is None:
        register = load_register(s3, bucket)
    if versions is None:
        versions = _list_registry_versions(s3, bucket)

    # label → the bundle dates that label was fitted on, oldest first.
    dates_by_label: dict[str, list[str]] = {}
    for v in versions or []:
        label, date = v.get("model_version"), v.get("date")
        if label and date:
            dates_by_label.setdefault(label, []).append(str(date))
    for label in dates_by_label:
        dates_by_label[label] = sorted(set(dates_by_label[label]))

    arm_id_by_label: dict[str, str] = {}
    for arm in arms:
        spec = recipe_spec(arm, cadence=_cadence(n_applicable, budget, arm.spec_id))
        from nousergon_lib.arena.arms import derive_arm_id

        arm_id = derive_arm_id(SLOT, arm.spec_id, spec)
        fitted = dates_by_label.get(arm.model_version_label, [])

        if arm_id not in register and not fitted:
            # A declared spec that has never been fitted is a SPEC, not yet an
            # arm. Registering it would make it an active arm with no fit to
            # vouch for, and §3 treats an unasserted fit exactly as a failed
            # one — so registering here would fail every cycle from the moment
            # a spec is added until its first training run lands. The arm
            # enters the register the week its first bundle does, which is also
            # the first week it has anything to say.
            log.info(
                "arena[M]: spec %r is declared but has no registered bundle — "
                "not yet an arm. It registers on its first fit.", arm.spec_id,
            )
            continue

        if arm_id not in register:
            prior = [
                a for a in register.all_arms()
                if register.state(a).record.name == arm.spec_id
                and register.state(a).retired_date is None
            ]
            created = fitted[0] if fitted else as_of
            register, _ = register.register(
                slot=SLOT, name=arm.spec_id, spec=spec, created_date=created,
                supersedes=prior[-1] if prior else None,
                notes=(
                    f"recipe: label={arm.model_version_label} horizon={arm.horizon_days}d "
                    f"cadence={spec['refit_cadence']}"
                    + (f"; supersedes {prior[-1]} (recipe changed)" if prior else "")
                ),
            )
            if prior:
                # The superseded recipe stops being an active arm the moment a
                # different recipe replaces it under the same name. It keeps
                # its series and is scored for its trailing window (§6.3).
                register = register.retire(
                    prior[-1], as_of,
                    reason=(
                        f"recipe changed — superseded by {arm_id}. Not a "
                        "retirement on evidence: the arm this replaces no "
                        "longer exists as a declared recipe (policy §3.1)."
                    ),
                )
            # Every bundle after the first is a refit of the new arm.
            for d in fitted[1:]:
                register = register.refit(arm_id, d, reason="scheduled refit (backfilled)")
        else:
            recorded = set(register.refits(arm_id))
            state = register.state(arm_id)
            for d in fitted:
                if d > state.record.created_date and d not in recorded:
                    register = register.refit(arm_id, d, reason="scheduled refit")

        if arm.applicability == "retired" and register.state(arm_id).retired_date is None:
            register = register.retire(
                arm_id, arm.retired_date or as_of,
                reason=f"declared status=retired in the spec register: {arm.reason}",
            )
        arm_id_by_label[arm.model_version_label] = arm_id

    return register, arm_id_by_label


def _list_registry_versions(s3, bucket: str) -> list:
    """Every registered bundle's lineage row. A read failure RAISES.

    The register is a producer artifact: degrading a failed read into "this arm
    has never been fitted" would append a false registration date and corrupt
    an append-only log permanently.
    """
    from model.registry import list_versions
    return list_versions(s3, bucket)


# ── 2. the score series (I9322: realized market-relative rank-IC) ────────────

def build_series(
    bucket: str, *, arm_id_by_label: dict, version_labels: dict,
    n_days: int = 90, horizon_days: int | None = None, s3_client=None,
    pairs: list | None = None,
):
    """One :class:`ArmSeries` per arm — per-date realized market-relative rank-IC.

    ``version_labels`` maps ``version_id -> model_version_label``, which is what
    folds a per-VERSION series onto its arm: every refit of one recipe carries
    the same label, so the arm's line is continuous across refits by
    construction. That is the recipe boundary paying for itself.

    A date on which an arm produced an artifact containing no usable rows is a
    MISS — the arm legitimately had nothing to say. A date on which it produced
    nothing at all, or whose 21-day forward window has not matured, is simply
    ABSENT from the series; it is neither a miss nor a zero, and the engine's
    common-window pairing handles it (policy §3).
    """
    from analysis import observe_leaderboard as ol
    from nousergon_lib.arena.window import ArmSeries

    cfg = _cfg()
    horizon_days = horizon_days or int(getattr(cfg, "FORWARD_DAYS", 21))
    if pairs is None:
        pairs = load_arm_pairs(
            bucket, version_ids=sorted(version_labels), n_days=n_days,
            horizon_days=horizon_days, s3_client=s3_client,
        )

    by_version = ol.per_date_rank_ic_by_version(pairs)

    scores: dict[str, dict[str, float]] = {}
    misses: dict[str, set] = {}
    for version_id, per_date in by_version.items():
        label = version_labels.get(version_id)
        arm_id = arm_id_by_label.get(label)
        if arm_id is None:
            # An artifact from a version no arm claims. Recorded, never folded
            # into some other arm's series — that is I9313's `thinktank_coverage`
            # defect and the 2026-07-28 false-provenance defect at once (§7.5).
            log.warning(
                "arena[M]: predictions from version_id=%r (label=%r) belong to "
                "no registered arm — NOT scored, and never attributed to "
                "another arm (policy §3, §7.5).", version_id, label,
            )
            continue
        for date, ic in per_date.items():
            if ic is None:
                misses.setdefault(arm_id, set()).add(date)
            else:
                # Two refits of one recipe never both produce on one date (a
                # single version serves a given day), so this cannot collide.
                scores.setdefault(arm_id, {})[date] = float(ic)

    out = {}
    for arm_id in set(arm_id_by_label.values()):
        arm_scores = scores.get(arm_id, {})
        arm_misses = frozenset(misses.get(arm_id, set()) - set(arm_scores))
        out[arm_id] = ArmSeries(arm_id=arm_id, scores=arm_scores, misses=arm_misses)
    return out


def load_arm_pairs(bucket: str, *, version_ids, n_days: int, horizon_days: int,
                   s3_client=None) -> list:
    """(date, ticker, predicted_alpha, realized_alpha, champion_version_id) rows.

    Reuses ``observe_leaderboard``'s loaders and
    ``variant_cutover_gate.compute_realized_alpha_for_pairs`` — the fleet's one
    realized market-relative alpha implementation. Nothing new is produced here.
    """
    from analysis import observe_leaderboard as ol
    from analysis.triple_barrier_cutover_runner import (
        _load_prices_from_s3, _load_sector_map_from_s3,
    )
    from analysis.variant_cutover_gate import compute_realized_alpha_for_pairs

    pairs = ol._load_live_pairs(bucket, n_days, s3_client=s3_client)
    for vid in version_ids:
        shadow = ol._load_shadow_pairs(bucket, vid, n_days, s3_client=s3_client)
        for p in shadow:
            # The shadow directory names the arm that produced the row. The
            # file's own `champion_version_id` is the LIVE champion at the time
            # — reading it here would attribute every shadow row to the
            # champion (policy §7.5: provenance true by construction).
            p["champion_version_id"] = vid
        pairs.extend(shadow)

    tickers = {p["ticker"] for p in pairs if p.get("ticker")}
    sector_map = _load_sector_map_from_s3(bucket, s3_client=s3_client)
    bench = set(sector_map.values()) | {"SPY"}
    prices = _load_prices_from_s3(bucket, tickers | bench, s3_client=s3_client)
    compute_realized_alpha_for_pairs(
        pairs, horizon_days=horizon_days, prices_by_ticker=prices,
        sector_map=sector_map,
    )
    return pairs


# ── 3. training integrity (policy §3 — Brian's ruling, 2026-08-29) ──────────

def training_statuses(manifests_by_arm: dict) -> dict:
    """A :class:`TrainingStatus` per arm, from what already measures soundness.

    ``manifests_by_arm`` maps ``arm_id -> the arm's most recent registry
    manifest`` (or None). No second measurement is written here: the manifest
    already carries ``arm_validity`` (``training/arm_validity.py``, the L2
    post-fit block whose ``constant_input_column`` check is what the seven
    hard-zeroed features of 2026-08-29 trip) and ``l1_fit_validity``
    (``training/l1_fit_validity.py``). This reads them.

    An arm with no manifest at all gets **no entry**, and the engine treats a
    missing status exactly as a failed one — an unasserted fit is a failed fit.
    That is the ruling's teeth and it is deliberate that it is not softened
    here.
    """
    engine = _arena()
    out = {}
    for arm_id, manifest in (manifests_by_arm or {}).items():
        if not manifest:
            continue
        reasons = []
        for block_name in ("arm_validity", "l1_fit_validity"):
            block = manifest.get(block_name) or {}
            if not isinstance(block, dict):
                continue
            for failure in block.get("failures") or []:
                reasons.append(f"{block_name}: {failure.get('reason') or failure}")
        # Input-side degradation is an unsound fit too: the 2026-08-29 run
        # reported unqualified success over a panel with seven dead features.
        if manifest.get("data_coverage_degraded"):
            reasons.append(
                "data_coverage_degraded=true — the panel this arm was fitted on "
                "was itself short/frozen/absent in part"
            )
        out[arm_id] = engine.TrainingStatus(
            arm_id=arm_id, ok=not reasons, reason="; ".join(reasons),
        )
    return out


# ── 4. serving preconditions (policy §5.3) ─────────────────────────────────

def serving_preconditions(
    *, arm_ids, manifests_by_arm: dict, incumbent_arm: str | None,
    served_metrics_by_arm: dict | None = None,
) -> dict:
    """Hard gates on SERVING, evaluated here and handed to the engine.

    Two, exactly as §5.3 names them, and NEITHER is folded into the ranking:

    1. **the behavioural veto** — ``training/promotion_behavioral_veto.py``,
       passed through UNMODIFIED and in its scale-DEPENDENT form. It is not
       normalised here and must never be: a standardized ratio would have
       passed both the collapsed 2026-08-28 model (0.943) and the 2026-08-21
       model that produced five live sessions with zero high-confidence names
       (0.973), because dividing by the spread divides the collapse away.
    2. **input completeness** — the arm's own manifest declaring whether the
       inputs it was fitted and is serving on are complete.

    An ``insufficient`` behavioural verdict is reported and NON-BLOCKING
    (§5.1: you cannot gate on a statistic you did not measure), and it is never
    rendered as a pass — it reaches the artifact as its own reason string.
    """
    engine = _arena()
    from training.promotion_behavioral_veto import evaluate_behavioral_veto

    served = served_metrics_by_arm or {}
    incumbent_manifest = (manifests_by_arm or {}).get(incumbent_arm)
    out: dict = {}
    for arm_id in arm_ids:
        manifest = (manifests_by_arm or {}).get(arm_id)
        checks = []

        verdict = evaluate_behavioral_veto(
            manifest, incumbent_manifest,
            candidate_served_metrics=served.get(arm_id),
            incumbent_served_metrics=served.get(incumbent_arm),
        )
        checks.append(engine.ServingPrecondition(
            name="behavioral_veto",
            passed=verdict.get("status") != "veto",
            reason=(
                "; ".join(v.get("reason", "") for v in verdict.get("vetoes") or [])
                or f"status={verdict.get('status')}; "
                   f"uncomputable={verdict.get('uncomputable')}"
            ),
        ))

        incomplete = bool((manifest or {}).get("data_coverage_degraded"))
        completeness = ((manifest or {}).get("data_completeness") or {})
        if isinstance(completeness, dict) and completeness.get("status") == "fail":
            incomplete = True
        checks.append(engine.ServingPrecondition(
            name="input_completeness",
            passed=not incomplete,
            reason=(
                "the arm's declared inputs are incomplete — an arm scored on "
                "partial inputs may rank first and still be unfit to trade "
                "(policy §5.3)" if incomplete else "inputs complete"
            ),
        ))
        out[arm_id] = tuple(checks)
    return out


# ── the cycle ───────────────────────────────────────────────────────────────

def _clip_diagnostics(series_by_arm, clip: float) -> dict:
    """What fraction of paired per-date differences reached the declared clip.

    The declared sub-Gaussian scale is only honest while essentially nothing
    saturates it, so the cycle records the number rather than asserting the
    choice was right (policy §5.0).
    """
    n_total = n_clipped = 0
    arms = sorted(series_by_arm)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            sa, sb = series_by_arm[a].scores, series_by_arm[b].scores
            for date in set(sa) & set(sb):
                n_total += 1
                if abs(sa[date] - sb[date]) >= clip:
                    n_clipped += 1
    return {
        "diff_clip": clip,
        "n_paired_diffs": n_total,
        "n_at_or_beyond_clip": n_clipped,
        "clip_rate": (n_clipped / n_total) if n_total else None,
        "rationale": DIFF_CLIP_RATIONALE,
    }


def run_model_slot_cycle(
    *, as_of: str, register, series_by_arm: dict, incumbent: str | None,
    preconditions: dict, training: dict, diagnostics: dict | None = None,
):
    """Run one M-slot cycle. The engine decides; this asserts §5.3's loud failure.

    ``TrainingIntegrityError`` is deliberately NOT caught: it must reach the
    Step Function as a FAILED state, not a degraded one (policy §3, §11).
    """
    engine = _arena()
    config = ARENA_CONFIG.resolve()
    cycle = engine.run_cycle(
        config=config, as_of=as_of, register=register,
        series_by_arm=series_by_arm, incumbent=incumbent,
        preconditions=preconditions, training=training,
    )
    doc = cycle.to_dict()
    doc["diff_clip_diagnostics"] = _clip_diagnostics(series_by_arm, config.diff_clip)
    doc["arena_config"] = {
        "alpha": config.alpha, "diff_clip": config.diff_clip,
        "variance_mode": config.variance_mode, "min_paired_dates": config.min_paired_dates,
        "cap": config.cap, "grace_weeks": config.grace_weeks,
        "min_active_arms": config.min_active_arms,
        "retired_trailing_cycles": config.retired_trailing_cycles,
        "retire_evidence": config.retire_evidence,
        "ranking_statistic": "realized-market-relative-rank-ic",
        "ranking_statistic_ruling": "alpha-engine-config-I9322",
    }
    if diagnostics:
        # CPCV IC and friends are RETAINED as diagnostics and are never the
        # ranking basis (I9322). Kept in their own namespace so no reader can
        # mistake one for the score.
        doc["retained_diagnostics"] = diagnostics
    return cycle, doc


def assert_servable(cycle) -> None:
    """Policy §5.3: a cycle where no arm may serve fails loud."""
    if cycle.decision.status == "unservable":
        raise ArenaSlotUnservable(
            f"M slot is UNSERVABLE as of {cycle.as_of}: {cycle.decision.reason}. "
            "Every arm failed a hard serving precondition, so there is no arm "
            "to point at. Continuing to serve a known-unfit arm is never the "
            "safer option (champion-challenger-policy §5.3)."
        )


def emit_cycle(s3, bucket: str, doc: dict, *, as_of: str) -> list:
    """Write the validated ``arena_cycle`` to its dated key and the mirror.

    Validation runs BEFORE the write: a slot that emits a malformed artifact
    every cycle is worse than one that emits nothing, because it reads as
    coverage (policy §11, ``principles.md`` §2.7).
    """
    from nousergon_lib.contracts import validate

    validate("arena_cycle", doc)
    body = json.dumps(doc, indent=2, default=str, sort_keys=False).encode("utf-8")
    keys = [f"{CYCLE_PREFIX}/{as_of}.json", f"{CYCLE_PREFIX}/latest.json"]
    for key in keys:
        s3.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/json")
        log.info("arena[M]: cycle written to s3://%s/%s", bucket, key)
    return keys


def run_slot(
    s3, bucket: str, *, as_of: str, specs=None, incumbent_version_id: str | None = None,
    n_days: int = 90, served_metrics_by_arm: dict | None = None,
    diagnostics: dict | None = None, write: bool = True,
) -> dict:
    """The whole M-slot cycle, end to end. One call from the rotation.

    Returns ``{"cycle", "doc", "register", "pointer_arm", "pointer_version_id",
    "bundle_by_arm", "keys"}``.

    ``pointer_version_id`` is the bundle to serve: the pointer arm's newest
    registered bundle. When the pointer did not move, that bundle may still be
    newer than what is serving — a **refit**, which is the arm executing the
    cadence its own recipe declares, not a promotion (policy §3.1). That is why
    there is no ``champion-arch-refresh`` kind any more and why one cannot be
    written: the concept it named has no event of its own.

    ``TrainingIntegrityError`` and :class:`ArenaSlotUnservable` both propagate.
    """
    versions = _list_registry_versions(s3, bucket)
    register, arm_id_by_label = build_register(
        s3, bucket, as_of=as_of, specs=specs, versions=versions,
    )
    label_by_arm = {arm: label for label, arm in arm_id_by_label.items()}

    # arm -> its newest registered bundle, and version_id -> label.
    bundle_by_arm: dict[str, dict] = {}
    version_labels: dict[str, str] = {}
    for v in versions or []:
        label, vid = v.get("model_version"), v.get("version_id")
        if not (label and vid):
            continue
        version_labels[vid] = label
        arm_id = arm_id_by_label.get(label)
        if arm_id is None:
            continue
        cur = bundle_by_arm.get(arm_id)
        key = (str(v.get("date") or ""), str(v.get("created_utc") or ""))
        if cur is None or key > cur["_key"]:
            bundle_by_arm[arm_id] = {**v, "_key": key}

    manifests_by_arm = {}
    for arm_id, bundle in bundle_by_arm.items():
        manifests_by_arm[arm_id] = _read_manifest(s3, bucket, bundle["version_id"])

    series_by_arm = build_series(
        bucket, arm_id_by_label=arm_id_by_label, version_labels=version_labels,
        n_days=n_days, s3_client=s3,
    )
    # Every arm the register says is scored this cycle must have a series, even
    # an empty one — the engine refuses a missing series, and rightly: an arm
    # silently absent from the scoring is the defect, not the omission.
    from nousergon_lib.arena.window import ArmSeries
    for arm_id in register.scored_arms(as_of, ARENA_CONFIG.retired_trailing_cycles):
        series_by_arm.setdefault(arm_id, ArmSeries(arm_id=arm_id, scores={}))
    for arm_id in list(series_by_arm):
        if arm_id not in register:
            series_by_arm.pop(arm_id)

    incumbent_arm = None
    if incumbent_version_id:
        incumbent_arm = arm_id_by_label.get(version_labels.get(incumbent_version_id))
        if incumbent_arm is None:
            log.warning(
                "arena[M]: serving version %r maps to no registered arm — the "
                "cycle runs with NO incumbent, which is reported as a bootstrap "
                "pointer rather than silently treated as a hold.",
                incumbent_version_id,
            )

    active = register.active_arms()
    cycle, doc = run_model_slot_cycle(
        as_of=as_of, register=register, series_by_arm=series_by_arm,
        incumbent=incumbent_arm,
        preconditions=serving_preconditions(
            arm_ids=active, manifests_by_arm=manifests_by_arm,
            incumbent_arm=incumbent_arm,
            served_metrics_by_arm=served_metrics_by_arm,
        ),
        training=training_statuses({a: manifests_by_arm.get(a) for a in active}),
        diagnostics=diagnostics,
    )
    assert_servable(cycle)

    # The engine's retirement verdicts are appended to the log here — the one
    # place a cap-with-grace retirement is ever written.
    for verdict in cycle.retirements:
        if verdict.retire and register.state(verdict.arm_id).retired_date is None:
            register = register.retire(verdict.arm_id, as_of, reason=verdict.reason)

    pointer_arm = cycle.decision.champion
    pointer_bundle = bundle_by_arm.get(pointer_arm) if pointer_arm else None
    keys = []
    if write:
        keys = emit_cycle(s3, bucket, doc, as_of=as_of)
        persist_register(s3, bucket, register)
    return {
        "cycle": cycle,
        "doc": doc,
        "register": register,
        "arm_id_by_label": arm_id_by_label,
        "label_by_arm": label_by_arm,
        "bundle_by_arm": bundle_by_arm,
        "pointer_arm": pointer_arm,
        "pointer_version_id": (pointer_bundle or {}).get("version_id"),
        "keys": keys,
    }


def _read_manifest(s3, bucket: str, version_id: str) -> dict | None:
    """One registry bundle's manifest. Absence is reported as None.

    A manifest that cannot be read yields no training status for its arm, and
    the engine treats a missing status exactly as a failed one — so an
    unreadable manifest fails the cycle rather than passing it. That is the
    correct direction and it is why this returns None instead of raising here.
    """
    try:
        body = s3.get_object(
            Bucket=bucket, Key=f"predictor/registry/{version_id}/manifest.json",
        )["Body"].read()
    except Exception as exc:  # noqa: BLE001 — surfaced as a missing training status
        log.warning(
            "arena[M]: manifest unreadable for %s (%s) — the arm will report NO "
            "training status, which the engine treats exactly as a FAILED fit "
            "(champion-challenger-policy §3).", version_id, exc,
        )
        return None
    return json.loads(body)


def persist_register(s3, bucket: str, register) -> str:
    """Write the append-only event log back. A write failure RAISES."""
    body = json.dumps(register.to_dicts(), indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=REGISTER_KEY, Body=body,
                  ContentType="application/json")
    log.info("arena[M]: register persisted to s3://%s/%s (%d events)",
             bucket, REGISTER_KEY, len(register.events))
    return REGISTER_KEY
