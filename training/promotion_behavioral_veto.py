"""training/promotion_behavioral_veto.py — the BEHAVIORAL veto on promotion.

alpha-engine-config-I9024 section 4.

A model can win on IC and still be worse to trade. IC is rank-only: it does not
see dispersion, and dispersion is what the executor actually consumes for entry
ordering and sizing. The 2026-08-21 promotion is the worked example — the
candidate reported a 2.3x CPCV IC improvement while collapsing output dispersion
4-6x, and every gate that could have seen the collapse was either observe-only
or looking at the wrong number.

This is champion-challenger-policy section 7.3 applied to the M slot: gate on the
invariant the actor consumes, not on a transform of it. It is an ABSOLUTE veto —
it does not sit behind a soak flag, and it outranks the CPCV ranking entirely.
A candidate that trips any rule is not promotable however well it scored.

The sibling inference-side dispersion check (crucible-predictor-PR569,
alpha-engine-config-I9019) is deliberately observe-only: halting at inference
means no predictions are written and the executor falls back to the prior day,
which is the shape of the 2026-06-29 false-halt. Refusal belongs at promotion,
which is here.

Computability
-------------
Three of the four rules named in I9024 section 4 read metrics that the TRAINING
manifest does not yet carry — they are produced by the inference path, per
served day, so a candidate that has never served has no value for them. Those
are reported as ``uncomputable`` and named, never silently skipped and never
counted as a pass (champion-challenger-policy section 5.1 and section 7.2). The
moment a producer starts writing one onto the manifest, the rule arms itself
with no code change here.

``stdev_p_up`` IS carried today, by both the candidate's manifest and the
incumbent's registry bundle, under ``output_distribution_gate.metrics``.

The SERVED slice (alpha-engine-config-I9061)
--------------------------------------------
The manifest metrics are not enough, and the 2026-08-21 rotation is the proof.
Measured on the two registry bundles:

    v3.0-meta-2026-08-14-119e069b (incumbent)  stdev_p_up 0.113644
    v3.0-meta-2026-08-21-7d3d1cce (candidate)  stdev_p_up 0.130132

ratio 1.145 — a comfortable PASS against the 0.5 floor, on the candidate whose
served output then collapsed 4-6x and produced zero high-confidence names for
five sessions. The manifest number could not have seen it: it comes from
``model.output_distribution_gate.validate_calibrator_distribution``, a sweep of
25 SYNTHETIC alphas through the calibrator alone, which never touches the
meta-model, the universe, or the selection rule.

So the caller supplies ``candidate_served_metrics`` / ``incumbent_served_metrics``
— the same three metric names, measured over the ~30 names the executor would
actually trade, by ``training/served_slice_dispersion.py``. They are merged LAST
and therefore win over anything a manifest carries, because they are the
measurement of the quantity the executor consumes and the manifest values are a
transform of a transform of it (champion-challenger-policy section 7.3).

On the real artifacts those served-slice values are:

    incumbent  alpha_stdev 0.016624  stdev_p_up 0.071064  n_high_confidence 30
    candidate  alpha_stdev 0.005429  stdev_p_up 0.041531  n_high_confidence  0

which refuses the 2026-08-21 candidate twice over: alpha_stdev at 33% of the
incumbent's, and zero high-confidence names. NO threshold moved to achieve that.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# A candidate whose dispersion falls below this fraction of the incumbent's is
# refused. Half is I9024 section 4's stated bar: not a tuned threshold, a
# statement that halving the spread the executor ranks on is a different model,
# not a better one.
MIN_DISPERSION_RATIO = 0.5

# Metrics compared as a RATIO against the incumbent's value for the same metric.
DISPERSION_METRICS: tuple[str, ...] = ("alpha_stdev", "stdev_p_up")

# Metrics vetoed on an absolute zero — no incumbent value needed.
ZERO_VETO_METRICS: tuple[str, ...] = ("n_high_confidence",)

# Metrics vetoed on falling below an absolute floor.
#
# ``xsec_variance_share`` (training/xsec_variance_share.py) is the share of the
# fitted L2's variance that lives WITHIN a date. It is deliberately an absolute
# floor and not a ratio against the incumbent: the 2026-09-08 arms carried a
# perfectly respectable dispersion ratio while holding essentially all of that
# dispersion in the time-series axis, so every ratio rule here passed them.
# A model below the floor moves every name together and separates none —
# whatever its IC, it is not doing the cross-sectional task.
FLOOR_VETO_METRICS: dict[str, float] = {
    "model_hit_rate_30d": 0.50,
    "xsec_variance_share": 0.10,
}

# Per-metric refusal text. A floor rule that reports the wrong reason sends the
# reader to the wrong defect — which is exactly how the 2026-09-08 collapse was
# read as a calibrator problem for the first hour of its investigation.
_FLOOR_VETO_REASON: dict[str, str] = {
    "model_hit_rate_30d": (
        "{name} {c} is below the {floor} floor — realized direction accuracy "
        "at or under a coin flip"
    ),
    "xsec_variance_share": (
        "{name} {c} is below the {floor} floor — less than {floor:.0%} of this "
        "model's fitted variance is CROSS-SECTIONAL, so it moves every name "
        "together and separates none. Its served alphas collapse onto a "
        "handful of values and any calibrator applied to them plateaus, which "
        "surfaces downstream as a calibrator-collapse page on live tickers "
        "(measured 2026-09-08 on spec-sota-combine-2026-07-{{24,29,30}}: 6 "
        "distinct predicted_alpha across 29 names)"
    ),
}
_FLOOR_VETO_REASON_DEFAULT = "{name} {c} is below the {floor} floor"

# ── the MAGNITUDE leg (alpha-engine-config-I10185) ───────────────────────────
#
# ``xsec_variance_share`` above is a RATIO, and PR611's module docstring names
# its invariance under a uniform coefficient rescale as a deliberate design
# property. So it structurally cannot see a SCALE collapse — a model that keeps
# 70% of its variance inside the date and simply produces 4.7x less of it.
# Measured on the live champion v3.0-meta-2026-09-04-cc3271ea: share 0.702720
# (passes the 0.10 floor by 7x) with xsec_sd 0.010350, against the 2026-08-14
# incumbent's 0.048588 on the IDENTICAL panel. Its first served batch collapsed
# to alpha_stdev 0.005819 and produced zero high-confidence names.
#
# The absolute floor is not a new number. It is the predictor's own absolute
# serving ``alpha_stdev`` floor (alpha-engine-config-I9267, "derived from the
# measured healthy population"), applied to the TRAINING panel — where the same
# quantity is knowable before the model ever serves a batch. On the champion it
# was already below that floor at fit time and nothing looked.
XSEC_SD_ABSOLUTE_FLOOR: float = 0.015

# The relative floor is this module's existing dispersion bar (MIN_DISPERSION_
# RATIO), reused deliberately: halving the spread the executor ranks on is a
# different model, not a better one.
XSEC_SD_MIN_RATIO_VS_INCUMBENT: float = 0.5

# The incumbent's xsec_sd RECOMPUTED ON THE CANDIDATE'S OWN PANEL. The panel
# must be held constant or the comparison measures the data, not the model —
# which is why this is a field of the CANDIDATE's manifest (written by
# training/xsec_magnitude.py at fit time) and is never taken from the incumbent
# manifest's own ``xsec_sd``.
XSEC_SD_INCUMBENT_SAME_PANEL = "xsec_sd_incumbent_same_panel"

# The metric names a served-slice measurement may contribute. Restricting the
# merge to these keeps the measurement's bookkeeping fields (``n_dates``,
# ``top_n``, ``min_confidence``) out of the metric namespace the rules read, so
# a future field cannot accidentally arm a rule it was never meant to feed.
_SERVED_METRIC_NAMES: frozenset = frozenset(
    DISPERSION_METRICS + ZERO_VETO_METRICS + tuple(FLOOR_VETO_METRICS)
)

# ``xsec_sd`` and its same-panel reference are FIT-TIME properties of the
# training design matrix. A served-slice measurement has no such quantity, so
# admitting one here would rank a served number against a fitted one — the
# cross-basis comparison the served-slice merge exists to prevent.
assert "xsec_sd" not in _SERVED_METRIC_NAMES


def behavioral_metrics(manifest: dict | None,
                       served_metrics: dict | None = None) -> dict:
    """Every behavioral metric available for one version, in one namespace.

    Three sources, in ascending precedence:

    1. ``output_distribution_gate.metrics`` — what the trainer records today.
       A synthetic calibrator sweep; see this module's docstring for why it is
       the WEAKEST of the three and why it must never be the only one.
    2. a top-level ``behavioral_metrics`` block on the manifest — the forward
       slot for any served-side metric a producer starts emitting.
    3. ``served_metrics`` — measured at promotion time over the batch the
       executor would actually trade (``training/served_slice_dispersion.py``).
       Highest precedence, because it is the only one of the three that measures
       the invariant the actor consumes (champion-challenger-policy section 7.3).
    """
    out: dict = {}
    manifest = manifest or {}
    gate = (manifest.get("output_distribution_gate") or {}).get("metrics") or {}
    if isinstance(gate, dict):
        out.update(gate)
    explicit = manifest.get("behavioral_metrics") or {}
    if isinstance(explicit, dict):
        out.update(explicit)
    if isinstance(served_metrics, dict):
        out.update({
            k: v for k, v in served_metrics.items()
            if k in _SERVED_METRIC_NAMES
        })
    return out


def _as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN is not a measurement



def _evaluate_xsec_magnitude(cand: dict) -> tuple[list[dict], dict, list[str]]:
    """The MAGNITUDE leg (alpha-engine-config-I10185): is the candidate's
    cross-sectional dispersion, measured on its OWN training panel, too small in
    absolute terms AND too small against the incumbent scored on that same panel?

    Returns ``(vetoes, measured, uncomputable)``.

    Both sub-legs must fire, and that conjunction is measured, not assumed.
    Every registered ``v3.0-meta`` coefficient vector (20, 2026-06-06 ..
    2026-09-04) was scored on the one persisted fitted panel (2026-09-04), so
    the panel is held constant and the number measures the model:

    * an absolute floor of 0.015 puts exactly two below it — 2026-09-04
      (0.010350, the case this leg exists for) and 2026-08-21 (0.014842, the
      worked-example bad promotion this module's docstring already names) —
      with the nearest passing arm at 0.020270, 1.35x the floor;
    * a RATIO-only rule would additionally have refused 2026-06-26 (0.026215,
      comfortably above the floor, 0.393x an unusually wide predecessor), a
      candidate with no evidence of any defect.

    Two conditions arm the absolute leg ON ITS OWN, because in both the relative
    leg has no valid reference and ``unmeasurable`` is never a pass:

    1. no same-panel incumbent number at all;
    2. a same-panel incumbent number that is itself below the absolute floor —
       a broken reference cannot exonerate anything, and this is what stops a
       joint drift (both models shrinking together) from walking under the gate.
    """
    vetoes: list[dict] = []
    measured: dict = {}
    uncomputable: list[str] = []

    c = _as_float(cand.get("xsec_sd"))
    ref = _as_float(cand.get(XSEC_SD_INCUMBENT_SAME_PANEL))
    if ref is None or ref <= 0:
        uncomputable.append(XSEC_SD_INCUMBENT_SAME_PANEL)
    if c is None:
        uncomputable.append("xsec_sd")
        return vetoes, measured, uncomputable

    ratio = (c / ref) if (ref is not None and ref > 0) else None
    measured["xsec_sd"] = {
        "candidate": c,
        "floor": XSEC_SD_ABSOLUTE_FLOOR,
        "incumbent_same_panel": ref,
        "ratio_vs_incumbent_same_panel": (
            None if ratio is None else round(ratio, 6)
        ),
        "min_ratio": XSEC_SD_MIN_RATIO_VS_INCUMBENT,
        # WHICH panel both numbers came from. A reader must never have to
        # assume this: the whole point of the leg is that the incumbent's
        # stored, differently-scaled number is the wrong comparison.
        "basis": "candidate_training_panel",
    }

    below_floor = c < XSEC_SD_ABSOLUTE_FLOOR
    if not below_floor:
        return vetoes, measured, uncomputable

    common = (
        f"xsec_sd {c} is below the {XSEC_SD_ABSOLUTE_FLOOR} floor — the "
        f"cross-sectional standard deviation of this model's own fitted output, "
        f"on its own training panel, is smaller than the absolute alpha_stdev "
        f"floor its predictions must clear to be served at all "
        f"(alpha-engine-config-I9267). This is a SCALE collapse: the model can "
        f"still rank names (xsec_variance_share is a ratio and passes) while "
        f"producing far too little dispersion for the executor to size on"
    )
    if ref is None or ref <= 0:
        vetoes.append({
            "metric": "xsec_sd", "candidate": c,
            "rule": f">= {XSEC_SD_ABSOLUTE_FLOOR}",
            "reason": (
                f"{common}. No incumbent xsec_sd was recomputed on this "
                f"candidate's panel, so the relative leg could not be "
                f"evaluated and cannot exonerate it "
                f"(alpha-engine-config-I10185)"
            ),
        })
        return vetoes, measured, uncomputable

    if ref < XSEC_SD_ABSOLUTE_FLOOR:
        vetoes.append({
            "metric": "xsec_sd", "candidate": c, "incumbent_same_panel": ref,
            "ratio": None if ratio is None else round(ratio, 6),
            "rule": f">= {XSEC_SD_ABSOLUTE_FLOOR}",
            "reason": (
                f"{common}. The incumbent's same-panel xsec_sd ({ref}) is "
                f"ITSELF below the floor, so it is not a valid reference — a "
                f"broken reference cannot exonerate a candidate, and a drift "
                f"in which both models shrink together must not walk under "
                f"this gate (alpha-engine-config-I10185)"
            ),
        })
        return vetoes, measured, uncomputable

    if ratio is not None and ratio < XSEC_SD_MIN_RATIO_VS_INCUMBENT:
        vetoes.append({
            "metric": "xsec_sd", "candidate": c, "incumbent_same_panel": ref,
            "ratio": round(ratio, 6),
            "rule": (
                f">= {XSEC_SD_ABSOLUTE_FLOOR} OR >= "
                f"{XSEC_SD_MIN_RATIO_VS_INCUMBENT}x the incumbent on the same panel"
            ),
            "reason": (
                f"{common}, and it is {ratio:.0%} of the incumbent's "
                f"{ref} measured on the IDENTICAL panel (so the comparison is "
                f"of the models, not of the data). Measured 2026-09-08: the "
                f"champion that shipped with exactly this shape (0.010350 vs "
                f"0.048588) collapsed its first served batch to alpha_stdev "
                f"0.005819 with zero high-confidence names "
                f"(alpha-engine-config-I10185)"
            ),
        })
    return vetoes, measured, uncomputable


def evaluate_behavioral_veto(
    candidate_manifest: dict | None,
    incumbent_manifest: dict | None,
    *,
    min_dispersion_ratio: float = MIN_DISPERSION_RATIO,
    candidate_served_metrics: dict | None = None,
    incumbent_served_metrics: dict | None = None,
) -> dict:
    """Return the behavioral verdict for one candidate against the incumbent.

    ``{"status": "veto" | "pass" | "insufficient", "vetoes": [...],
       "uncomputable": [...], "measured": {...}, "served_slice_metrics": [...]}``

    ``veto`` is BLOCKING and is not gated on any config flag. ``insufficient``
    means nothing at all could be measured — reported, non-blocking, and never
    rendered as a pass.

    ``candidate_served_metrics`` / ``incumbent_served_metrics`` are the
    promotion-time served-slice measurement (alpha-engine-config-I9061). A
    metric is compared like-for-like or not at all: a rule only reads the
    served-slice value for the candidate when the incumbent's SAME metric also
    came from a served-slice measurement, so a served number is never ranked
    against a synthetic-sweep one.
    """
    served_both = sorted(
        (set((candidate_served_metrics or {})) & set((incumbent_served_metrics or {})))
        & _SERVED_METRIC_NAMES
    )
    cand_served = {k: (candidate_served_metrics or {})[k] for k in served_both}
    inc_served = {k: (incumbent_served_metrics or {})[k] for k in served_both}
    # A ZERO_VETO / FLOOR_VETO rule needs no incumbent counterpart, so the
    # candidate's served value arms it on its own.
    for name in tuple(ZERO_VETO_METRICS) + tuple(FLOOR_VETO_METRICS):
        if name in (candidate_served_metrics or {}):
            cand_served[name] = (candidate_served_metrics or {})[name]
    cand = behavioral_metrics(candidate_manifest, cand_served)
    inc = behavioral_metrics(incumbent_manifest, inc_served)

    vetoes: list[dict] = []
    uncomputable: list[str] = []
    measured: dict = {}

    for name in DISPERSION_METRICS:
        c, i = _as_float(cand.get(name)), _as_float(inc.get(name))
        if c is None or i is None or i <= 0:
            uncomputable.append(name)
            continue
        ratio = c / i
        measured[name] = {"candidate": c, "incumbent": i, "ratio": round(ratio, 6)}
        if ratio < min_dispersion_ratio:
            vetoes.append({
                "metric": name, "candidate": c, "incumbent": i,
                "ratio": round(ratio, 6), "rule": f"ratio >= {min_dispersion_ratio}",
                "reason": (
                    f"{name} collapsed to {ratio:.0%} of the incumbent "
                    f"({c} vs {i}) — the executor ranks and sizes on this "
                    f"spread, so halving it is a different model, not a "
                    f"better one (alpha-engine-config-I9024 s4)"
                ),
            })

    for name in ZERO_VETO_METRICS:
        c = _as_float(cand.get(name))
        if c is None:
            uncomputable.append(name)
            continue
        measured[name] = {"candidate": c}
        if c == 0:
            vetoes.append({
                "metric": name, "candidate": c, "rule": "> 0",
                "reason": (
                    f"{name} is zero — the model emits no actionable "
                    f"high-confidence names at all"
                ),
            })

    for name, floor in FLOOR_VETO_METRICS.items():
        c = _as_float(cand.get(name))
        if c is None:
            uncomputable.append(name)
            continue
        measured[name] = {"candidate": c, "floor": floor}
        if c < floor:
            vetoes.append({
                "metric": name, "candidate": c, "rule": f">= {floor}",
                "reason": _FLOOR_VETO_REASON.get(
                    name, _FLOOR_VETO_REASON_DEFAULT,
                ).format(name=name, c=c, floor=floor),
            })

    # alpha-engine-config-I10185 — the MAGNITUDE leg. Independent of the
    # xsec_variance_share floor above: a candidate must clear BOTH.
    _mag_vetoes, _mag_measured, _mag_uncomputable = _evaluate_xsec_magnitude(cand)
    vetoes.extend(_mag_vetoes)
    measured.update(_mag_measured)
    uncomputable.extend(_mag_uncomputable)

    if vetoes:
        status = "veto"
    elif measured:
        status = "pass"
    else:
        status = "insufficient"
    return {
        "status": status,
        "vetoes": vetoes,
        "uncomputable": sorted(set(uncomputable)),
        "measured": measured,
        "min_dispersion_ratio": min_dispersion_ratio,
        # WHICH metrics were decided on a served-slice measurement rather than
        # on a manifest number. Recorded so a reader of the leaderboard can tell
        # a verdict about the traded batch from a verdict about a synthetic
        # calibrator sweep, without having to know which fields exist.
        "served_slice_metrics": sorted(set(cand_served)),
    }
