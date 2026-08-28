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
FLOOR_VETO_METRICS: dict[str, float] = {"model_hit_rate_30d": 0.50}

# The metric names a served-slice measurement may contribute. Restricting the
# merge to these keeps the measurement's bookkeeping fields (``n_dates``,
# ``top_n``, ``min_confidence``) out of the metric namespace the rules read, so
# a future field cannot accidentally arm a rule it was never meant to feed.
_SERVED_METRIC_NAMES: frozenset = frozenset(
    DISPERSION_METRICS + ZERO_VETO_METRICS + tuple(FLOOR_VETO_METRICS)
)


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
                "reason": (
                    f"{name} {c} is below the {floor} floor — realized "
                    f"direction accuracy at or under a coin flip"
                ),
            })

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
