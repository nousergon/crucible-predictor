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
incumbent's registry bundle, under
``output_distribution_gate.metrics`` — so the dispersion-collapse rule is live,
and it refuses the 2026-08-21 candidate (0.060 against the incumbent's 0.191,
a ratio of 0.31).
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


def behavioral_metrics(manifest: dict | None) -> dict:
    """Every behavioral metric a manifest carries, flattened to one namespace.

    Read from ``output_distribution_gate.metrics`` (what the trainer records
    today) and from a top-level ``behavioral_metrics`` block (the forward slot
    for the served-side metrics I9024 section 4 names, once a producer emits
    them onto the manifest). Later sources win, so an explicit block overrides a
    gate metric of the same name.
    """
    out: dict = {}
    manifest = manifest or {}
    gate = (manifest.get("output_distribution_gate") or {}).get("metrics") or {}
    if isinstance(gate, dict):
        out.update(gate)
    explicit = manifest.get("behavioral_metrics") or {}
    if isinstance(explicit, dict):
        out.update(explicit)
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
) -> dict:
    """Return the behavioral verdict for one candidate against the incumbent.

    ``{"status": "veto" | "pass" | "insufficient", "vetoes": [...],
       "uncomputable": [...], "measured": {...}}``

    ``veto`` is BLOCKING and is not gated on any config flag. ``insufficient``
    means nothing at all could be measured — reported, non-blocking, and never
    rendered as a pass.
    """
    cand = behavioral_metrics(candidate_manifest)
    inc = behavioral_metrics(incumbent_manifest)

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
    }
