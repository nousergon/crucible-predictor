"""A cross-sectionally degenerate candidate must be refused at PROMOTION.

The 2026-09-08 `predictor-inference` page was raised by three shadow arms whose
L2 emits one alpha for 24 of 29 tickers, because every coefficient above 0.10
sits on a date-constant macro feature. `promotion_behavioral_veto`'s own
docstring states the doctrine this closes: "Refusal belongs at promotion, which
is here." The existing dispersion rules are RATIOS against the incumbent and so
cannot see this — a model can hold its dispersion ratio while carrying that
dispersion entirely in the time-series axis.
"""

from __future__ import annotations

from training.promotion_behavioral_veto import (
    FLOOR_VETO_METRICS,
    behavioral_metrics,
    evaluate_behavioral_veto,
)


def _manifest(share, **extra):
    m = {
        "output_distribution_gate": {"metrics": {"alpha_stdev": 0.017, "stdev_p_up": 0.071}},
        "behavioral_metrics": {"n_high_confidence": 30, "model_hit_rate_30d": 0.55},
    }
    if share is not None:
        m["behavioral_metrics"]["xsec_variance_share"] = share
    m["behavioral_metrics"].update(extra)
    return m


def test_the_metric_is_a_registered_floor_rule():
    assert "xsec_variance_share" in FLOOR_VETO_METRICS
    assert FLOOR_VETO_METRICS["xsec_variance_share"] == 0.10


def test_behavioral_metrics_surfaces_the_share_from_the_forward_slot():
    assert behavioral_metrics(_manifest(0.004))["xsec_variance_share"] == 0.004


def test_a_degenerate_candidate_is_vetoed():
    out = evaluate_behavioral_veto(_manifest(0.004), _manifest(0.42))
    assert out["status"] == "veto"
    hit = [v for v in out["vetoes"] if v["metric"] == "xsec_variance_share"]
    assert len(hit) == 1
    # The reason must name the actual defect, not the hit-rate boilerplate.
    reason = hit[0]["reason"].lower()
    assert "cross-section" in reason
    assert "coin flip" not in reason


def test_a_cross_sectionally_live_candidate_is_not_vetoed_on_this_rule():
    out = evaluate_behavioral_veto(_manifest(0.42), _manifest(0.38))
    assert out["status"] == "pass"
    assert out["measured"]["xsec_variance_share"] == {"candidate": 0.42, "floor": 0.10}


def test_an_absent_share_is_uncomputable_never_a_pass_of_this_rule():
    out = evaluate_behavioral_veto(_manifest(None), _manifest(None))
    assert "xsec_variance_share" in out["uncomputable"]
    assert not [v for v in out["vetoes"] if v["metric"] == "xsec_variance_share"]


def test_the_rule_is_absolute_not_a_ratio_against_the_incumbent():
    """Both arms degenerate must still veto — a ratio rule would pass this."""
    out = evaluate_behavioral_veto(_manifest(0.004), _manifest(0.003))
    assert out["status"] == "veto"
    assert [v for v in out["vetoes"] if v["metric"] == "xsec_variance_share"]


def test_the_existing_hit_rate_floor_keeps_its_own_reason_text():
    out = evaluate_behavioral_veto(
        _manifest(0.42, model_hit_rate_30d=0.41), _manifest(0.38),
    )
    hit = [v for v in out["vetoes"] if v["metric"] == "model_hit_rate_30d"]
    assert len(hit) == 1
    assert "coin flip" in hit[0]["reason"]
