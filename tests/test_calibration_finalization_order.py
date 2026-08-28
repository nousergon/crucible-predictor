"""The variance fallback must actually reach the batch it claims to repair.

alpha-engine-config-I9086. Regression cover for the 2026-08-28 live incident:

  * ERROR  "VARIANCE FALLBACK ENGAGED: calibrator outputs collapsed to 2 unique
    p_up bins across 30 tickers ... Falling through to linear heuristic rescale
    to recover variance for today's batch."
  * WARNING "output_distribution_gate_shadow_platt FAILED (direction_skew):
    direction skew 100.00% (n_up=30, n_down=0)"

Both came from one mechanism. ``_rescale_cross_sectional`` ran BEFORE
``_apply_level_neutralization``, so:

  1. the collapse was detected on p_up derived from the pre-centering alpha;
  2. the fallback's linear recovery was then overwritten by
     ``calibrate_prediction(centered)`` — the same collapsed calibrator;
  3. ``p_up_platt`` was never re-derived on the centered alpha at all, so the
     config#1176 shadow soak compared Platt(raw) against isotonic(centered) and
     reported the batch's common-mode macro level as a Platt direction skew.

These tests are RED against the pre-fix tree.
"""

from __future__ import annotations


class _CollapsedCalibrator:
    """Isotonic-shaped: maps every alpha in a narrow band onto one step.

    Mirrors the live 2026-08-28 shape, where 30 raw alphas spanning
    [+0.00104, +0.03190] landed on 2 unique p_up values.
    """

    method = "isotonic"
    is_fitted = True
    _ece_after = 0.01

    def calibrate_prediction(self, alpha, label_clip=0.15):
        p_up = 0.5658 if alpha >= 0.0 else 0.4637
        return {
            "p_up": p_up,
            "p_down": round(1.0 - p_up, 4),
            "predicted_direction": "UP" if alpha >= 0 else "DOWN",
            "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
        }


class _PlattShadow:
    """Smooth logistic — strictly monotonic, so it never plateaus."""

    method = "platt"
    is_fitted = True

    def calibrate_prediction(self, alpha, label_clip=0.15):
        p_up = 0.5 + 2.197 * float(alpha)
        p_up = min(max(p_up, 0.0), 1.0)
        return {
            "p_up": round(p_up, 4),
            "p_down": round(1.0 - p_up, 4),
            "predicted_direction": "UP" if alpha >= 0 else "DOWN",
            "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
        }


class _Ctx:
    def __init__(self, predictions, calibrator, shadow_calibrator=None):
        self.predictions = predictions
        self.calibrator = calibrator
        self.shadow_calibrator = shadow_calibrator
        self.level_neutralization = None
        self.calibration_degradation = None


def _live_20260828_batch():
    """The measured live shape: 30 names, every raw alpha POSITIVE.

    mean +0.016410, stdev 0.007099, min +0.001040, max +0.031895 — a
    common-mode macro level 2.31x the cross-sectional spread.
    """
    lo, hi, n = 0.001040, 0.031895, 30
    step = (hi - lo) / (n - 1)
    out = []
    for i in range(n):
        a = lo + i * step
        out.append({
            "ticker": f"T{i:02d}",
            "predicted_alpha": a,
            "p_up": 0.5658,
            "p_down": 0.4342,
            "predicted_direction": "UP",
            "prediction_confidence": 0.1316,
            "p_up_platt": round(0.5 + 2.197 * a, 4),
        })
    return out


def _finalize(preds, *, enabled, shadow=None, monkeypatch=None):
    from inference.stages import run_inference as ri

    ctx = _Ctx(preds, _CollapsedCalibrator(), shadow_calibrator=shadow)
    monkeypatch.setattr(ri.cfg, "XSEC_DEMEAN_ALPHA_ENABLED", enabled, raising=False)
    ri._finalize_calibration(ctx)
    return ctx


def test_fallback_output_survives_neutralization(monkeypatch):
    """The recovered variance must be what the artifact carries."""
    ctx = _finalize(_live_20260828_batch(), enabled=True, monkeypatch=monkeypatch)

    assert ctx.level_neutralization["applied"] is True

    p_ups = [p["p_up"] for p in ctx.predictions]
    # The collapsed calibrator can only ever emit 0.5658 / 0.4637. If the
    # fallback's linear rescale had been overwritten (the pre-fix behaviour),
    # every served p_up would be drawn from that two-element set.
    assert not set(p_ups) <= {0.5658, 0.4637}, (
        "The variance fallback engaged and its output was then overwritten by "
        "the collapsed calibrator — the ERROR's promised recovery never "
        "reached the batch."
    )
    assert len(set(p_ups)) > 2, (
        f"Fallback must recover cross-sectional variance; got {len(set(p_ups))} "
        f"unique p_up bins"
    )
    assert all(p["calibration_basis"] == "linear_heuristic_fallback"
               for p in ctx.predictions)


def test_fallback_p_up_is_sign_coherent_with_served_alpha(monkeypatch):
    """Centering first makes direction and p_up agree by construction.

    On the live 2026-08-28 artifact 4 of 30 rows were `DOWN` with `p_up > 0.5`
    (THC, CART, ANF, RNR) because direction came from the centered alpha's sign
    and p_up came from an isotonic step that does not cross 0.5 at alpha=0.
    """
    ctx = _finalize(_live_20260828_batch(), enabled=True, monkeypatch=monkeypatch)
    incoherent = [
        p["ticker"] for p in ctx.predictions
        if (p["p_up"] >= 0.5) != (p["predicted_direction"] == "UP")
    ]
    assert not incoherent, f"p_up disagrees with predicted_direction on {incoherent}"


def test_degradation_is_recorded_on_the_context(monkeypatch):
    ctx = _finalize(_live_20260828_batch(), enabled=True, monkeypatch=monkeypatch)
    deg = ctx.calibration_degradation
    assert deg is not None, "A collapsed calibrator must leave a durable mark"
    assert deg["degraded"] is True
    assert deg["reason"] == "calibrator_collapse"
    assert deg["basis"] == "linear_heuristic_fallback"
    assert deg["n_unique_p_up_pre_fallback"] == 2
    assert deg["batch_size"] == 30


def test_healthy_batch_records_an_undegraded_verdict(monkeypatch):
    """No data is never rendered as green (principle 7): the healthy path
    records `degraded: False` rather than leaving the field absent."""
    class _Healthy(_CollapsedCalibrator):
        def calibrate_prediction(self, alpha, label_clip=0.15):
            p_up = round(min(max(0.5 + 8.0 * float(alpha), 0.0), 1.0), 4)
            return {
                "p_up": p_up,
                "p_down": round(1.0 - p_up, 4),
                "predicted_direction": "UP" if alpha >= 0 else "DOWN",
                "prediction_confidence": round(abs(p_up - 0.5) * 2.0, 4),
            }

    from inference.stages import run_inference as ri
    preds = _live_20260828_batch()
    ctx = _Ctx(preds, _Healthy())
    monkeypatch.setattr(ri.cfg, "XSEC_DEMEAN_ALPHA_ENABLED", True, raising=False)
    ri._finalize_calibration(ctx)

    assert ctx.calibration_degradation["degraded"] is False
    assert ctx.calibration_degradation["basis"] == "calibrator"
    assert all(p["calibration_basis"] == "calibrator" for p in ctx.predictions)


def test_shadow_platt_is_rederived_on_the_centered_alpha(monkeypatch):
    """The config#1176 soak must compare like with like.

    Platt on the RAW 2026-08-28 alphas puts all 30 names above 0.5
    (direction_skew 1.00 — the WARNING). Platt on the CENTERED alphas, which is
    what the live isotonic p_up is derived from, splits the batch.
    """
    ctx = _finalize(
        _live_20260828_batch(), enabled=True,
        shadow=_PlattShadow(), monkeypatch=monkeypatch,
    )

    assert ctx.level_neutralization["shadow_platt_basis"] == "centered"

    platt = [p["p_up_platt"] for p in ctx.predictions]
    n_up = sum(1 for v in platt if v >= 0.5)
    assert n_up != len(platt), (
        "Shadow-Platt still reports a 100% direction skew — it is being read "
        "off the pre-centering alpha, so the shadow gate is measuring the "
        "batch's common-mode macro level, not the calibration method."
    )
    assert abs(n_up / len(platt) - 0.5) <= 0.15, (
        f"centered Platt skew {n_up}/{len(platt)} should be near balanced"
    )
    # The pre-centering value is preserved so the soak keeps its history.
    assert all("p_up_platt_raw" in p for p in ctx.predictions)


def test_shadow_platt_untouched_when_neutralization_is_disabled(monkeypatch):
    """With centering off, `predicted_alpha` IS the raw alpha, so the shadow
    stays on the raw vintage and the two are still comparable."""
    ctx = _finalize(
        _live_20260828_batch(), enabled=False,
        shadow=_PlattShadow(), monkeypatch=monkeypatch,
    )
    assert ctx.level_neutralization["shadow_platt_basis"] == "raw"
    assert all("p_up_platt_raw" not in p for p in ctx.predictions)
