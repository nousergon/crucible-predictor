"""
model/calibrator.py — Probability calibration for GBM alpha predictions.

Replaces the linear confidence mapping (p_up = 0.5 + alpha / (2 * LABEL_CLIP))
with a proper calibration model fitted on walk-forward out-of-sample predictions.

Two methods supported:
  - "platt"    : Logistic regression (Platt scaling) — parametric, stable with small N
  - "isotonic" : Isotonic regression — non-parametric, more flexible, needs more data

The calibrator maps raw continuous alpha predictions to calibrated P(UP) values
that match empirical hit rates: when the model says 70% confidence, the stock
actually goes UP ~70% of the time.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nousergon_lib.quant.stats.calibration import (
    expected_calibration_error as _lib_expected_calibration_error,
)

log = logging.getLogger(__name__)


def derive_direction(alpha: float | None) -> str:
    """Binary predicted direction = the SIGN of the predicted alpha.

    config#1815 (2026-07-06): the single source of truth for
    ``predicted_direction`` across every derivation site (calibrator,
    linear fallback, cross-sectional rescale, level-neutralization).
    Direction is the model's cross-sectional opinion — NOT
    ``p_up >= 0.5``, because the isotonic p_up is a calibrated
    P(beat market) whose 0.5-crossing sits away from α=0 under an
    asymmetric realized base rate (~60% of names underperform SPY at
    the 21d horizon). Argmaxing it produced the all-UP/all-DOWN
    headline flips (24:2 → 2:26) and positive-α-but-DOWN rows.

    Binary by construction: FLAT is not a class (ratified 2026-07-06;
    ``p_flat`` remains as a permanently-0.0 additive-schema shadow).
    α == 0 / None maps to DOWN — no opinion is not a buy thesis.
    """
    return "UP" if (alpha or 0.0) > 0 else "DOWN"


class PlattCalibrator:
    """Calibrate raw GBM alpha scores to P(direction=UP)."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method!r} (expected 'platt' or 'isotonic')")
        self.method = method
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._ece_before = None  # ECE of linear calibration (baseline)
        self._ece_after = None   # IN-SAMPLE ECE of the fitted calibrator
        # alpha-engine-config-I10181 — the number that can actually go bad:
        # ECE on a held-out, time-ordered, embargoed tail. None (with a reason)
        # whenever it could not be computed; NEVER silently the in-sample one.
        self._ece_after_oos = None
        self._n_oos_samples = 0
        self._ece_oos_reason = None
        self._ece_oos_split = None
        self._class_weight = None  # set in fit() under method='platt'
        self._C = None             # set in fit() under method='platt'

    def fit(
        self,
        raw_alphas: np.ndarray,
        actual_up: np.ndarray,
        label_clip: float = 0.15,
        *,
        class_weight: str | dict | None = None,
        C: float = 1.0,
        dates=None,
        holdout_frac: float = 0.2,
        embargo_days: int | None = None,
    ) -> "PlattCalibrator":
        """
        Fit calibrator on walk-forward OOS predictions.

        Parameters
        ----------
        raw_alphas : array of raw continuous alpha predictions (clipped to [-label_clip, label_clip])
        actual_up  : binary array, 1 if forward_return_5d > 0 else 0
        label_clip : max absolute alpha used for clipping (default 0.15)
        class_weight : passed through to ``sklearn.linear_model.LogisticRegression``
            when ``method='platt'``. Use ``'balanced'`` so the intercept doesn't
            anchor on an imbalanced marginal UP-rate (the 2026-05-23 failure
            class: OOS rows 100% bear+neutral → Platt's intercept pushed
            negative → 95% DOWN-skew across the synthetic alpha sweep).
            Ignored under ``method='isotonic'`` (isotonic is per-quantile
            non-parametric — no global intercept to rebalance).
        C : inverse L2 regularisation strength for the Platt logistic. Higher
            C = less shrinkage. Ignored under isotonic. Default 1.0 preserves
            historical behavior.
        dates : optional per-row date labels, same length as ``raw_alphas``.
            When supplied, an OUT-OF-SAMPLE ECE is measured on a time-ordered,
            embargoed tail of DATES (alpha-engine-config-I10181). Without them
            no OOS number is produced: a random split would put the same day on
            both sides, and every row of one day shares the market's move.
        holdout_frac : fraction of DISTINCT DATES held out for the OOS ECE.
        embargo_days : distinct dates dropped between the train block and the
            held-out block. Pass ``cfg.WF_EMBARGO_DAYS`` so the split matches
            the walk-forward discipline every other fit-time number uses.

        The returned calibrator is always the one fitted on ALL rows. The
        split fit exists only to produce ``ece_after_oos`` and is discarded —
        serving on 80% of the data would be a capability regression smuggled in
        with a metric.
        """
        raw_alphas = np.asarray(raw_alphas, dtype=np.float64).ravel()
        actual_up = np.asarray(actual_up, dtype=np.int32).ravel()

        if len(raw_alphas) != len(actual_up):
            raise ValueError(f"Length mismatch: {len(raw_alphas)} alphas vs {len(actual_up)} labels")

        # Remove NaN/inf
        date_arr = None
        if dates is not None:
            date_arr = np.asarray(list(dates), dtype=object)
            if date_arr.shape[0] != raw_alphas.shape[0]:
                raise ValueError(
                    f"Length mismatch: {date_arr.shape[0]} dates vs "
                    f"{raw_alphas.shape[0]} alphas"
                )

        valid = np.isfinite(raw_alphas)
        raw_alphas = raw_alphas[valid]
        actual_up = actual_up[valid]
        if date_arr is not None:
            date_arr = date_arr[valid]

        if len(raw_alphas) < 100:
            log.warning("Calibrator: only %d valid samples (need 100+) — skipping fit", len(raw_alphas))
            return self

        self._n_samples = len(raw_alphas)
        self._class_weight = class_weight if self.method == "platt" else None
        self._C = C if self.method == "platt" else None

        # Compute baseline ECE (linear calibration)
        linear_p_up = np.clip(0.5 + raw_alphas / (2.0 * label_clip), 0.0, 1.0)
        self._ece_before = _expected_calibration_error(linear_p_up, actual_up)

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                C=C, class_weight=class_weight,
                solver="lbfgs", max_iter=1000,
            )
            self._model.fit(raw_alphas.reshape(-1, 1), actual_up)
        else:
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._model.fit(raw_alphas, actual_up)

        self._fitted = True

        # IN-SAMPLE ECE. Retained because it is a real property of the fit and
        # because two readers already consume it — but it is NOT evidence of
        # calibration under isotonic, where PAVA reproduces each block's pooled
        # empirical frequency and the number is ~0 by construction
        # (alpha-engine-config-I10181).
        calibrated_p_up = self.predict_proba(raw_alphas)
        self._ece_after = _expected_calibration_error(calibrated_p_up, actual_up)

        self._fit_oos_ece(
            raw_alphas, actual_up, date_arr,
            label_clip=label_clip, class_weight=class_weight, C=C,
            holdout_frac=holdout_frac, embargo_days=embargo_days,
        )

        log.info(
            "Calibrator fitted (%s): n=%d  ECE_before=%.4f  "
            "ECE_after_IN_SAMPLE=%.4f  ECE_after_OOS=%s (n_oos=%d%s)",
            self.method, self._n_samples, self._ece_before, self._ece_after,
            "n/a" if self._ece_after_oos is None
            else format(self._ece_after_oos, ".4f"),
            self._n_oos_samples,
            "" if self._ece_oos_reason is None else f"; {self._ece_oos_reason}",
        )
        if self._ece_after_oos is None:
            log.warning(
                "Calibrator (%s): no out-of-sample ECE was produced (%s). The "
                "in-sample number above is NOT a substitute — under isotonic it "
                "is ~0 by construction and cannot go bad "
                "(alpha-engine-config-I10181).",
                self.method, self._ece_oos_reason,
            )
        return self

    #: Distinct dates below which a held-out block is not worth splitting.
    MIN_OOS_SAMPLES = 100

    def _fit_oos_ece(
        self, raw_alphas, actual_up, date_arr, *, label_clip, class_weight, C,
        holdout_frac, embargo_days,
    ) -> None:
        """Measure ECE on a time-ordered, embargoed held-out block of DATES.

        alpha-engine-config-I10181. Splits on distinct dates rather than row
        index: every row of one date shares the market's move, so a row split
        puts the same day on both sides. Refits a SEPARATE calibrator of the
        same method/hyper-parameters on the train block and scores the held-out
        block with it — ``self`` is untouched and keeps its all-rows fit.

        Sets ``_ece_after_oos`` to None with a ``_ece_oos_reason`` whenever the
        split cannot be made. Never raises: a diagnostic must not take a
        training run down, and a null is reported, never a substituted number.
        """
        import numpy as np

        self._ece_after_oos = None
        self._n_oos_samples = 0
        self._ece_oos_split = None

        if date_arr is None:
            self._ece_oos_reason = (
                "no per-row date labels were supplied, so a time-ordered split "
                "could not be constructed; a random split would leak across "
                "dates and is not substituted"
            )
            return
        try:
            uniq = sorted({str(d) for d in date_arr.tolist()})
            embargo = int(embargo_days or 0)
            n_hold = max(1, int(round(len(uniq) * float(holdout_frac))))
            n_train = len(uniq) - n_hold - embargo
            if n_train < 1 or n_hold < 1:
                self._ece_oos_reason = (
                    f"only {len(uniq)} distinct date(s) after a "
                    f"{holdout_frac:.0%} holdout and a {embargo}-date embargo — "
                    f"not enough to split by date"
                )
                return
            train_dates = uniq[:n_train]
            oos_dates = uniq[n_train + embargo:]
            keys = np.asarray([str(d) for d in date_arr.tolist()], dtype=object)
            tr = np.isin(keys, list(train_dates))
            te = np.isin(keys, list(oos_dates))
            self._ece_oos_split = {
                "train_dates": list(train_dates),
                "oos_dates": list(oos_dates),
                "n_embargoed_dates": embargo,
                "n_train_samples": int(tr.sum()),
            }
            if int(te.sum()) < self.MIN_OOS_SAMPLES:
                self._ece_oos_reason = (
                    f"the held-out block has {int(te.sum())} rows, below the "
                    f"{self.MIN_OOS_SAMPLES}-sample floor a calibration error "
                    f"needs to mean anything"
                )
                return
            if int(tr.sum()) < self.MIN_OOS_SAMPLES:
                self._ece_oos_reason = (
                    f"the train block has {int(tr.sum())} rows, below the "
                    f"{self.MIN_OOS_SAMPLES}-sample floor the calibrator itself "
                    f"needs to fit"
                )
                return
            probe = PlattCalibrator(method=self.method)
            probe.fit(
                raw_alphas[tr], actual_up[tr], label_clip=label_clip,
                class_weight=class_weight, C=C,
            )
            if not probe.is_fitted:
                self._ece_oos_reason = (
                    "the train-block calibrator did not fit, so no held-out "
                    "error could be measured"
                )
                return
            self._ece_after_oos = _expected_calibration_error(
                probe.predict_proba(raw_alphas[te]), actual_up[te],
            )
            self._n_oos_samples = int(te.sum())
            self._ece_oos_reason = None
        except Exception as exc:  # noqa: BLE001 — reported as a null, never a pass
            self._ece_after_oos = None
            self._n_oos_samples = 0
            self._ece_oos_reason = f"out-of-sample ECE computation failed: {exc}"
            log.warning("Calibrator OOS ECE failed: %s", exc, exc_info=True)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict_proba(self, raw_alphas: np.ndarray) -> np.ndarray:
        """
        Map raw alpha predictions to calibrated P(UP).

        Returns array of probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted — call fit() first")

        raw_alphas = np.asarray(raw_alphas, dtype=np.float64).ravel()

        if self.method == "platt":
            return self._model.predict_proba(raw_alphas.reshape(-1, 1))[:, 1]
        else:
            return self._model.predict(raw_alphas)

    def calibrate_prediction(
        self,
        raw_alpha: float,
        label_clip: float = 0.15,
    ) -> dict:
        """
        Calibrate a single prediction. Returns p_up, p_down, direction, confidence.

        Falls back to linear calibration if not fitted.

        Direction semantics (config#1815, 2026-07-06): ``predicted_direction``
        comes from :func:`derive_direction` — the SIGN of the raw alpha, not
        ``p_up >= 0.5``. p_up keeps its honest base-rate-aware
        P(beat market) semantics for calibration/veto consumers; direction
        no longer argmaxes it (see derive_direction docstring for the
        incident this closes).

        Confidence semantics: ``|p_up - 0.5| * 2`` — distance from coin-flip on
        [0, 1]. At p_up=0.5 the model has no opinion → confidence=0.0; at
        p_up=1.0 or 0.0 the model is certain → confidence=1.0. Pre-2026-05-12
        this was ``max(p_up, p_down)`` (range [0.5, 1.0]), which treated a
        coin-flip prediction as "0.5 confident" — load-bearing on the
        DOWN-veto inversion at the 75%+ band (ROADMAP L1594).
        """
        alpha = float(np.clip(raw_alpha, -label_clip, label_clip))

        if self._fitted:
            p_up = float(self.predict_proba(np.array([alpha]))[0])
        else:
            # Linear fallback (existing behavior)
            p_up = float(np.clip(0.5 + alpha / (2.0 * label_clip), 0.0, 1.0))

        p_down = 1.0 - p_up
        direction = derive_direction(raw_alpha)
        confidence = abs(p_up - 0.5) * 2.0

        return {
            "p_up": round(p_up, 4),
            "p_down": round(p_down, 4),
            "p_flat": 0.0,
            "predicted_direction": direction,
            "prediction_confidence": round(confidence, 4),
        }

    def metrics(self) -> dict:
        """Return calibration metrics for reporting."""
        return {
            "method": self.method,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "ece_before": round(self._ece_before, 6) if self._ece_before is not None else None,
            # `ece_after` KEEPS its historical meaning (in-sample) and its key.
            # It is read by training/train_handler.py's training email and by a
            # sidecar reader in crucible-backtester; breaking it here would
            # break a cross-repo reader for a rename. `ece_after_in_sample` is
            # the name that cannot be misread, and `ece_after_oos` is the number
            # that can actually go bad (alpha-engine-config-I10181).
            "ece_after": round(self._ece_after, 6) if self._ece_after is not None else None,
            "ece_after_in_sample": (
                round(self._ece_after, 6) if self._ece_after is not None else None
            ),
            "ece_after_oos": (
                round(self._ece_after_oos, 6)
                if self._ece_after_oos is not None else None
            ),
            "n_oos_samples": int(self._n_oos_samples or 0),
            "ece_oos_reason": self._ece_oos_reason,
            "ece_oos_split": self._ece_oos_split,
            "class_weight": self._class_weight,
            "C": self._C,
        }

    def save(self, path: str | Path) -> None:
        """Save calibrator to pickle + metadata JSON.

        The sidecar includes ``deployed_at`` (ISO-8601 UTC). The backtester's
        retrain_alert uses this to apply a grace period after a fresh
        calibrator lands — during that window predictor_outcomes holds
        mixed pre/post calibrator semantics and ECE is structurally noisy.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._model, f)

        meta = self.metrics()
        meta["deployed_at"] = datetime.now(timezone.utc).isoformat()
        Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Calibrator saved to %s (deployed_at=%s)", path, meta["deployed_at"])

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        """Load a previously saved calibrator."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator not found: {path}")

        meta_path = Path(str(path) + ".meta.json")
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        method = meta.get("method", "platt")
        cal = cls(method=method)

        with open(path, "rb") as f:
            cal._model = pickle.load(f)

        cal._fitted = meta.get("fitted", True)
        cal._n_samples = meta.get("n_samples", 0)
        cal._ece_before = meta.get("ece_before")
        cal._ece_after = meta.get("ece_after_in_sample", meta.get("ece_after"))
        cal._ece_after_oos = meta.get("ece_after_oos")
        cal._n_oos_samples = meta.get("n_oos_samples", 0)
        cal._ece_oos_reason = meta.get(
            "ece_oos_reason",
            None if "ece_after_oos" in meta else (
                "this sidecar predates alpha-engine-config-I10181 and carries "
                "no out-of-sample ECE"
            ),
        )
        cal._ece_oos_split = meta.get("ece_oos_split")
        cal._class_weight = meta.get("class_weight")
        cal._C = meta.get("C")
        log.info("Calibrator loaded from %s (method=%s, n=%d)", path, method, cal._n_samples)
        return cal


def _expected_calibration_error(
    predicted_probs: np.ndarray,
    actual_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error of ``predicted_probs`` vs ``actual_labels``.

    Thin wrapper over the fleet-canonical implementation in
    ``nousergon_lib.quant.stats.calibration`` so the calibrator's fit-time
    ECE (``ece_before``/``ece_after``) and the backtester's production-time ECE
    are computed by the SAME code on the SAME quantity (calibrated probability
    vs binary outcome) — the only way the two numbers are comparable as a
    calibration-drift signal. ``predicted_probs`` must be a P(UP), not a margin.

    Returns the scalar ECE (0.0 when there is no usable bin), preserving the
    prior float contract for callers.
    """
    result = _lib_expected_calibration_error(
        predicted_probs, actual_labels, n_bins=n_bins,
    )
    ece = result.get("ece")
    return float(ece) if ece is not None else 0.0
