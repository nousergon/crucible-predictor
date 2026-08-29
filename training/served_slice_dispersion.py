"""training/served_slice_dispersion.py — dispersion over the batch the executor
would actually TRADE, measured for the candidate and the incumbent on the SAME
rows, at promotion time.

alpha-engine-config-I9061 (restores the champion-arch refresh path of -I9024 s1
behind an honest guard) · -I9024 s4 (the behavioral veto this feeds).

Why this module exists
----------------------
The behavioral veto (``training/promotion_behavioral_veto.py``) reads
``output_distribution_gate.metrics`` off the training manifest. Measured on the
two registry bundles of the 2026-08-21 rotation:

    v3.0-meta-2026-08-14-119e069b (incumbent)  stdev_p_up 0.113644
    v3.0-meta-2026-08-21-7d3d1cce (candidate)  stdev_p_up 0.130132

a ratio of 1.145 — a comfortable PASS against the 0.5 floor. The veto did not
bite, and the candidate promoted. It then collapsed the SERVED batch:

    2026-08-21 (incumbent serving)  stdev_p_up 0.191162  alpha_stdev 0.043427  n_high_confidence 5
    2026-08-24 (candidate serving)  stdev_p_up 0.060170  alpha_stdev 0.010437  n_high_confidence 0

The manifest number could not have seen it, and not because the training
distribution is merely wider than the served one. ``stdev_p_up`` on the manifest
comes from ``model.output_distribution_gate.validate_calibrator_distribution``,
which sweeps 25 evenly-spaced SYNTHETIC alphas through the calibrator alone. It
is a shape check on the calibrator, and it never touches the meta-model, the
universe, or the selection rule. It is two transforms removed from the quantity
the executor consumes.

What the executor consumes is the spread ACROSS THE ~30 NAMES IT RANKS. That is
the quantity measured here, and champion-challenger-policy §7.3 is explicit that
the gate belongs on the invariant the actor consumes rather than on a transform
of it.

How it is measured
------------------
Both models are scored, at promotion time, over ONE common real feature panel —
this rotation's own OOS row dump, ``predictor/diagnostics/oos_rows/{date}.parquet``
(written by the live champion-arch spec; shadow specs write elsewhere, see
``training/io_spec.py``). Per panel date, the top ``top_n`` rows by predicted
alpha are taken — the executor's own selection rule — and the dispersion of
``predicted_alpha`` and ``p_up`` WITHIN that slice is measured, plus the count of
names clearing ``MIN_CONFIDENCE``. The per-date values are reduced by MEDIAN.

The scoring head is each version's OWN ``meta_model.pkl`` + ``isotonic_calibrator.pkl``,
read from its immutable registry bundle. That pair is exactly what would serve:
``model.registry.promote_to_champion`` copies the whole bundle into
``predictor/weights/meta/``, which is both ``cfg.META_WEIGHTS_PREFIX`` and the
prefix of ``cfg.CALIBRATOR_WEIGHTS_KEY``.

Why not a shadow inference run
------------------------------
``inference/stages/shadow_versions.py`` re-scores a challenger bundle over the
live batch, and is the more faithful measurement in principle. It is rejected
here for three reasons, all measured: (1) it is only cheap because it REUSES a
live inference context — universe, prices, alt-data, macro — which the Saturday
training box does not have and would have to build; (2) it loads the meta-model
from the challenger bundle but the CALIBRATOR from the live prefix
(``inference/stages/load_model.py:206`` reads ``cfg.CALIBRATOR_WEIGHTS_KEY``, not
``ctx.weights_prefix_override``), so its ``p_up`` is the challenger's alpha
through the CHAMPION's calibrator — not the pair that would serve; and (3) it
measures one day, where the panel here gives 167.

Verified against the real artifacts (top_n=30, min_confidence=0.30, the
2026-08-21 panel, 167 dates):

    incumbent v3.0-meta-2026-08-14-119e069b   alpha_stdev 0.016624  stdev_p_up 0.071064  n_high_confidence 30
    candidate v3.0-meta-2026-08-21-7d3d1cce   alpha_stdev 0.005429  stdev_p_up 0.041531  n_high_confidence  0

    alpha_stdev ratio 0.327 → VETO (floor 0.5) · n_high_confidence 0 → VETO

Two independent refusals of the candidate that actually promoted, and the
incumbent's n_high_confidence of 30/30 against the candidate's 0/30 reproduces
the live 5-vs-0 split of 2026-08-21 vs 2026-08-24 in sign and magnitude.

Failure posture
---------------
Every failure path returns ``status: "uncomputable"`` with a named reason and is
NON-BLOCKING (champion-challenger-policy §5.1 — you cannot gate on a statistic
you did not measure), and it is never rendered as a pass: the caller surfaces the
reason on the leaderboard. Nothing here is swallowed silently.
"""
from __future__ import annotations

import io as _io
import logging
import tempfile
from pathlib import Path

import config as cfg

log = logging.getLogger(__name__)

_REGISTRY_PREFIX = "predictor/registry/"
_OOS_ROWS_PREFIX = "predictor/diagnostics/oos_rows/"

# The width of the batch the executor trades. The live batch is ~30 names.
DEFAULT_TOP_N = 30
# ``model.calibrator.PlattCalibrator.calibrate_prediction``'s own default — the
# clip applied to raw alpha before calibration on the serving path.
LABEL_CLIP = 0.15
# A date contributing fewer rows than this cannot represent a ~30-name slice.
MIN_SLICE_ROWS = 5
# Fewer measured dates than this is not a distribution — reported uncomputable.
MIN_DATES = 20
# The panel's realized forward-return column, in preference order. Used ONLY by
# the observe-only diagnostics below (alpha-engine-config-I9257); its absence
# never affects a verdict.
REALIZED_FWD_COLUMNS: tuple[str, ...] = ("actual_fwd_canonical", "actual_fwd")


def _top_n() -> int:
    return int(getattr(cfg, "MODEL_ZOO_SERVED_SLICE_TOP_N", DEFAULT_TOP_N))


def _min_confidence() -> float:
    return float(getattr(cfg, "MIN_CONFIDENCE", 0.30))


def _get_bytes(s3, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_oos_panel(s3, bucket: str, date_str: str | None) -> tuple[object, str]:
    """The rotation's own OOS feature panel, plus the key it came from.

    Prefers the dated key so the panel is provably THIS rotation's vintage;
    falls back to ``latest.parquet`` (logged) when the dated one is absent.
    Raises when neither is readable — the caller turns that into
    ``uncomputable``, never into a pass.
    """
    import pandas as pd

    keys = []
    if date_str:
        keys.append(f"{_OOS_ROWS_PREFIX}{date_str}.parquet")
    keys.append(f"{_OOS_ROWS_PREFIX}latest.parquet")
    last_exc: Exception | None = None
    for key in keys:
        try:
            panel = pd.read_parquet(_io.BytesIO(_get_bytes(s3, bucket, key)))
        except Exception as exc:  # noqa: BLE001 — tried in order, reported below
            last_exc = exc
            continue
        if key.endswith("latest.parquet") and date_str:
            log.warning(
                "served-slice dispersion: dated OOS panel %s%s.parquet is not "
                "readable — falling back to latest.parquet. The panel's vintage "
                "is therefore not proven to be this rotation's.",
                _OOS_ROWS_PREFIX, date_str,
            )
        return panel, key
    raise RuntimeError(
        f"no readable OOS feature panel at any of {keys} ({last_exc})"
    )


def load_meta_model(s3, bucket: str, version_id: str):
    """The Layer-2 ``MetaModel`` from a version's immutable registry bundle.

    Shared with ``training/incumbent_rescore.py`` so the two promotion-time
    reads of a frozen model cannot drift apart.
    """
    from model.meta_model import MetaModel

    src = f"{_REGISTRY_PREFIX}{version_id}/"
    with tempfile.TemporaryDirectory() as tmp:
        mm_path = Path(tmp) / "meta_model.pkl"
        mm_path.write_bytes(_get_bytes(s3, bucket, f"{src}meta_model.pkl"))
        return MetaModel.load(mm_path)


def load_scoring_head(s3, bucket: str, version_id: str) -> tuple[object, object]:
    """``(MetaModel, calibrator_estimator)`` from a version's registry bundle.

    Exactly the pair ``promote_to_champion`` would copy into the live prefix, so
    what is measured is what would serve.
    """
    import pickle

    meta_model = load_meta_model(s3, bucket, version_id)
    calibrator = pickle.loads(
        _get_bytes(s3, bucket, f"{_REGISTRY_PREFIX}{version_id}/isotonic_calibrator.pkl")
    )
    return meta_model, calibrator


def _calibrate(calibrator, alphas):
    """``p_up`` for a vector of raw alphas, matching the serving path's shape.

    Mirrors ``model.calibrator.PlattCalibrator``: isotonic estimators expose
    ``predict``, Platt ones ``predict_proba``. The clip and the 4dp rounding are
    the serving path's own (``calibrate_prediction``), so ``n_high_confidence``
    is counted on the same quantization the executor sees.
    """
    import numpy as np

    clipped = np.clip(np.asarray(alphas, dtype=float), -LABEL_CLIP, LABEL_CLIP)
    if hasattr(calibrator, "predict_proba"):
        p_up = calibrator.predict_proba(clipped.reshape(-1, 1))[:, 1]
    else:
        p_up = calibrator.predict(clipped)
    return np.round(np.asarray(p_up, dtype=float), 4)


def score_panel(meta_model, calibrator, panel) -> tuple[object, object]:
    """``(predicted_alpha, p_up)`` for every panel row, under one version's head.

    Raises when the panel does not carry every feature the stored ridge was fit
    on. Zero-filling a missing META_FEATURE is the ``alpha-engine-config-I5949``
    defect — a model silently scored on zeros — so it is refused here too.
    """
    import numpy as np

    names = list(getattr(meta_model, "_feature_names", None) or [])
    if not names:
        raise RuntimeError("meta-model carries no feature_names — cannot score")
    missing = [f for f in names if f not in panel.columns]
    if missing:
        raise RuntimeError(
            f"OOS panel is missing {len(missing)} of the model's features "
            f"{missing[:6]}{'…' if len(missing) > 6 else ''} — refusing to "
            f"zero-fill (alpha-engine-config-I5949)"
        )
    alpha = np.asarray(
        meta_model.predict(panel[names].to_numpy(dtype=float)), dtype=float
    )
    return alpha, _calibrate(calibrator, alpha)


def _t_stat(values) -> float | None:
    """Mean over standard error, or ``None`` when it is not defined."""
    import numpy as np

    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size < 2:
        return None
    se = float(np.std(arr, ddof=1)) / float(np.sqrt(arr.size))
    if not se > 0:
        return None
    return round(float(np.mean(arr)) / se, 4)


def _spearman(a, b) -> float | None:
    """Rank correlation without a SciPy dependency on this path.

    Pearson on ranks, which is Spearman by definition. Returns ``None`` rather
    than NaN when either side is constant — a degenerate correlation is not a
    measurement and must never render as zero.
    """
    import numpy as np
    import pandas as pd

    x = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    y = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def selected_slice_metrics(alpha, p_up, dates, *, top_n: int,
                           min_confidence: float,
                           min_dates: int = MIN_DATES,
                           realized=None) -> dict:
    """Dispersion WITHIN the top-``top_n``-by-alpha slice, per date, median-reduced.

    ``n_high_confidence`` counts names whose ``|p_up - 0.5| * 2`` clears
    ``min_confidence`` — the identical definition
    ``inference/stages/write_output.py`` uses on the live batch.

    The verdict-bearing keys are ``alpha_stdev``, ``stdev_p_up`` and
    ``n_high_confidence`` and nothing here changes them.

    OBSERVE-ONLY (alpha-engine-config-I9257)
    ----------------------------------------
    Two nested blocks are added, and neither can ever arm a rule:
    ``promotion_behavioral_veto._SERVED_METRIC_NAMES`` filters the merge to the
    four flat rule metrics by name, so a nested key is structurally unreachable
    from the veto. They exist because the three verdict metrics above are all
    RAW-MAGNITUDE quantities, and a uniform rescaling of the linear predictor
    collapses every one of them with no loss of ranking information — which is
    exactly what happened to the 2026-08-28 vintage, where the champion-
    ARCHITECTURE control was vetoed alongside every challenger.

    ``scale_invariant.alpha_stdev_standardized``
        The same top-N slice dispersion, but measured on alpha z-scored
        CROSS-SECTIONALLY WITHIN EACH DATE before slicing. Invariant to any
        per-date affine rescaling, so a gap that survives here is a shape
        change and a gap that vanishes here was only scale.

    ``realized.rank_ic`` / ``realized.top_n_lift``
        Per-date Spearman of alpha against the panel's realized forward return,
        and the top-N slice's realized return net of that date's universe mean.
        Measured over the SAME rows as the dispersion, so a candidate whose
        magnitude collapsed can be told apart from one whose ranking did.

    Measured on the real artifacts, why both blocks are needed and why NEITHER
    replaces the raw metric (panel ``oos_rows/2026-08-28.parquet`` and
    ``oos_rows/2026-08-21.parquet``, 167 dates each, top_n=30):

        08-28 champion-arch vs 08-14 incumbent
            alpha_stdev ratio            0.347  → VETO
            standardized ratio           0.943
            top-30 realized lift        +0.0323 (t +7.25) vs -0.0037 (t -0.70)

        08-21 candidate (the one that promoted and collapsed the live batch)
            alpha_stdev ratio            0.327  → VETO
            standardized ratio           0.973  ← a scale-invariant rule PASSES it
            top-30 realized lift        -0.0353 (t -7.44)

    The standardized quantity alone would have admitted the 2026-08-21 failure.
    That is precisely why this is a diagnostic and NOT a replacement rule, and
    why no threshold moves in the change that introduced it.
    """
    import numpy as np
    import pandas as pd

    columns = {"date": np.asarray(dates), "alpha": alpha, "p_up": p_up}
    has_realized = realized is not None
    if has_realized:
        columns["realized"] = np.asarray(realized, dtype=float)
    frame = pd.DataFrame(columns)
    frame = frame[np.isfinite(frame["alpha"]) & np.isfinite(frame["p_up"])]
    per_date_alpha: list[float] = []
    per_date_p_up: list[float] = []
    per_date_n_hi: list[int] = []
    per_date_z: list[float] = []
    per_date_ic: list[float] = []
    per_date_lift: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < MIN_SLICE_ROWS:
            continue
        raw = group["alpha"].to_numpy()
        sliced = group.nlargest(min(top_n, len(group)), "alpha")
        per_date_alpha.append(float(np.std(sliced["alpha"].to_numpy())))
        per_date_p_up.append(float(np.std(sliced["p_up"].to_numpy())))
        confidence = np.round(np.abs(sliced["p_up"].to_numpy() - 0.5) * 2.0, 4)
        per_date_n_hi.append(int(np.sum(confidence >= min_confidence)))

        # ── observe-only ────────────────────────────────────────────────────
        spread = float(np.std(raw))
        if spread > 0:
            standardized = (raw - float(np.mean(raw))) / spread
            order = np.argsort(-raw, kind="stable")[:min(top_n, len(raw))]
            per_date_z.append(float(np.std(standardized[order])))
        if has_realized:
            realized_v = group["realized"].to_numpy()
            finite = np.isfinite(realized_v)
            if finite.sum() >= MIN_SLICE_ROWS:
                ic = _spearman(raw[finite], realized_v[finite])
                if ic is not None:
                    per_date_ic.append(ic)
                sliced_realized = sliced["realized"].to_numpy()
                sliced_finite = np.isfinite(sliced_realized)
                if sliced_finite.any():
                    per_date_lift.append(
                        float(np.mean(sliced_realized[sliced_finite]))
                        - float(np.mean(realized_v[finite]))
                    )
    if len(per_date_alpha) < min_dates:
        raise RuntimeError(
            f"only {len(per_date_alpha)} panel dates yielded a slice of at least "
            f"{MIN_SLICE_ROWS} rows (need {min_dates}) — not a distribution"
        )
    scale_invariant: dict = {
        "alpha_stdev_standardized": (
            round(float(np.median(per_date_z)), 6) if per_date_z else None
        ),
        "n_dates": len(per_date_z),
        "reason": None if per_date_z else (
            "no panel date had a non-zero cross-sectional alpha spread"
        ),
    }
    if has_realized:
        realized_block: dict = {
            "column": None,  # filled by the caller, which knows which one it read
            "rank_ic": (
                round(float(np.mean(per_date_ic)), 6) if per_date_ic else None
            ),
            "rank_ic_t": _t_stat(per_date_ic),
            "top_n_lift": (
                round(float(np.mean(per_date_lift)), 6) if per_date_lift else None
            ),
            "top_n_lift_t": _t_stat(per_date_lift),
            "n_dates": len(per_date_ic),
            "reason": None if per_date_ic else (
                "the panel carries a realized column but no date yielded a "
                "measurable rank correlation"
            ),
        }
    else:
        realized_block = {
            "column": None, "rank_ic": None, "rank_ic_t": None,
            "top_n_lift": None, "top_n_lift_t": None, "n_dates": 0,
            "reason": (
                f"panel carries none of the realized forward-return columns "
                f"{list(REALIZED_FWD_COLUMNS)} — the realized diagnostics are "
                f"UNMEASURED, not zero (champion-challenger-policy §7.2)"
            ),
        }
    return {
        "alpha_stdev": round(float(np.median(per_date_alpha)), 8),
        "stdev_p_up": round(float(np.median(per_date_p_up)), 6),
        "n_high_confidence": int(np.median(per_date_n_hi)),
        "n_dates": len(per_date_alpha),
        "top_n": top_n,
        "min_confidence": min_confidence,
        # Nested, and therefore unreachable from the veto's flat metric merge.
        "scale_invariant": scale_invariant,
        "realized": realized_block,
    }


def _cross_version_diagnostics(alphas: dict, dates, reference_version_id, *,
                               top_n: int) -> dict:
    """Per-version agreement with the REFERENCE version, per date.

    Observe-only (alpha-engine-config-I9257). Two quantities, both invariant to
    any per-date rescaling of either side:

    ``top_n_overlap_vs_reference``
        ``|A ∩ B| / max(|A|, |B|)`` over the two top-N selections. Divided by
        the WIDER set, never the intersection — champion-challenger-policy §4,
        so a narrow set contained in a wide one scores as the different
        (narrower) selection it is rather than as a perfect clone.

    ``rank_corr_vs_reference``
        Mean per-date Spearman between the two alpha vectors on the same rows.

    Together they answer the question the raw dispersion ratio cannot: did the
    candidate pick DIFFERENT names, or the same names at a different scale?
    """
    import numpy as np
    import pandas as pd

    out: dict = {}
    if not reference_version_id or reference_version_id not in alphas:
        return out
    ref_alpha = alphas[reference_version_id]
    date_arr = np.asarray(dates)
    groups = list(pd.Series(range(len(date_arr))).groupby(pd.Series(date_arr)))
    for vid, cand_alpha in alphas.items():
        if vid == reference_version_id:
            continue
        overlaps: list[float] = []
        corrs: list[float] = []
        for _, idx in groups:
            rows = idx.to_numpy()
            if rows.size < MIN_SLICE_ROWS:
                continue
            c, r = cand_alpha[rows], ref_alpha[rows]
            finite = np.isfinite(c) & np.isfinite(r)
            if finite.sum() < MIN_SLICE_ROWS:
                continue
            c, r, rows = c[finite], r[finite], rows[finite]
            k = min(top_n, rows.size)
            cand_top = set(rows[np.argsort(-c, kind="stable")[:k]].tolist())
            ref_top = set(rows[np.argsort(-r, kind="stable")[:k]].tolist())
            overlaps.append(
                len(cand_top & ref_top) / max(len(cand_top), len(ref_top))
            )
            corr = _spearman(c, r)
            if corr is not None:
                corrs.append(corr)
        out[vid] = {
            "reference_version_id": reference_version_id,
            "top_n_overlap_vs_reference": (
                round(float(np.mean(overlaps)), 6) if overlaps else None
            ),
            "rank_corr_vs_reference": (
                round(float(np.mean(corrs)), 6) if corrs else None
            ),
            "n_dates": len(overlaps),
            "reason": None if overlaps else (
                "no panel date had enough finite rows under both versions"
            ),
        }
    return out


def served_slice_metrics(s3, bucket: str, version_ids, *,
                         date_str: str | None = None,
                         panel=None, panel_key: str | None = None,
                         reference_version_id: str | None = None) -> dict:
    """Served-slice metrics for every version in ``version_ids``, on ONE panel.

    Returns::

        {"status": "measured" | "uncomputable",
         "reason": str | None,
         "panel_key": str | None, "n_panel_rows": int | None,
         "top_n": int, "min_confidence": float,
         "metrics": {version_id: {...}},
         "errors":  {version_id: reason}}

    ``uncomputable`` is NON-BLOCKING and carries its reason
    (champion-challenger-policy §5.1). A per-version failure is recorded in
    ``errors`` and named on the leaderboard; it is never a pass, and the veto
    simply cannot arm for that version.
    """
    top_n, min_conf = _top_n(), _min_confidence()
    out: dict = {
        "status": "uncomputable", "reason": None, "panel_key": panel_key,
        "n_panel_rows": None, "top_n": top_n, "min_confidence": min_conf,
        "metrics": {}, "errors": {},
    }
    version_ids = [v for v in dict.fromkeys(version_ids) if v]
    if not version_ids:
        out["reason"] = "no versions to measure"
        return out
    if panel is None:
        try:
            panel, panel_key = read_oos_panel(s3, bucket, date_str)
        except Exception as exc:  # noqa: BLE001 — reported, never a pass
            out["reason"] = f"OOS feature panel unavailable: {exc}"
            log.warning(
                "served-slice dispersion is UNCOMPUTABLE for this rotation: %s. "
                "The behavioral veto falls back to whatever the manifests carry "
                "and is reported insufficient where they carry nothing — it is "
                "NOT a pass (champion-challenger-policy §5.1).", out["reason"],
            )
            return out
    out["panel_key"] = panel_key
    if "date" not in getattr(panel, "columns", []):
        out["reason"] = f"OOS panel {panel_key} carries no 'date' column"
        log.warning("served-slice dispersion uncomputable: %s", out["reason"])
        return out
    out["n_panel_rows"] = int(len(panel))
    dates = panel["date"].to_numpy()
    # alpha-engine-config-I9257 — the realized forward-return column feeding the
    # OBSERVE-ONLY diagnostics. Absent is a named reason on every version's
    # block, never a silent None and never a verdict change.
    realized_column = next(
        (c for c in REALIZED_FWD_COLUMNS if c in getattr(panel, "columns", [])),
        None,
    )
    out["realized_column"] = realized_column
    if realized_column is None:
        log.warning(
            "served-slice dispersion: panel %s carries none of %s, so the "
            "observe-only realized diagnostics (alpha-engine-config-I9257) are "
            "UNMEASURED for this rotation. The VERDICT metrics are unaffected.",
            panel_key, list(REALIZED_FWD_COLUMNS),
        )
    realized = (
        panel[realized_column].to_numpy(dtype=float)
        if realized_column is not None else None
    )
    alphas: dict = {}
    for vid in version_ids:
        try:
            meta_model, calibrator = load_scoring_head(s3, bucket, vid)
            alpha, p_up = score_panel(meta_model, calibrator, panel)
            metrics = selected_slice_metrics(
                alpha, p_up, dates, top_n=top_n, min_confidence=min_conf,
                realized=realized)
            metrics["realized"]["column"] = realized_column
            out["metrics"][vid] = metrics
            alphas[vid] = alpha
        except Exception as exc:  # noqa: BLE001 — per-version, named, non-blocking
            out["errors"][vid] = str(exc)
            log.warning(
                "served-slice dispersion could not be measured for %s: %s — "
                "recorded on the leaderboard as an error, NOT as a pass.",
                vid, exc, exc_info=True,
            )
    # Observe-only cross-version agreement against the SERVING version.
    for vid, block in _cross_version_diagnostics(
            alphas, dates, reference_version_id, top_n=top_n).items():
        if vid in out["metrics"]:
            out["metrics"][vid]["scale_invariant"].update(block)
    # The standardized dispersion RATIO against the reference, so a reader does
    # not have to divide two medians by hand to tell scale from shape.
    ref_metrics = out["metrics"].get(reference_version_id or "") or {}
    ref_std = (ref_metrics.get("scale_invariant") or {}).get("alpha_stdev_standardized")
    for vid, metrics in out["metrics"].items():
        if vid == reference_version_id:
            continue
        cand_std = (metrics.get("scale_invariant") or {}).get("alpha_stdev_standardized")
        metrics["scale_invariant"]["alpha_stdev_standardized_ratio"] = (
            round(cand_std / ref_std, 6)
            if cand_std is not None and ref_std not in (None, 0) else None
        )
    out["reference_version_id"] = reference_version_id
    if out["metrics"]:
        out["status"] = "measured"
        log.info(
            "served-slice dispersion measured over %s (%s rows, top_n=%s, "
            "min_confidence=%s): %s",
            panel_key, out["n_panel_rows"], top_n, min_conf, out["metrics"],
        )
    else:
        out["reason"] = (
            f"no version could be scored on {panel_key} "
            f"({'; '.join(f'{k}: {v}' for k, v in out['errors'].items())})"
        )
    return out
