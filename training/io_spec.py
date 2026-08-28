"""training/io_spec.py — Training data-source + output-path specification.

A single immutable bundle that pins, for one training run, WHICH ArcticDB
universe library to read, WHICH price column the label is computed from, and
WHERE every produced artifact (weights, manifest, feature-list, training
summary, OOS diagnostics) is written — plus whether the run may touch the
live champion / model-zoo registry at all.

Two factories:

  * :meth:`TrainingIOSpec.live` — the production default. Reads the canonical
    ``universe`` library, labels off ``Close``, writes its weight contract to
    the TRAINING STAGING root ``predictor/weights/meta_staging/`` (scoped per
    run by :meth:`for_run`) and its metrics to ``predictor/metrics/``, and is
    eligible for champion promotion + model-zoo registration.

    alpha-engine-config-I9018: ``live()`` deliberately does NOT write to
    ``predictor/weights/meta/``. That prefix is the LIVE SERVING contract and
    has exactly one writer, ``model.registry.promote_to_champion``. Until
    2026-08-28 ``live()`` pointed straight at it, so the Saturday retrain
    replaced the served model's manifest and feature contract as a side effect
    of training — no promotion decision required, and no way to roll back that
    survived the next rotation.

  * :meth:`TrainingIOSpec.shadow` — an evidence-only run on an alternate basis
    (e.g. ``crsp`` total-return). Reads the SCRATCH ``universe_crsp`` library,
    labels off ``total_return_close``, writes every artifact under DISTINCT
    ``*_shadow/{basis}/`` prefixes, and is HARD-BLOCKED from promoting the
    champion or entering the model-zoo selection pool. This lets the system
    retrain on the total-return basis and compare its IC / subsample-IC to the
    live champion WITHOUT any risk of swapping in unblessed weights.

The live champion path (``predictor/weights/meta/…``) is never reachable from
a shadow spec — the prefixes are disjoint and ``allow_live_promote`` /
``register_in_zoo`` are both False — so a shadow run cannot, by construction,
alter what production serves.

PR7-step-7b of epic config#1433 / config#1434. Cross-repo dependency: the
``universe_crsp`` scratch library is built by ne-data (nousergon-data #554).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Supported shadow bases → (universe library, label close column). The only
# basis today is the CRSP total-return rebuild; adding a basis is a one-line
# entry here plus the matching scratch library on the ne-data side.
_SHADOW_BASES: dict[str, tuple[str, str]] = {
    "crsp": ("universe_crsp", "total_return_close"),
}


@dataclass(frozen=True)
class TrainingIOSpec:
    """Immutable data-source + output-path + promotion-eligibility bundle.

    Field defaults describe the LIVE production run; :meth:`shadow` overrides
    them for an evidence-only alternate-basis run. ``run_meta_training`` and
    ``train_handler.main`` accept an optional instance and fall back to
    :meth:`live` when ``None``, so existing callers are unaffected.
    """

    # ── Inputs ────────────────────────────────────────────────────────────
    universe_lib: str = "universe"
    close_col: str = "Close"

    # ── Output paths ──────────────────────────────────────────────────────
    # alpha-engine-config-I9018 — the defaults are the TRAINING STAGING root,
    # never the live serving prefix. Training must not be able to reach
    # ``predictor/weights/meta/`` even by forgetting to pass a spec.
    weights_prefix: str = "predictor/weights/meta_staging/"
    manifest_key: str = "predictor/weights/meta_staging/manifest.json"
    feature_list_key: str = "predictor/weights/meta_staging/feature_list.json"
    summary_key_tmpl: str = "predictor/metrics/training_summary_{date}.json"
    summary_latest_key: str = "predictor/metrics/training_summary_latest.json"
    oos_rows_prefix: str = "predictor/diagnostics/oos_rows/"
    # Destination a downstream SHADOW inference run writes predictions to.
    # Not consumed by training itself; carried here so the operational 7c
    # shadow-inference run reads one authoritative path bundle.
    predictions_prefix: str = "predictor/predictions/"

    # ── Promotion eligibility ─────────────────────────────────────────────
    # When False the run can never overwrite the live champion weights AND is
    # never registered as a model-zoo challenger (so select_winner can never
    # promote it). When False we also skip the live-path side artifacts
    # (factor-risk-model, triple-barrier cutover gate) that would otherwise
    # write to shared production keys.
    allow_live_promote: bool = True
    register_in_zoo: bool = True
    write_side_artifacts: bool = True

    # Non-None marks this as a shadow run (e.g. "crsp").
    shadow_basis: str | None = None

    # alpha-engine-config-I9018 — when True, :meth:`for_run` scopes the output
    # paths by ``{date}/{model_version}/`` so two specs in the same weekly Map
    # iteration cannot overwrite each other. False for a shadow run, whose
    # prefixes are already disjoint from everything else by construction.
    run_scoped_outputs: bool = True

    @property
    def is_shadow(self) -> bool:
        return self.shadow_basis is not None

    def summary_key(self, date_str: str) -> str:
        return self.summary_key_tmpl.format(date=date_str)

    def oos_rows_key(self, date_str: str) -> str:
        return f"{self.oos_rows_prefix}{date_str}.parquet"

    @property
    def oos_rows_latest_key(self) -> str:
        return f"{self.oos_rows_prefix}latest.parquet"

    def for_run(self, *, date_str: str, model_version: str) -> "TrainingIOSpec":
        """Return this spec with its WEIGHT-CONTRACT paths scoped to one run.

        alpha-engine-config-I9018. The weekly rotation trains several specs, in
        parallel Step Functions Map iterations, against one bucket. Sharing a
        single output prefix between them is what let the dated archive capture
        "whichever spec wrote last" instead of the model it was labelled with
        (I9028), and it is why a per-run prefix — not merely a non-serving one —
        is the fix.

        Scoped by BOTH the rotation date and the spec's ``MODEL_VERSION_LABEL``,
        so the path is deterministic and an operator can find a given run's
        output without a listing. Returns ``self`` unchanged for a shadow spec
        (already disjoint) or when scoping is switched off.

        Summary / OOS-diagnostic paths are deliberately NOT scoped: they are
        dated artifacts with their own freshness SLAs and their own consumers.
        """
        if not self.run_scoped_outputs:
            return self
        if not date_str or not model_version:
            raise ValueError(
                "TrainingIOSpec.for_run requires a non-empty date_str and "
                "model_version — an unscoped training output prefix is the "
                "defect this exists to prevent (alpha-engine-config-I9018)."
            )
        root = f"{self.weights_prefix}{date_str}/{model_version}/"
        return replace(
            self,
            weights_prefix=root,
            manifest_key=f"{root}manifest.json",
            feature_list_key=f"{root}feature_list.json",
        )

    @classmethod
    def live(cls) -> "TrainingIOSpec":
        """The production default — reads ``config`` for the live keys so any
        future key rename in ``config.py`` is honoured. Byte-identical to the
        pre-shadow behaviour."""
        import config as cfg

        return cls(
            universe_lib="universe",
            close_col="Close",
            # alpha-engine-config-I9018 — the STAGING root, NOT
            # cfg.META_WEIGHTS_PREFIX. ``predictor/weights/meta/`` is the live
            # serving contract and is written by exactly one caller,
            # ``model.registry.promote_to_champion``. A training run that could
            # write there swaps the served model as a side effect of training,
            # which is what happened every Saturday until 2026-08-28.
            weights_prefix=cfg.META_STAGING_PREFIX,
            manifest_key=f"{cfg.META_STAGING_PREFIX}manifest.json",
            feature_list_key=f"{cfg.META_STAGING_PREFIX}feature_list.json",
            summary_key_tmpl="predictor/metrics/training_summary_{date}.json",
            summary_latest_key="predictor/metrics/training_summary_latest.json",
            oos_rows_prefix="predictor/diagnostics/oos_rows/",
            predictions_prefix="predictor/predictions/",
            allow_live_promote=True,
            register_in_zoo=True,
            write_side_artifacts=True,
            shadow_basis=None,
            run_scoped_outputs=True,
        )

    @classmethod
    def shadow(cls, basis: str) -> "TrainingIOSpec":
        """An evidence-only run on ``basis`` (e.g. ``crsp``). Reads the scratch
        library, labels off the alternate close column, isolates every output
        under ``*_shadow/{basis}/``, and is hard-blocked from promotion +
        model-zoo registration.

        Raises ``ValueError`` (fail loud) for an unknown basis — a typo must
        never silently degrade to the live basis."""
        key = (basis or "").strip().lower()
        if key not in _SHADOW_BASES:
            raise ValueError(
                f"unknown shadow basis {basis!r} — supported: "
                f"{sorted(_SHADOW_BASES)}"
            )
        universe_lib, close_col = _SHADOW_BASES[key]
        wroot = f"predictor/weights_shadow/{key}/"
        mroot = f"predictor/metrics_shadow/{key}/"
        return cls(
            universe_lib=universe_lib,
            close_col=close_col,
            weights_prefix=wroot,
            manifest_key=f"{wroot}manifest.json",
            feature_list_key=f"{wroot}feature_list.json",
            summary_key_tmpl=mroot + "training_summary_{date}.json",
            summary_latest_key=f"{mroot}training_summary_latest.json",
            oos_rows_prefix=f"predictor/diagnostics_shadow/{key}/oos_rows/",
            predictions_prefix=f"predictor/predictions_shadow/{key}/",
            allow_live_promote=False,
            register_in_zoo=False,
            write_side_artifacts=False,
            shadow_basis=key,
            run_scoped_outputs=False,
        )

    @classmethod
    def resolve(
        cls, shadow_basis: str | None, env: dict | None = None
    ) -> "TrainingIOSpec":
        """Resolve the spec from an explicit ``shadow_basis`` arg, falling back
        to the ``CRSP_SHADOW_ENABLED`` / ``SHADOW_BASIS`` env pair so the spot
        heredoc can flip shadow mode with an env export (mirrors the
        ``PREDICTOR_DEFER_TRAINING_EMAIL`` env-fallback pattern). An explicit
        arg always wins. Returns :meth:`live` when neither selects shadow."""
        import os

        e = os.environ if env is None else env
        basis = shadow_basis
        if basis is None:
            flag = (e.get("CRSP_SHADOW_ENABLED") or "").strip().lower()
            if flag not in ("", "0", "false", "no", "off"):
                basis = (e.get("SHADOW_BASIS") or "crsp").strip() or "crsp"
        if basis is None:
            return cls.live()
        return cls.shadow(basis)
