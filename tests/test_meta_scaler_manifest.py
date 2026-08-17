"""tests/test_meta_scaler_manifest.py — persist meta_scaler into the LIVE
manifest.json (alpha-engine-config-I7502).

Root cause: ``MetaModel.save()`` writes the fitted L4565 directional
standardize+winsorize scaler (mean/std/winsor bounds + column list) into
BOTH the ``meta_model.pkl`` payload (correctness-critical for inference —
see model/meta_model.py's ``_PICKLE_SCHEMA`` v3 note) and the
``meta_model.pkl.meta.json`` sidecar written alongside it locally. But the
sidecar's S3 upload is gated on ``if promoted:`` in ``run_meta_training``
(training/meta_trainer.py), and training has been permanently
challenger-first since L4469 (``gate_passed = promoted; promoted = False``,
unconditionally, every run) — so that upload branch is dead code and the
live sidecar has been frozen since whatever training run last executed it
pre-cutover (measured 2026-08-17: dated 2026-05-30 beside an 2026-08-15
meta_model.pkl). Any consumer that cannot unpickle the model directly
(e.g. crucible-backtester's
analysis/contribution_lift/groups/predictor_ensemble.py, which
reconstructs the L2 input space from JSON) had no fresh source for the
scaler once META_STANDARDIZE_ENABLED is switched on.

``manifest.json`` is written UNCONDITIONALLY every training run (the
``put_object`` sits outside the ``if promoted:`` gate — see the comment at
the ``meta_scaler`` key in the manifest literal) and is a REQUIRED_CONTRACT_FILE
for every registry snapshot/promotion (model/registry.py), so it is the
correct single, always-fresh, JSON-readable transport for this field.
Two invariants are pinned:

1. Source-text: the persisted ``manifest`` dict literal in
   ``run_meta_training`` carries a ``"meta_scaler":`` key (same pattern as
   ``tests/test_dead_l1_observe_manifest.py`` — a behavioral test would
   require spinning up the whole trainer with synthetic data, out of scope
   per ``tests/test_meta_trainer_oos_ic_field.py``).
2. Behavioral: ``MetaModel.metrics()["meta_scaler"]`` — the exact value
   ``run_meta_training`` reads into that manifest key (``meta_model._scaler``
   via ``.metrics()``) — is non-null whenever the model was fit with
   ``standardize_directional=True`` on data that yields a scaler, so the
   manifest can never silently carry ``None`` for a standardized model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.meta_model import MetaModel  # noqa: E402


_META_TRAINER = (
    Path(__file__).resolve().parent.parent / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def manifest_block() -> str:
    """The ``manifest = { ... }`` literal that gets json.dump'd to S3.

    Sliced from ``manifest = {`` to the ``manifest["ic_reliability"]``
    augmentation that immediately precedes the ``put_object`` write — the
    dict literal is fully closed by then, so any key in this slice is a
    persisted manifest field.
    """
    src = _META_TRAINER.read_text()
    start = src.index("manifest = {")
    end = src.index('manifest["ic_reliability"]', start)
    return src[start:end]


def test_meta_scaler_persisted_in_manifest(manifest_block):
    assert '"meta_scaler":' in manifest_block, (
        "meta_scaler is missing from the persisted manifest dict — the "
        "moment META_STANDARDIZE_ENABLED flips on, no JSON-only consumer "
        "(e.g. crucible-backtester's predictor_ensemble.py) can reconstruct "
        "the L2 input space, and the stale meta_model.pkl.meta.json sidecar "
        "is the only other source (alpha-engine-config-I7502 regression)."
    )


def test_meta_scaler_also_mirrored_in_run_result():
    """The run-result / training_summary dict mirrors the manifest field too,
    so config#2882-style SSOT drift between the two S3 records can't recur
    for this key specifically."""
    src = _META_TRAINER.read_text()
    # The run-result `return { ... }` literal sits after the manifest write;
    # slice from its own meta_coefficients occurrence (the second one in the
    # file — the manifest's is asserted separately above) to a distinctive
    # sibling key.
    idx = src.index('"meta_coefficients": meta_model._coefficients,')
    idx2 = src.index(
        '"meta_coefficients": meta_model._coefficients,', idx + 1
    )
    block = src[idx2:idx2 + 400]
    assert '"meta_scaler": meta_model._scaler,' in block


def _fit_standardized_model(n=200, seed=5):
    feats = [
        "research_calibrator_prob", "momentum_score", "research_composite_score",
        "macro_spy_20d_return",
    ]
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, len(feats)))
    y = 0.5 * X[:, 0] + rng.normal(0.0, 0.2, n)
    m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=True)
    return m


class TestMetaScalerAlwaysPopulatedWhenStandardized:
    def test_scaler_non_null_after_standardized_fit(self):
        m = _fit_standardized_model()
        assert m._scaler is not None
        assert m._scaler["directional"]  # at least one directional column

    def test_metrics_meta_scaler_matches_internal_state(self):
        """This is exactly the value run_meta_training reads into both S3
        records (manifest.json and the training_summary) — metrics()['meta_scaler']
        must be the same object/content as ._scaler, never re-derived or
        dropped in serialization."""
        m = _fit_standardized_model()
        metrics = m.metrics()
        assert metrics["meta_scaler"] is m._scaler
        assert metrics["meta_scaler"] is not None
        for key in ("directional", "mean", "std", "winsor"):
            assert key in metrics["meta_scaler"]

    def test_unstandardized_fit_yields_null_scaler(self):
        """Sanity check on the other branch: no standardization → manifest's
        meta_scaler is legitimately None, not a producer bug."""
        feats = ["research_calibrator_prob", "momentum_score"]
        rng = np.random.default_rng(9)
        X = rng.normal(0.0, 1.0, (200, len(feats)))
        y = 0.5 * X[:, 0] + rng.normal(0.0, 0.2, 200)
        m = MetaModel().fit(X, y, feature_names=feats, standardize_directional=False)
        assert m.metrics()["meta_scaler"] is None
