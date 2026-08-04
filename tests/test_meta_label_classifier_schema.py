"""MetaLabelClassifier feature-schema persistence + backwards-compat.

alpha-engine-config#3116: predict_proba raised "X has 13 features, but
LogisticRegression is expecting 16" for 100% of tickers on every inference
run since 2026-07-20 (barrier_win_prob null / evidence-gap alert, OBSERVE-only
— non-sizing). Root cause: save() pickled the bare CalibratedClassifierCV and
load() sourced feature_names from the external .pkl.meta.json sidecar only;
the sidecar drifted stale (13 names, 2026-05-30) relative to a retrained
16-feature serving artifact.

Fix mirrors model.meta_model.MetaModel's v2+ pickle contract (see
tests/test_meta_model_schema.py): feature_names travel INSIDE the pickle next
to the fitted estimator, so load() is sidecar-independent. These tests cover
the same three cases as MetaModel's schema tests, plus the new count-guard
that fails loud on a feature-count mismatch instead of silently truncating or
padding the input vector.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from model.meta_label_classifier import MetaLabelClassifier

_MIN_FIT_SAMPLES = 50


def _separable_dataset(feature_list, n: int = 240, seed: int = 0):
    """Two Gaussian blobs in feature-0 → linearly separable binary target,
    over an arbitrary named feature list (mirrors the helper in
    tests/test_meta_label_classifier.py but parameterized on feature count)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(n, len(feature_list)))
    y = (rng.random(n) > 0.5).astype(int)
    X[:, 0] += np.where(y == 1, 2.0, -2.0)
    return X, y.astype(float), list(feature_list)


def test_feature_names_embedded_in_pickle(tmp_path):
    """save() embeds feature_names IN the pickle (schema v2) so the
    load-bearing column order/count travels with the immutable model bytes,
    mirroring MetaModel's L4543 fix."""
    names = [f"feat_{i}" for i in range(5)]
    X, y, names = _separable_dataset(names)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)
    pkl_path = tmp_path / "clf.pkl"
    clf.save(pkl_path)

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict), "v2 pickle must be a schema dict, not a bare estimator"
    assert payload["_meta_label_classifier_schema"] >= 2
    assert payload["feature_names"] == names


def test_new_format_pickle_loads_sidecar_independent(tmp_path):
    """New-format (embedded-names) pickle loads correctly even when the
    sidecar is stale/mismatched — the embedded names are authoritative, not
    the sidecar's, closing the exact landmine that caused #3116."""
    names = [f"feat_{i}" for i in range(6)]
    X, y, names = _separable_dataset(names)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)
    pkl_path = tmp_path / "clf.pkl"
    clf.save(pkl_path)

    # Simulate a stale sidecar from a PRIOR (fewer-feature) model version —
    # exactly the #3116 scenario.
    meta_path = Path(str(pkl_path) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["feature_names"] = names[:3]  # stale, fewer names
    meta_path.write_text(json.dumps(meta))

    clf2 = MetaLabelClassifier.load(pkl_path)
    # feature_names comes from the embedded (authoritative) payload, NOT the
    # stale sidecar.
    assert clf2._feature_names == names
    np.testing.assert_allclose(clf.predict(X), clf2.predict(X), rtol=1e-9)


def test_old_format_pickle_falls_back_to_sidecar(tmp_path):
    """Backward-compat: a pre-fix pickle is a BARE CalibratedClassifierCV (no
    embedded names) — load() must still source feature_names from the
    sidecar, exactly like the pre-fix behavior, so existing S3 artifacts keep
    loading without requiring a retrain first."""
    names = [f"legacy_feat_{i}" for i in range(4)]
    X, y, names = _separable_dataset(names)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)

    # Write a LEGACY-format pickle: the bare estimator only (pre-fix save()).
    pkl_path = tmp_path / "legacy_raw.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(clf._model, f)
    Path(str(pkl_path) + ".meta.json").write_text(
        json.dumps({
            "feature_names": names,
            "n_samples": len(y),
            "auc": 0.9,
            "brier": 0.1,
            "base_rate": 0.5,
        })
    )

    clf2 = MetaLabelClassifier.load(pkl_path)
    assert clf2.is_fitted
    assert clf2._feature_names == names
    np.testing.assert_allclose(clf.predict(X), clf2.predict(X), rtol=1e-9)


def test_feature_count_mismatch_fails_loud_at_load(tmp_path):
    """The count-guard: a persisted feature_names list whose length doesn't
    match what the fitted estimator expects must raise at load() time with a
    clear diagnostic — never silently truncate/pad the input vector into a
    downstream sklearn shape error (or worse, a wrong-width prediction)."""
    names = [f"feat_{i}" for i in range(6)]
    X, y, names = _separable_dataset(names)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)
    pkl_path = tmp_path / "clf.pkl"
    clf.save(pkl_path)

    # Corrupt the EMBEDDED payload itself to carry the wrong feature count —
    # simulates a stale/mismatched artifact reaching load() with no honest
    # sidecar to fall back to.
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    payload["feature_names"] = names[:3]  # 3 names, model expects 6
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    # Sidecar deliberately absent so the mismatch can only be caught by the
    # count-guard, not accidentally papered over by a sidecar re-read.
    Path(str(pkl_path) + ".meta.json").unlink()

    with pytest.raises(ValueError, match="expects 6 features"):
        MetaLabelClassifier.load(pkl_path)


def test_old_format_pickle_with_sidecar_mismatch_also_fails_loud(tmp_path):
    """Same count-guard, exercised through the legacy (bare-estimator) load
    path: a sidecar whose feature_names count doesn't match the fitted
    estimator — the #3116 scenario exactly — must raise, not emit a
    wrong-width prediction that sklearn only rejects deep inside
    predict_proba on the first inference call."""
    names = [f"feat_{i}" for i in range(6)]
    X, y, names = _separable_dataset(names)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)

    pkl_path = tmp_path / "legacy_raw.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(clf._model, f)
    # Stale sidecar: only 3 of the 6 trained feature names (mirrors the
    # live #3116 13-vs-16 skew).
    Path(str(pkl_path) + ".meta.json").write_text(
        json.dumps({"feature_names": names[:3], "n_samples": len(y)})
    )

    with pytest.raises(ValueError, match="expects 6 features"):
        MetaLabelClassifier.load(pkl_path)


def test_save_load_roundtrip_still_predicts_identically(tmp_path):
    """Sanity: the schema change doesn't alter predictions for the normal
    (matched) roundtrip case."""
    names = [f"feat_{i}" for i in range(4)]
    X, y, names = _separable_dataset(names, n=_MIN_FIT_SAMPLES + 20)
    clf = MetaLabelClassifier().fit(X, y, feature_names=names)
    pkl_path = tmp_path / "clf.pkl"
    clf.save(pkl_path)

    clf2 = MetaLabelClassifier.load(pkl_path)
    assert clf2._feature_names == names
    np.testing.assert_allclose(clf.predict(X), clf2.predict(X), rtol=1e-9)
