"""alpha-engine-config-I11520: sklearn pickles are version-checked on load,
and the training version travels with the artifact.

The 2026-09-23 rehearsal loaded ``BayesianRidge`` pickled by sklearn 1.9.0
under 1.9.1. sklearn's ``InconsistentVersionWarning`` went to stderr and
nothing read it. Skew is simulated here by pickling under a patched
``sklearn.base.__version__``, which is what sklearn stamps into the state.
"""
from __future__ import annotations

import logging
import pickle
import warnings

import numpy as np
import pytest
import sklearn
import sklearn.base
from sklearn.linear_model import BayesianRidge

from model.sklearn_pickle import (
    SklearnVersionSkewError,
    loads_checked,
    sklearn_version,
)


def _pickled_under(version: str) -> bytes:
    est = BayesianRidge().fit(np.arange(20.0).reshape(-1, 2), np.arange(10.0))
    real = sklearn.base.__version__
    sklearn.base.__version__ = version
    try:
        return pickle.dumps(est)
    finally:
        sklearn.base.__version__ = real


def _bump(version: str, *, minor: int = 0, patch: int = 0) -> str:
    major, mnr, *rest = version.split(".")
    pat = int(rest[0].split("rc")[0].split("dev")[0]) if rest else 0
    return f"{major}.{int(mnr) + minor}.{max(pat + patch, 0)}"


def test_same_version_is_ok_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obj, report = loads_checked(_pickled_under(sklearn.__version__), artifact="t")
    assert isinstance(obj, BayesianRidge)
    assert report["status"] == "ok"
    assert report["skews"] == []


def test_patch_skew_is_accepted_and_named(caplog):
    older = sklearn.__version__.rsplit(".", 1)[0] + ".999"
    with caplog.at_level(logging.WARNING, logger="model.sklearn_pickle"), \
         warnings.catch_warnings():
        # sklearn's raw warning must be consumed, not re-emitted.
        warnings.simplefilter("error")
        obj, report = loads_checked(_pickled_under(older), artifact="MetaModel meta_model.pkl")
    assert isinstance(obj, BayesianRidge)
    assert report["status"] == "patch_skew"
    assert report["skews"] == [{
        "estimator": "BayesianRidge", "saved_with": older,
        "loaded_with": sklearn.__version__,
    }]
    assert "sklearn_version_skew (MetaModel meta_model.pkl)" in caplog.text
    assert f"saved {older}" in caplog.text


def test_minor_skew_fails_loudly_naming_the_artifact():
    other_minor = _bump(sklearn.__version__, minor=-1)
    with pytest.raises(SklearnVersionSkewError, match="MetaModel meta_model.pkl") as exc:
        loads_checked(_pickled_under(other_minor), artifact="MetaModel meta_model.pkl")
    assert other_minor in str(exc.value)
    assert "BayesianRidge" in str(exc.value)


def test_unknown_saved_version_is_treated_as_breaking():
    with pytest.raises(SklearnVersionSkewError):
        loads_checked(_pickled_under("pre-0.18"), artifact="t")


def test_meta_model_records_the_training_sklearn_version(tmp_path):
    from model.meta_model import MetaModel

    mm = MetaModel()
    X = np.random.default_rng(0).normal(size=(80, 3))
    mm._feature_names = ["a", "b", "c"]
    mm._model = BayesianRidge().fit(X, X @ np.array([1.0, -1.0, 0.5]))
    mm._fitted = True
    path = tmp_path / "meta_model.pkl"
    mm.save(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    assert payload["sklearn_version"] == sklearn_version()
    loaded = MetaModel.load(path)
    assert loaded._sklearn_load_report["status"] == "ok"
