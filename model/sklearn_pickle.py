"""model/sklearn_pickle.py — version-checked unpickling of sklearn artifacts.

alpha-engine-config-I11520. Pickled sklearn estimators are only guaranteed to
load under the exact sklearn version that saved them. The rehearsal of
2026-09-23 loaded ``BayesianRidge`` pickled by 1.9.0 under 1.9.1, and sklearn
said so with an ``InconsistentVersionWarning`` on stderr that nothing read.
``requirements*.txt`` pins ``scikit-learn>=1.9.0``, so every image rebuild
may move the runtime forward under artifacts that do not move.

Every sklearn pickle this repo loads goes through :func:`loads_checked`.
It captures sklearn's own warning, which carries the estimator name and both
versions, and applies one rule:

* **same major.minor** (patch drift, e.g. 1.9.0 -> 1.9.1): accepted and
  DEGRADED BY NAME. One WARNING names the artifact, the estimator and both
  versions, and the returned report says ``patch_skew``. sklearn keeps
  pickle compatibility within a minor series in practice, and this is the
  state every serving artifact is in today, so raising here would take
  preopen inference down with no evidence of a defect.
* **different major or minor**: :class:`SklearnVersionSkewError`. That is
  the upgrade sklearn warns "might lead to breaking code or invalid
  results", and an invalid result served silently is worse than a loud
  failure. The deploy canary loads the model, so this fails the deploy
  before it fails a preopen.

The artifacts themselves are never re-saved here. Re-pickling under a new
version is a retrain-shaped change and belongs to the training run.
"""
from __future__ import annotations

import logging
import pickle
import warnings
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "SklearnVersionSkewError",
    "sklearn_version",
    "loads_checked",
    "load_checked",
]


class SklearnVersionSkewError(RuntimeError):
    """A pickled sklearn estimator was saved under a different major/minor
    sklearn version than the one loading it."""


def sklearn_version() -> str | None:
    """The running sklearn version, recorded beside every saved artifact."""
    try:
        import sklearn
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        return None
    return str(sklearn.__version__)


def _major_minor(version: str | None) -> tuple[str, ...] | None:
    if not version:
        return None
    parts = str(version).split(".")
    return tuple(parts[:2]) if len(parts) >= 2 else None


def loads_checked(data: bytes, *, artifact: str) -> tuple[Any, dict]:
    """Unpickle ``data`` and return ``(obj, report)``.

    ``report["status"]`` is ``"ok"`` (no skew) or ``"patch_skew"`` (accepted,
    named). A major/minor skew, or a saved version that cannot be read, raises
    :class:`SklearnVersionSkewError`.
    """
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # pragma: no cover
        InconsistentVersionWarning = None  # type: ignore[assignment]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = pickle.loads(data)

    current = sklearn_version()
    skews: list[dict] = []
    for w in caught:
        if InconsistentVersionWarning is not None and issubclass(w.category, InconsistentVersionWarning):
            msg = w.message
            skews.append({
                "estimator": getattr(msg, "estimator_name", None),
                "saved_with": getattr(msg, "original_sklearn_version", None),
                "loaded_with": getattr(msg, "current_sklearn_version", current),
            })
        else:
            # Not ours to judge: re-emit so nothing is swallowed.
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    report: dict = {"artifact": artifact, "sklearn_loaded_with": current,
                    "status": "ok", "skews": skews}
    if not skews:
        return obj, report

    # An unknown saved version (None, or sklearn's own "pre-0.18") cannot be
    # shown to share a minor series, so it is treated as breaking.
    breaking = [
        s for s in skews
        if _major_minor(s["saved_with"]) is None
        or _major_minor(s["saved_with"]) != _major_minor(s["loaded_with"])
    ]
    if breaking:
        detail = "; ".join(
            f"{s['estimator']} saved with sklearn {s['saved_with']}, loading with "
            f"{s['loaded_with']}" for s in breaking
        )
        raise SklearnVersionSkewError(
            f"{artifact}: refusing to load across a sklearn major/minor version "
            f"change ({detail}). Retrain the artifact under the running version, "
            "or pin scikit-learn to the version that saved it "
            "(alpha-engine-config-I11520)."
        )
    report["status"] = "patch_skew"
    log.warning(
        "sklearn_version_skew (%s): %s. Accepted: same major.minor, "
        "degraded by name (alpha-engine-config-I11520).",
        artifact,
        "; ".join(f"{s['estimator']} saved {s['saved_with']} -> loaded {s['loaded_with']}"
                  for s in skews),
    )
    return obj, report


def load_checked(fileobj, *, artifact: str) -> tuple[Any, dict]:
    """File-object form of :func:`loads_checked`."""
    return loads_checked(fileobj.read(), artifact=artifact)
