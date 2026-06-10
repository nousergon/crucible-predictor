"""Producer-side contract test for the predictor → predictions.json boundary (L4520).

The predictor produces ``predictor/predictions/{trading_day}.json``, consumed
cross-repo by the executor (``alpha-engine/executor/signal_reader.py::
read_predictions``) FAIL-SOFT (``.get(...)`` defaults). A producer that silently
stops emitting a contract field would have the executor size/veto on a default
with no error — the silent-structural-break class L4520 exists to kill.

This pins the field set the producer MUST keep emitting, via AST source-binding
(mirrors the backtester ``test_pipeline_manifest.py`` anti-drift pattern — no
90-min inference run needed): the per-ticker prediction dict literal in
``inference/stages/run_inference.py`` and the envelope dict literal in
``inference/stages/write_output.py::write_predictions``. A future PR that drops a
contract key from either literal fails this test loud.

Contract SoT: ``alpha-engine-config/private-docs/PIPELINE_CONTRACT.yaml``
boundary ``predictions`` (drift-proof lib-hosted binding is a filed follow-up).
"""

from __future__ import annotations

import ast
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MUST match PIPELINE_CONTRACT.yaml boundary `predictions`.
_REQUIRED_ENVELOPE = {"date", "n_predictions", "predictions"}
_REQUIRED_PER_ITEM = {
    "ticker",
    "predicted_direction",
    "prediction_confidence",
    "predicted_alpha",
    "combined_rank",
}


def _dict_literals_with_key(path: str, marker_key: str) -> list[set[str]]:
    """Return the string-key sets of every dict literal in `path` that contains
    a string key == marker_key. AST-based so it's robust to formatting /
    multi-line literals (a regex over these paren-bearing dicts would be fragile)."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    out: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if marker_key in keys:
            out.append(keys)
    return out


def test_per_ticker_prediction_dict_carries_every_required_field():
    path = os.path.join(_REPO, "inference", "stages", "run_inference.py")
    # The prediction result dict is the literal carrying BOTH "ticker" and
    # "predicted_direction" (distinguishes it from other ticker-keyed dicts).
    candidates = [
        keys for keys in _dict_literals_with_key(path, "ticker")
        if "predicted_direction" in keys
    ]
    assert candidates, (
        "could not locate the per-ticker prediction result dict in run_inference.py "
        "(expected a dict literal with both 'ticker' and 'predicted_direction'). "
        "If the builder moved, update this test's locator."
    )
    for keys in candidates:
        missing = _REQUIRED_PER_ITEM - keys
        assert not missing, (
            f"per-ticker prediction dict dropped contract field(s): {sorted(missing)}. "
            "The executor reads these fail-soft, so a drop is a SILENT break — update "
            "PIPELINE_CONTRACT.yaml deliberately if intended."
        )


def test_predictions_envelope_carries_every_required_field():
    path = os.path.join(_REPO, "inference", "stages", "write_output.py")
    # The envelope is the literal carrying both "predictions" and "n_predictions".
    candidates = [
        keys for keys in _dict_literals_with_key(path, "predictions")
        if "n_predictions" in keys
    ]
    assert candidates, (
        "could not locate the predictions.json envelope dict in write_output.py "
        "(expected a dict literal with both 'predictions' and 'n_predictions')."
    )
    for keys in candidates:
        missing = _REQUIRED_ENVELOPE - keys
        assert not missing, (
            f"predictions.json envelope dropped contract field(s): {sorted(missing)}."
        )
