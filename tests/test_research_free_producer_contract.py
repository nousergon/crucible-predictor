"""Producer-side contract test for the predictor → predictions_research_free
boundary (alpha-engine-config-I10067).

Sibling of ``tests/test_predictions_producer_contract.py``. The predictor
produces ``predictor/predictions_research_free/{trading_day}.json``, consumed
cross-repo by crucible-research's ``producers/filling_arms.py::
load_research_free_pool`` (the ``scanner_predictor_direct`` filling arm),
which ranks the entries by ``predicted_alpha``.

Two failure classes this pins:

1. **Schema drift.** The consumer reads ``predictions[].ticker`` and
   ``predictions[].predicted_alpha`` out of a ``{"date": ..., "predictions":
   [...]}`` envelope. A producer that renames or drops either field, or moves
   the list, breaks the arm — and (because the consumer raises rather than
   emitting an empty cohort) it breaks it LOUD but a week later, on the
   canonical Saturday. Pin the literals here so the break lands on the PR.

2. **The dead-producer class I10067 is about.** ``run_research_free_inference``
   sat in the tree with ZERO callers from 2026-08 until I10067 — a producer
   nobody invoked, whose consumer therefore read the backtester's offline
   parquet instead. ``test_the_research_free_producer_is_actually_invoked``
   fails if the daily pipeline ever stops calling it again.

Both tests bind to source via AST (mirrors the sibling contract test's
anti-drift pattern) — no ArcticDB, no S3, no 90-minute inference run.
"""

from __future__ import annotations

import ast
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Consumed by crucible-research producers/filling_arms.py::load_research_free_pool.
_REQUIRED_ENVELOPE = {"schema_version", "date", "n_predictions", "predictions"}
_REQUIRED_PER_ITEM = {"ticker", "prediction_date", "predicted_alpha"}

_PRODUCER = os.path.join(_REPO, "inference", "research_free_inference.py")
_STAGE = os.path.join(_REPO, "inference", "stages", "research_free.py")
_PIPELINE = os.path.join(_REPO, "inference", "pipeline.py")


def _parse(path: str) -> ast.Module:
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _dict_literals_with_key(path: str, marker_key: str) -> list[set[str]]:
    out: list[set[str]] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if marker_key in keys:
            out.append(keys)
    return out


def test_per_entry_dict_carries_every_required_field():
    candidates = [
        keys for keys in _dict_literals_with_key(_PRODUCER, "ticker")
        if "predicted_alpha" in keys
    ]
    assert candidates, (
        "could not locate the per-ticker research-free entry dict in "
        "inference/research_free_inference.py (expected a dict literal with both "
        "'ticker' and 'predicted_alpha'). If the builder moved, update this locator."
    )
    for keys in candidates:
        missing = _REQUIRED_PER_ITEM - keys
        assert not missing, (
            f"research-free entry dict dropped contract field(s): {sorted(missing)}. "
            "crucible-research's load_research_free_pool ranks on 'predicted_alpha' "
            "keyed by 'ticker' — dropping either strands the scanner_predictor_direct "
            "filling arm (alpha-engine-config-I10067)."
        )


def test_envelope_carries_every_required_field():
    candidates = _dict_literals_with_key(_PRODUCER, "predictions")
    candidates = [keys for keys in candidates if "n_predictions" in keys]
    assert candidates, (
        "could not locate the research-free envelope dict in "
        "inference/research_free_inference.py (expected a dict literal with both "
        "'predictions' and 'n_predictions')."
    )
    for keys in candidates:
        missing = _REQUIRED_ENVELOPE - keys
        assert not missing, (
            f"research-free envelope dropped contract field(s): {sorted(missing)}."
        )


def test_the_artifact_keys_are_the_declared_config_constants():
    """The producer must write the keys the consumer reads, sourced from
    config.py rather than restated as literals in the writer."""
    import config as cfg

    assert cfg.PREDICTIONS_RESEARCH_FREE_KEY == (
        "predictor/predictions_research_free/{date}.json"
    )
    assert cfg.PREDICTIONS_RESEARCH_FREE_LATEST_KEY == (
        "predictor/predictions_research_free/latest.json"
    )

    src = open(_PRODUCER).read()
    assert "cfg.PREDICTIONS_RESEARCH_FREE_KEY" in src
    assert "cfg.PREDICTIONS_RESEARCH_FREE_LATEST_KEY" in src


def test_the_research_free_producer_is_actually_invoked():
    """A producer with no caller writes nothing, and its consumer silently
    reads something else instead — the exact defect alpha-engine-config-I10067
    records. Bind the call site so a future refactor cannot re-orphan it."""
    calls = [
        node for node in ast.walk(_parse(_STAGE))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_research_free_inference"
    ]
    assert calls, (
        "inference/stages/research_free.py no longer calls "
        "run_research_free_inference — the producer is orphaned again "
        "(alpha-engine-config-I10067)."
    )


def test_the_research_free_stage_is_registered_in_the_daily_pipeline():
    """And the stage that calls it must be in STAGES, or nothing runs it."""
    src = open(_PIPELINE).read()
    tree = ast.parse(src, filename=_PIPELINE)
    stages: list[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets
        ):
            continue
        assert isinstance(node.value, ast.List), "STAGES is no longer a list literal"
        for elt in node.value.elts:
            assert isinstance(elt, ast.Tuple)
            stages.append(tuple(
                e.value if isinstance(e, ast.Constant) else None for e in elt.elts
            ))
    assert stages, "could not locate the STAGES table in inference/pipeline.py"

    names = [s[0] for s in stages]
    assert "research_free" in names, (
        "the research_free stage is not registered in inference/pipeline.py::STAGES "
        "— the daily producer would never run (alpha-engine-config-I10067)."
    )
    assert "inference.stages.research_free" in [s[1] for s in stages]

    # Ordering invariant: the live predictions artifact must already be written
    # before the counterfactual runs, so a research-free failure can never delay
    # or affect the executor's input.
    assert names.index("research_free") > names.index("write_output"), (
        "research_free must run AFTER write_output — the live predictions "
        "artifact must be on S3 before the counterfactual producer runs."
    )


def test_a_research_free_failure_does_not_abort_the_daily_pipeline():
    """Non-critical by design: nothing on the live trading path consumes
    predictions_research_free, so this counterfactual must never fail the
    weekday PredictorInference Lambda and halt trading. It is not SILENT,
    though — see the companion assertion on the ops-alert surface."""
    tree = ast.parse(open(_PIPELINE).read(), filename=_PIPELINE)
    critical_by_name: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets
        ):
            for elt in node.value.elts:
                name, _mod, crit = elt.elts
                critical_by_name[name.value] = crit.value
    assert critical_by_name.get("research_free") is False


def test_a_research_free_failure_reaches_the_ops_alert_surface():
    """Non-critical must not mean unobserved: run_pipeline's generic handler
    only logs a warning, so the stage publishes its own alert."""
    src = open(_STAGE).read()
    assert "publish_ops_alert" in src, (
        "inference/stages/research_free.py no longer publishes an ops alert on "
        "failure — a non-critical stage that only logs is a silent swallow."
    )
    # The source literal must stay a REGISTERED alert class. `predictor_inference`
    # (nousergon-data infrastructure/overseer/playbooks.yaml) is the operator-ruled
    # `severities: [dynamic]` row covering every failure shape inside this Lambda;
    # inventing a new literal here reddens the alert-class PR guard until a
    # companion nousergon-data row lands.
    assert '"alpha-engine-predictor-inference"' in src, (
        "the research_free stage's alert source is no longer the registered "
        "predictor_inference class — an unregistered source has no Overseer "
        "playbook and fails the alert-class PR guard."
    )
    # Measurement-coverage signal, not a trading halt — severity and response
    # are chosen together (the 2026-08-28 alert-destination lesson).
    assert 'severity="warning"' in src or "severity='warning'" in src
