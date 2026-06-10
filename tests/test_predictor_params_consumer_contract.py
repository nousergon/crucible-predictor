"""L4520 slice 4 — consumer contract for config/predictor_params.json.

Pins the predictor side of the PIPELINE_CONTRACT.yaml `predictor_params`
boundary (producer side: backtester tests/test_config_writers_producer_
contract.py, #315 — two writers merge over the live key: veto_analysis's
veto leg + predictor_optimizer's Phase 4 leg; the regime/drawdown flip keys
are operator-written). The inference veto gate reads explicit keys with
defaults, so a producer rename (e.g. `veto_confidence` → something else)
silently reverts the gate to cfg.MIN_CONFIDENCE forever — this test catches
the rename at PR time.

The declared read set is a hard-coded mirror of the YAML (per-repo CI can't
import the config repo — the established per-repo contract-test pattern).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "inference" / "stages" / "write_output.py"

# Mirror of PIPELINE_CONTRACT.yaml predictor_params consumers[].reads
DECLARED_READS = {
    "veto_confidence",
    "regime_veto_enabled", "regime_veto_scale", "regime_veto_cap",
    "regime_forced_bear_enabled", "drawdown_regime_enabled",
}


def _params_keys_read() -> set[str]:
    """Every string key read off the ``params`` dict anywhere in
    write_output.py — both ``params["k"]`` subscripts and ``params.get("k",
    ...)`` calls — extracted from source so a new read can't dodge the pin."""
    tree = ast.parse(_SRC.read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "params"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "params"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def test_veto_gate_reads_exactly_the_declared_keys():
    read = _params_keys_read()
    undeclared = read - DECLARED_READS
    unread = DECLARED_READS - read
    assert not undeclared, (
        f"write_output.py reads predictor_params key(s) the contract does not "
        f"declare: {sorted(undeclared)} — add them to PIPELINE_CONTRACT.yaml "
        f"(predictor_params) AND confirm a producer/operator actually writes "
        f"them, else the read is dead config."
    )
    assert not unread, (
        f"contract declares consumer reads {sorted(unread)} that "
        f"write_output.py no longer performs — prune the YAML mirror + this "
        f"test together."
    )


def test_membership_check_matches_primary_key():
    # The gate applies S3 tuning only via `"veto_confidence" in params` —
    # pin that the membership check and the declared primary key agree (a
    # producer-side rename would otherwise revert the gate to
    # cfg.MIN_CONFIDENCE silently, forever).
    src = _SRC.read_text()
    assert '"veto_confidence" in params' in src
