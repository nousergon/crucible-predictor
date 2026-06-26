"""Unit tests for inference.pipeline_contract_check — pre-spend pipeline-contract
preflight probe (L4595 / config#693).

Exercises the self-consistency validation rules + the fail-open degraded mode.
``_fetch_raw`` (the GitHub read) is mocked so tests are hermetic.
"""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import yaml

import inference.pipeline_contract_check as pcc


# ── fixtures ─────────────────────────────────────────────────────────────────

# A minimal but internally-consistent contract: one boundary whose artifact_id
# exists in the registry and whose consumer only reads declared fields.
_GOOD_CONTRACT = {
    "schema_version": 1,
    "boundaries": [
        {
            "boundary_id": "signals",
            "producer_stage": "research",
            "producer_repo": "alpha-engine-research",
            "artifact_id": "research_signals",
            "required_top_level_fields": ["date", "signals", "universe"],
            "required_per_item_fields": ["ticker", "score"],
            "per_item_collections": ["universe"],
            "consumers": [
                {
                    "repo": "alpha-engine",
                    "reads": ["signals", "ticker"],
                    "fail_posture": "fail_loud",
                }
            ],
        }
    ],
}

_GOOD_REGISTRY = {"artifacts": [{"artifact_id": "research_signals"}]}


def _yaml(obj) -> str:
    return yaml.safe_dump(obj)


def _patch_raw(contract_obj, registry_obj):
    """Patch _fetch_raw to return YAML for the contract/registry (or None)."""
    contract_text = None if contract_obj is None else _yaml(contract_obj)
    registry_text = None if registry_obj is None else _yaml(registry_obj)

    def _side_effect(path, **_):
        if path == pcc._CONTRACT_PATH:
            return contract_text
        if path == pcc._REGISTRY_PATH:
            return registry_text
        return None

    return patch.object(pcc, "_fetch_raw", side_effect=_side_effect)


# ── happy path ───────────────────────────────────────────────────────────────

def test_consistent_contract_no_violation():
    with _patch_raw(_GOOD_CONTRACT, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is False
    assert out["reason"] == "in_sync"
    assert out["violations"] == []
    assert out["boundary_count"] == 1


# ── confirmed violations halt ────────────────────────────────────────────────

def test_dangling_artifact_id_halts():
    contract = {
        "schema_version": 1,
        "boundaries": [
            {**_GOOD_CONTRACT["boundaries"][0], "artifact_id": "ghost_artifact"}
        ],
    }
    with _patch_raw(contract, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert out["reason"] == "violation_detected"
    assert any("ghost_artifact" in v and "ARTIFACT_REGISTRY" in v
               for v in out["violations"])


def test_consumer_reads_undeclared_field_halts():
    contract = {
        "schema_version": 1,
        "boundaries": [
            {
                **_GOOD_CONTRACT["boundaries"][0],
                "consumers": [
                    {
                        "repo": "alpha-engine",
                        "reads": ["signals", "not_a_field"],
                        "fail_posture": "fail_soft",
                    }
                ],
            }
        ],
    }
    with _patch_raw(contract, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert any("not_a_field" in v for v in out["violations"])


def test_missing_required_key_halts():
    boundary = dict(_GOOD_CONTRACT["boundaries"][0])
    del boundary["consumers"]
    with _patch_raw({"schema_version": 1, "boundaries": [boundary]}, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert any("missing required keys" in v for v in out["violations"])


def test_duplicate_boundary_id_halts():
    b = _GOOD_CONTRACT["boundaries"][0]
    with _patch_raw({"schema_version": 1, "boundaries": [b, dict(b)]}, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert any("duplicate boundary_id" in v for v in out["violations"])


def test_bad_fail_posture_halts():
    contract = {
        "schema_version": 1,
        "boundaries": [
            {
                **_GOOD_CONTRACT["boundaries"][0],
                "consumers": [
                    {"repo": "x", "reads": ["signals"], "fail_posture": "explode"}
                ],
            }
        ],
    }
    with _patch_raw(contract, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert any("fail_posture" in v for v in out["violations"])


def test_per_item_collection_not_in_top_level_halts():
    contract = {
        "schema_version": 1,
        "boundaries": [
            {**_GOOD_CONTRACT["boundaries"][0], "per_item_collections": ["ghost_coll"]}
        ],
    }
    with _patch_raw(contract, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert any("ghost_coll" in v for v in out["violations"])


def test_collects_all_violations_not_just_first():
    # Two independent breaches in one contract → both surfaced in one halt
    # (unlike the CI validator, which sys.exit(1)s on the first).
    contract = {
        "schema_version": 1,
        "boundaries": [
            {
                **_GOOD_CONTRACT["boundaries"][0],
                "artifact_id": "ghost_artifact",        # breach 1
                "consumers": [
                    {"repo": "x", "reads": ["nope"], "fail_posture": "fail_loud"}  # breach 2
                ],
            }
        ],
    }
    with _patch_raw(contract, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is True
    assert len(out["violations"]) >= 2


# ── fail-open degraded mode ──────────────────────────────────────────────────

def test_contract_fetch_failure_fails_open():
    with _patch_raw(None, _GOOD_REGISTRY):  # contract unreachable
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is False
    assert out["reason"] == "fetch_failed"


def test_registry_fetch_failure_fails_open():
    with _patch_raw(_GOOD_CONTRACT, None):  # registry unreachable
        out = pcc.check_pipeline_contract()
    assert out["has_violation"] is False
    assert out["reason"] == "fetch_failed"


def test_malformed_yaml_fails_open():
    bad_yaml = "schema_version: 1\nboundaries: [unterminated"
    def _side_effect(path, **_):
        return bad_yaml
    with patch.object(pcc, "_fetch_raw", side_effect=_side_effect):
        out = pcc.check_pipeline_contract()
    # A parse failure is the checker's own fragility — must never false-halt.
    assert out["has_violation"] is False
    assert out["reason"] == "fetch_failed"


# ── parse-helper sanity ──────────────────────────────────────────────────────

def test_registry_artifact_ids_extraction():
    reg = {"artifacts": [{"artifact_id": "a"}, {"artifact_id": "b"}, {"no_id": 1}]}
    assert pcc._registry_artifact_ids(reg) == {"a", "b"}


def test_validates_real_contract_shape():
    # Sanity: the fixture round-trips through YAML the same way the real raw
    # file would, so the YAML path (not just dict-in) is exercised.
    contract = yaml.safe_load(textwrap.dedent(_yaml(_GOOD_CONTRACT)))
    ids = pcc._registry_artifact_ids(_GOOD_REGISTRY)
    assert pcc._validate_contract(contract, ids) == []
