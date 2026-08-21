"""Unit tests for inference.pipeline_contract_check — pre-spend pipeline-contract
preflight probe (L4595 / config#693).

Exercises the self-consistency validation rules + the fail-open degraded mode.
``_fetch_source`` (the S3 read) is mocked so tests are hermetic.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
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


_PUBLISHED_AT = datetime(2026, 8, 14, 1, 5, 0, tzinfo=timezone.utc)


def _patch_raw(contract_obj, registry_obj, failure_reason=pcc._REASON_FETCH_FAILED):
    """Patch _fetch_source to return YAML for the contract/registry (or a miss).

    `None` for either object means that S3 read failed, with `failure_reason`
    as its reason — the signature keeps the (text, reason, last_modified)
    contract the real `_fetch_source` returns.
    """
    contract_text = None if contract_obj is None else _yaml(contract_obj)
    registry_text = None if registry_obj is None else _yaml(registry_obj)

    def _one(text):
        if text is None:
            return None, failure_reason, None
        return text, None, _PUBLISHED_AT

    def _side_effect(key, **_):
        if key == pcc._CONTRACT_KEY:
            return _one(contract_text)
        if key == pcc._REGISTRY_KEY:
            return _one(registry_text)
        return None, pcc._REASON_MISSING, None

    return patch.object(pcc, "_fetch_source", side_effect=_side_effect)


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
    # alpha-engine-config-I7048: an unmeasured gate must not report a
    # definite verdict — has_violation is OMITTED, not set to False, so the
    # SF's IsPresent-guarded Choice routes to the visible degraded path
    # instead of a silent pass.
    assert "has_violation" not in out
    assert out["reason"] == pcc._REASON_FETCH_FAILED
    assert out["boundary_count"] is None


def test_registry_fetch_failure_fails_open():
    with _patch_raw(_GOOD_CONTRACT, None):  # registry unreachable
        out = pcc.check_pipeline_contract()
    assert "has_violation" not in out
    assert out["reason"] == pcc._REASON_FETCH_FAILED
    assert out["boundary_count"] is None


def test_malformed_yaml_fails_open():
    bad_yaml = "schema_version: 1\nboundaries: [unterminated"

    def _side_effect(key, **_):
        return bad_yaml, None, _PUBLISHED_AT

    with patch.object(pcc, "_fetch_source", side_effect=_side_effect):
        out = pcc.check_pipeline_contract()
    # A parse failure is the checker's own fragility — must never false-halt,
    # and must never report a measured verdict it does not have. It gets its
    # OWN reason (alpha-engine-config-I7281): the publisher validates before it
    # copies, so unparseable YAML in the bucket means something else wrote it —
    # a different operator action from a read that did not land.
    assert "has_violation" not in out
    assert out["reason"] == pcc._REASON_PARSE_FAILED


# ── alpha-engine-config-I7281: the source is S3, and WHY it failed matters ───

def test_reads_the_published_s3_keys_not_github():
    """The gate reads S3. It read raw.githubusercontent.com against a PRIVATE
    repo for its entire life, so every one of 190 invocations in the 30 days
    to 2026-08-13 returned fetch_failed and it never measured anything."""
    assert not hasattr(pcc, "_fetch_raw")
    assert not hasattr(pcc, "_RAW_URL")
    assert pcc._SOURCE_BUCKET == "alpha-engine-research"
    assert pcc._CONTRACT_KEY == "_pipeline_contract/PIPELINE_CONTRACT.yaml"
    # The registry is read at the key sync-artifact-registry.yml has published
    # to since 2026-06-05 — NOT republished under a contract-specific prefix.
    assert pcc._REGISTRY_KEY == "_freshness_monitor/ARTIFACT_REGISTRY.yaml"


def _client_error(code: str, status: int):
    from botocore.exceptions import ClientError
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )


@pytest.mark.parametrize(("code", "status", "expected"), [
    ("NoSuchKey", 404, pcc._REASON_MISSING),
    ("AccessDenied", 403, pcc._REASON_FORBIDDEN),
    ("InternalError", 500, pcc._REASON_FETCH_FAILED),
])
def test_unreachable_and_unauthorized_get_different_reasons(code, status, expected):
    """The defect that hid this gate for its whole life.

    GitHub answers an unauthorized read of a private repo with 404, not 403,
    so 'this can never work' and 'the network blipped' arrived as the same
    `fetch_failed` string. On S3 they are genuinely distinguishable, and a
    checker that collapses them back invites the same misreading.
    """
    client = MagicMock()
    client.get_object.side_effect = _client_error(code, status)
    with patch.object(pcc, "_s3", return_value=client):
        text, reason, mtime = pcc._fetch_source(pcc._CONTRACT_KEY)
    assert text is None
    assert mtime is None
    assert reason == expected


def test_a_missing_object_does_not_report_the_registrys_reason():
    """When both reads fail, the CONTRACT's reason wins.

    The registry is only the set the contract is checked against; reporting
    its reason for a missing contract sends the operator to the wrong
    publisher.
    """
    def _side_effect(key, **_):
        if key == pcc._CONTRACT_KEY:
            return None, pcc._REASON_MISSING, None
        return None, pcc._REASON_FORBIDDEN, None

    with patch.object(pcc, "_fetch_source", side_effect=_side_effect):
        out = pcc.check_pipeline_contract()
    assert out["reason"] == pcc._REASON_MISSING


def test_measured_payload_names_how_fresh_the_sources_were():
    """A gate that says in_sync without saying WHAT it read cannot tell a
    current contract from one published months ago."""
    with _patch_raw(_GOOD_CONTRACT, _GOOD_REGISTRY):
        out = pcc.check_pipeline_contract()
    assert out["status"] == pcc.STATUS_MEASURED
    assert out["contract_published_at"] == _PUBLISHED_AT.isoformat()
    assert out["registry_published_at"] == _PUBLISHED_AT.isoformat()


def test_the_whole_payload_is_json_serializable():
    """The SF's Lambda integration JSON-encodes this. A datetime here would
    fail the invocation, which the Catch would then report as a Lambda
    failure — the exact miscategorisation alpha-engine-config-I7302 fixed on
    the alert side."""
    import json

    with _patch_raw(_GOOD_CONTRACT, _GOOD_REGISTRY):
        json.dumps(pcc.check_pipeline_contract())
    with _patch_raw(None, None):
        json.dumps(pcc.check_pipeline_contract())


def test_check_pipeline_contract_takes_no_branch():
    """The source is the PUBLISHED copy of main, not a ref the caller picks.

    A `branch` parameter would invite gating a production run on an unmerged
    branch.

    alpha-engine-config-I7954 narrowed this from "no parameters at all" to its
    actual intent. `probe` was added as a keyword-only flag that moves ONLY the
    log severity of a detected violation (asserted byte-identical payload in
    test_probe_leaves_the_payload_identical) — it selects nothing about WHICH
    copy is read, which is the property this test exists to protect. Any new
    parameter must be keyword-only and must not name a source.
    """
    import inspect
    params = inspect.signature(pcc.check_pipeline_contract).parameters
    assert "branch" not in params
    assert "ref" not in params
    assert set(params) <= {"probe"}, (
        f"unexpected parameter(s) on check_pipeline_contract: {sorted(params)}"
    )
    for name, param in params.items():
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only"
        )


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


# ── probe=True: a synthetic invocation records, it does not page ─────────────
# alpha-engine-config-I7954, mirroring inference/lib_pin_drift.py. This action
# already received `dry_run: true` from `infrastructure/deploy.sh`'s canary and
# still logged violations at ERROR, where flow-doctor is attached — which is
# why threading `dry_run` alone was not the fix for the paging class.


def _violating_contract():
    return {
        "schema_version": 1,
        "boundaries": [
            {**_GOOD_CONTRACT["boundaries"][0], "artifact_id": "ghost_artifact"}
        ],
    }


def test_probe_downgrades_a_violation_to_warning(caplog):
    with caplog.at_level("WARNING", logger=pcc.log.name), _patch_raw(
        _violating_contract(), _GOOD_REGISTRY
    ):
        out = pcc.check_pipeline_contract(probe=True)
    assert out["has_violation"] is True
    records = [
        r for r in caplog.records
        if "Pipeline-contract preflight VIOLATION" in r.getMessage()
    ]
    assert records, "the finding must still be recorded, only at a lower level"
    assert [r.levelname for r in records] == ["WARNING"]


def test_default_invocation_still_logs_a_violation_at_error(caplog):
    with caplog.at_level("WARNING", logger=pcc.log.name), _patch_raw(
        _violating_contract(), _GOOD_REGISTRY
    ):
        pcc.check_pipeline_contract()
    records = [
        r for r in caplog.records
        if "Pipeline-contract preflight VIOLATION" in r.getMessage()
    ]
    assert [r.levelname for r in records] == ["ERROR"]


def test_probe_leaves_the_payload_identical():
    with _patch_raw(_violating_contract(), _GOOD_REGISTRY):
        as_probe = pcc.check_pipeline_contract(probe=True)
    with _patch_raw(_violating_contract(), _GOOD_REGISTRY):
        as_gate = pcc.check_pipeline_contract()
    assert as_probe == as_gate
