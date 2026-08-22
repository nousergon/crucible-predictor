"""No pre-spend gate may emit an unmeasured result a consumer can read as clean.

alpha-engine-config-I7277. `sf-pipeline-policy.md` §2.3a rule 2: *a missing
verdict propagates as `UNKNOWN`, never as pass*. §5's carve-out lets these
pre-spend probes fail OPEN — that stays — but a fail-open payload must say it
measured nothing, in every field it carries.

Measured live before this suite existed, on `ne-weekly-freshness-pipeline`
executions `watch-rerun-2026-08-13-2` and `-3` (predictor Lambda v473):

    "pipeline_contract_result": {"Payload": {
        "violations": [], "boundary_count": null, "reason": "fetch_failed"}}

`has_violation` was correctly absent (alpha-engine-config-I7048) and the SF's
IsPresent-guarded `PipelineContractGate` correctly routed to
`PipelineContractGateDegraded`. But `violations` — the field a consumer is
actually shaped to read — carried an empty list, which is an assertion, not an
absence. `boundary_count: null` said nothing was counted while `violations: []`
said nothing was wrong. Both cannot be true of the same run.

The invariant this suite pins is deliberately stated as a PROHIBITION on the
whole payload rather than an assertion about one key: an honest gate has no
field at all whose value reads as a verdict when no verdict was reached. That
is what makes it survive the next field somebody adds.

Both probes are covered by one parametrisation on purpose — they are the same
gate family with the same fail-open licence, and I7048's fix reached only one
of them by hand before a sweep caught the other.
"""

from __future__ import annotations

import pytest

from inference import lib_pin_drift, pipeline_contract_check

# Per probe: the verdict key the SF Choice reads, and every EVIDENCE key whose
# empty value would read as "measured, and found nothing". Neither class may
# appear in an unmeasured payload.
VERDICT_KEYS = {
    "check_pipeline_contract": "has_violation",
    "check_lib_pin_drift": "has_drift",
}
EVIDENCE_KEYS = {
    "check_pipeline_contract": ("violations",),
    "check_lib_pin_drift": ("offenders",),
}


def _drive_unmeasured(action: str, monkeypatch) -> dict:
    """Force each probe down its fail-open branch and return the raw payload."""
    if action == "check_pipeline_contract":
        monkeypatch.setattr(
            pipeline_contract_check,
            "_fetch_source",
            lambda *a, **k: (None, pipeline_contract_check._REASON_MISSING, None),
        )
        return pipeline_contract_check.check_pipeline_contract()
    monkeypatch.setattr(
        lib_pin_drift,
        "_fetch_repo_pin",
        lambda *a, **k: lib_pin_drift.PinRead(
            None, lib_pin_drift.UNREACHABLE, "patched"
        ),
    )
    return lib_pin_drift.check_lib_pin_drift()


def _drive_measured(action: str, monkeypatch) -> dict:
    """Drive each probe down a genuinely-measured, no-defect-found branch."""
    if action == "check_pipeline_contract":
        contract = (
            "schema_version: 1\n"
            "boundaries:\n"
            "  - boundary_id: b1\n"
            "    producer_stage: s\n"
            "    producer_repo: r\n"
            "    artifact_id: a1\n"
            "    required_top_level_fields: [items]\n"
            "    required_per_item_fields: [x]\n"
            "    per_item_collections: [items]\n"
            "    consumers:\n"
            "      - repo: c\n"
            "        reads: [items, x]\n"
            "        fail_posture: fail_loud\n"
        )
        registry = "artifacts:\n  - artifact_id: a1\n"
        monkeypatch.setattr(
            pipeline_contract_check,
            "_fetch_source",
            lambda key, **k: (
                (contract if key == pipeline_contract_check._CONTRACT_KEY else registry),
                None,
                None,
            ),
        )
        return pipeline_contract_check.check_pipeline_contract()
    monkeypatch.setattr(
        lib_pin_drift,
        "_fetch_repo_pin",
        lambda *a, **k: lib_pin_drift.PinRead("v999.0.0", None),
    )
    return lib_pin_drift.check_lib_pin_drift()


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_unmeasured_payload_declares_itself_unknown(action, monkeypatch):
    """The fail-open payload states UNKNOWN outright — not by omission alone."""
    result = _drive_unmeasured(action, monkeypatch)
    assert result.get("status") == "UNKNOWN", (
        f"{action}: an unmeasured gate must carry status=UNKNOWN so a consumer "
        f"can tell 'could not measure' from 'measured and clean' without "
        f"inferring it from an absent key; got {result!r}"
    )


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_unmeasured_payload_omits_its_verdict_key(action, monkeypatch):
    """alpha-engine-config-I7048's invariant, held under I7277's change."""
    result = _drive_unmeasured(action, monkeypatch)
    key = VERDICT_KEYS[action]
    assert key not in result, (
        f"{action}: {key} must be ABSENT when nothing was measured — a present "
        f"false is a definite verdict the probe never reached; got {result!r}"
    )


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_unmeasured_payload_carries_no_empty_evidence_list(action, monkeypatch):
    """THE regression this suite exists for (I7277).

    Verified to FAIL against the pre-fix payload shape: `check_pipeline_contract`
    returned `violations: []` and `check_lib_pin_drift` returned `offenders: []`
    on their fail-open branches, so a consumer keyed on the evidence list read a
    measurement failure as a clean check.
    """
    result = _drive_unmeasured(action, monkeypatch)
    present = [k for k in EVIDENCE_KEYS[action] if k in result]
    assert not present, (
        f"{action}: evidence key(s) {present} present in an unmeasured payload "
        f"{result!r}. An empty evidence list is an assertion that the probe "
        f"looked and found nothing — it did not look at all. Omit the key."
    )


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_unmeasured_payload_has_no_field_readable_as_a_clean_verdict(
    action, monkeypatch
):
    """Generalised form: nothing in the payload may read as a passed check.

    Stated over the whole dict so a NEW field cannot reintroduce the defect
    under a different name. A False boolean or an empty collection both read as
    'checked, nothing found'; only `reason`, `status`, `pins` and `unresolved`
    are legitimately populated diagnostics, and `min_lib_version` is a
    constant, not a measurement.
    """
    result = _drive_unmeasured(action, monkeypatch)
    diagnostic = {"reason", "status", "pins", "unresolved", "min_lib_version"}
    offending = {
        k: v
        for k, v in result.items()
        if k not in diagnostic
        and (v is False or (isinstance(v, (list, tuple, dict, set)) and not v))
    }
    assert not offending, (
        f"{action}: field(s) {offending} in an unmeasured payload read as a "
        f"clean verdict (False, or an empty collection). sf-pipeline-policy.md "
        f"§2.3a rule 2 — an unreached verdict propagates as UNKNOWN, never as "
        f"pass. Full payload: {result!r}"
    )


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_measured_payload_is_distinguishable_from_the_unmeasured_one(
    action, monkeypatch
):
    """A real measurement carries status=MEASURED, its verdict and its evidence.

    Without this, `status` could be satisfied by never emitting it on the
    measured path either, leaving the two shapes indistinguishable again.
    """
    result = _drive_measured(action, monkeypatch)
    assert result.get("status") == "MEASURED", (
        f"{action}: a measured run must say so; got {result!r}"
    )
    assert VERDICT_KEYS[action] in result, (
        f"{action}: a measured run must carry its verdict key; got {result!r}"
    )
    for key in EVIDENCE_KEYS[action]:
        assert key in result, (
            f"{action}: a measured run must carry its evidence list {key!r}; "
            f"got {result!r}"
        )


@pytest.mark.parametrize("action", sorted(VERDICT_KEYS))
def test_unmeasured_payload_keeps_the_fail_open_contract_the_sf_reads(
    action, monkeypatch
):
    """The fix must not convert a probe fault into a halt (§5 carve-out).

    The SF gates route an ABSENT verdict key to their `*GateDegraded` Pass —
    fail-open, visible, SNS-alerted. Emitting `status=UNKNOWN` must not add a
    truthy verdict that would send the run down the hard-fail branch instead.
    """
    result = _drive_unmeasured(action, monkeypatch)
    assert result.get(VERDICT_KEYS[action]) is not True, (
        f"{action}: an unmeasured gate must never assert a violation — that "
        f"would halt the weekly run on a probe fault; got {result!r}"
    )
    assert result.get("reason"), (
        f"{action}: `reason` is the key the deploy canary accepts on for the "
        f"degraded shape (tests/test_canary_call_sites.py DEGRADED_SHAPE_"
        f"ACTIONS); dropping it would freeze the live alias. Got {result!r}"
    )
