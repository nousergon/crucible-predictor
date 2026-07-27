"""
Tests for PredictorPreflight.run() mode composition.

BasePreflight primitives — including ``check_deploy_drift`` and its
``_fetch_origin_main_sha`` helper — are tested in alpha-engine-lib.
These tests verify that ``run()`` composes the expected primitive
calls in the expected order, that ``run_for_drift_gate()`` is a strict
subset for the SF gate, and that the predictor passes its own repo
slug to the inherited drift check.

Data-freshness assertions (universe + macro/SPY + inference-macro symbols)
moved upstream to ``alpha-engine-data``'s preflight 2026-05-05; the data
step in every Step Function hard-fails on staleness before any consumer
runs, so re-checking here is redundant.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nousergon_lib.preflight as _base_preflight_mod
from inference.preflight import PredictorPreflight


def _make() -> PredictorPreflight:
    return PredictorPreflight(bucket="test-bucket")


def test_run_composes_full_check_sequence():
    """Verify the full primitive call sequence: env + S3 + drift + weights."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key:
        pf.run()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once_with("nousergon/crucible-predictor")
    s3_key.assert_called_once()
    assert s3_key.call_args.args[0] == "predictor/weights/meta/meta_model.pkl"


def test_run_does_not_call_data_freshness_primitives():
    """Data-freshness checks moved upstream to alpha-engine-data preflight.
    The inference Lambda must not re-check macro/universe freshness — the
    SF data step already gated on it."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift"), \
         patch.object(pf, "check_s3_key"), \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        pf.run()

    fresh.assert_not_called()


def test_run_skips_drift_when_skip_deploy_drift_true():
    """dry_run=true canary path: env + S3 + weights still run, but the
    image-SHA-vs-origin/main drift assertion is skipped. A canary writes
    and emails nothing, so drift-vs-main is the wrong invariant and a
    false-failure source during merge bursts (config#1073)."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key:
        pf.run(skip_deploy_drift=True)

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_not_called()
    s3_key.assert_called_once()


def test_run_checks_drift_by_default():
    """Production (dry_run=false) default still drift-checks — protection
    for real runs is unchanged."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key"):
        pf.run(skip_deploy_drift=False)

    drift.assert_called_once_with("nousergon/crucible-predictor")


def test_run_for_drift_gate_is_strict_subset():
    """The drift gate runs ONLY env + S3 + image-SHA. No model weights,
    no macro reads — those are predict-path concerns and running them on
    the gate caused the 2026-05-01 SF timeout cascade."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key, \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        pf.run_for_drift_gate()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once_with("nousergon/crucible-predictor")
    s3_key.assert_not_called()
    fresh.assert_not_called()


def test_run_for_drift_gate_checks_drift_by_default():
    """Default (no kwarg) must still drift-check — this is the path used by
    the dedicated check_deploy_drift SF-gate action AND the SF's own
    (non-canary) pre-spend check_lib_pin_drift/check_pipeline_contract
    invocations. Neither may silently lose drift protection (config#2731)."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key"):
        pf.run_for_drift_gate()

    drift.assert_called_once_with("nousergon/crucible-predictor")


def test_run_for_drift_gate_skips_drift_when_skip_deploy_drift_true():
    """dry_run=true deploy-time canary path (config#2731): env + S3 still
    run, but the image-SHA-vs-origin/main drift assertion is skipped. A
    canary writes/emails nothing, so drift-vs-main is the wrong invariant
    and a false-failure source during merge bursts — same rationale as
    config#1073's run(skip_deploy_drift=...), now mirrored here for the
    check_pipeline_contract / check_lib_pin_drift deploy canaries."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift:
        pf.run_for_drift_gate(skip_deploy_drift=True)

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_not_called()


# ── End-to-end SHA-mismatch through the real check_deploy_drift primitive ────
# (config#2731) These do NOT mock ``check_deploy_drift`` itself — they mock
# only its two I/O boundaries (the baked-SHA file read + the GitHub HEAD
# fetch), exercising the REAL comparison logic. This is the concrete
# regression scenario from the issue: a deploy-canary invocation whose image
# SHA differs from main HEAD (the exact merge-burst shape) must PASS when
# skip_deploy_drift=True and RAISE when it is not.


def _patch_shas(baked_sha: str, upstream_sha: str):
    """Patch only check_deploy_drift's two I/O boundaries + stub out the
    unrelated env/S3 preflight primitives, so run_for_drift_gate() exercises
    its REAL drift-comparison logic hermetically (no AWS creds needed)."""
    return patch.multiple(
        _base_preflight_mod,
        _read_baked_git_sha=lambda *_a, **_kw: baked_sha,
        _fetch_origin_main_sha=lambda *_a, **_kw: upstream_sha,
    )


def test_drift_gate_dry_run_canary_passes_despite_sha_mismatch():
    """The core config#2731 regression: a deploy-time check_pipeline_contract
    canary (dry_run=true → skip_deploy_drift=True) against an image whose
    baked SHA differs from main HEAD (merge-burst shape) must NOT raise —
    the drift gate is skipped for this dry_run path only."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), patch.object(pf, "check_s3_bucket"), \
         _patch_shas("90608f6f5716" + "0" * 28, "bed589c34913" + "1" * 28):
        pf.run_for_drift_gate(skip_deploy_drift=True)  # must not raise


def test_drift_gate_check_deploy_drift_action_raises_on_same_sha_mismatch():
    """Hard invariant: the dedicated check_deploy_drift ACTION (no skip
    kwarg — its default) must still raise RuntimeError on the exact same
    SHA-mismatch condition the canary path above was made to tolerate."""
    import pytest

    pf = _make()
    with patch.object(pf, "check_env_vars"), patch.object(pf, "check_s3_bucket"), \
         _patch_shas("90608f6f5716" + "0" * 28, "bed589c34913" + "1" * 28):
        with pytest.raises(RuntimeError, match="Deploy drift"):
            pf.run_for_drift_gate()  # default skip_deploy_drift=False


def test_drift_gate_non_dry_run_sf_gate_raises_on_sha_mismatch():
    """Hard invariant: the SF's own pre-spend (non-canary, no dry_run)
    invocation of check_lib_pin_drift/check_pipeline_contract must retain
    its drift belt-and-suspenders — explicit skip_deploy_drift=False still
    raises on the same mismatch."""
    import pytest

    pf = _make()
    with patch.object(pf, "check_env_vars"), patch.object(pf, "check_s3_bucket"), \
         _patch_shas("90608f6f5716" + "0" * 28, "bed589c34913" + "1" * 28):
        with pytest.raises(RuntimeError, match="Deploy drift"):
            pf.run_for_drift_gate(skip_deploy_drift=False)


def test_drift_gate_matching_sha_never_raises_regardless_of_skip():
    """Sanity: when SHAs match (no drift), neither mode raises."""
    pf = _make()
    same_sha = "abc123def456" + "0" * 28
    with patch.object(pf, "check_env_vars"), patch.object(pf, "check_s3_bucket"), \
         _patch_shas(same_sha, same_sha):
        pf.run_for_drift_gate(skip_deploy_drift=False)
        pf.run_for_drift_gate(skip_deploy_drift=True)


def test_first_failure_short_circuits():
    """If env vars fail, no S3 / ArcticDB / GitHub calls happen."""
    import pytest

    pf = _make()
    with patch.object(
        pf, "check_env_vars", side_effect=RuntimeError("AWS_REGION missing")
    ), \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        with pytest.raises(RuntimeError, match="AWS_REGION"):
            pf.run()

    s3.assert_not_called()
    drift.assert_not_called()
    fresh.assert_not_called()
