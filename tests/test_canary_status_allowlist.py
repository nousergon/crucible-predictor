"""Unit tests for the deploy.sh canary status-acceptance allowlist (config#853).

The post-deploy canaries in ``infrastructure/deploy.sh`` gate promotion of a
freshly-published Lambda version on the ``statusCode`` returned by a dry_run
invocation. The acceptance decision is a single Bash helper,
``canary_status_ok``, shared by all three canary sites (inference, regime
substrate, regime retrospective-eval).

These tests source that helper directly (no AWS, no Docker — the deploy body is
skipped via ``CANARY_LIB_SOURCE_ONLY=1``) and assert the producer-status
contract: the legitimate no-op statuses are ACCEPTED (so a future skip-path
cannot false-red a clean deploy) while genuine error codes are REJECTED (so a
broken image still rolls back). Mirrors nousergon-data #295's explicit
``status in ('OK','SKIPPED')`` allowlist.

A full canary requires a live Lambda invoke, which is unavailable in CI; this
suite covers the pure status-acceptance logic that the live promote path
depends on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

DEPLOY_SH = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"


def _source_and_run(call: str, *argv: str) -> bool:
    """Source deploy.sh (lib-only) and run ``<call>``; return accept/reject.

    Exit 0 (accept → promote) maps to True, any non-zero (reject → roll back)
    maps to False, matching the ``if ! <helper> ...`` use sites.
    """
    script = f'source "{DEPLOY_SH}"\n' f"if {call}; then exit 0; else exit 1; fi\n"
    proc = subprocess.run(
        ["bash", "-c", script, "_", *argv],
        env={"CANARY_LIB_SOURCE_ONLY": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    # Any exit code other than the helper's clean 0/1 means the source step
    # itself blew up — surface it rather than silently treating it as reject.
    assert proc.returncode in (0, 1), (
        f"unexpected exit {proc.returncode} for {call} argv={argv!r}: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return proc.returncode == 0


def _canary_status_ok(status: str) -> bool:
    """Source deploy.sh and run ``canary_status_ok <status>``; return its exit."""
    return _source_and_run('canary_status_ok "$1"', status)


def _canary_accept(expect: str, status: str, present_keys: str) -> bool:
    """Source deploy.sh and run ``canary_accept <expect> <status> <keys>``."""
    return _source_and_run('canary_accept "$1" "$2" "$3"', expect, status, present_keys)


# ── Allowlisted statuses (200 + legitimate no-ops) → accept / promote ────────

@pytest.mark.parametrize("status", ["200", "204", "skipped"])
def test_allowlisted_statuses_accepted(status):
    assert _canary_status_ok(status) is True


# ── Genuine error statuses → reject / roll back ──────────────────────────────

@pytest.mark.parametrize("status", ["400", "500", "502", "418"])
def test_error_statuses_rejected(status):
    # 400 = deprecated train route / unknown regime action; 5xx = real failure.
    assert _canary_status_ok(status) is False


# ── Degenerate parser outputs → reject (fail closed) ─────────────────────────

@pytest.mark.parametrize("status", ["", "0", "UNKNOWN", "PARSE_ERROR"])
def test_degenerate_status_rejected(status):
    # The payload parsers emit "" / "0" on a malformed or missing statusCode;
    # those must NOT be mistaken for a no-op success.
    assert _canary_status_ok(status) is False


# ── Near-miss success codes are NOT silently accepted ────────────────────────

@pytest.mark.parametrize("status", ["201", "200 ", " 200", "20", "2000"])
def test_near_miss_codes_rejected(status):
    # Only the exact enumerated strings are no-ops; the allowlist must not
    # widen to "any 2xx" or tolerate stray whitespace.
    assert _canary_status_ok(status) is False


# ── canary_accept: per-action acceptance contract (config#2346 / PR #362) ────
#
# The handler emits TWO response shapes. `predict` is HTTP-shaped and gates on
# statusCode; the Step-Function gate actions return raw domain dicts with NO
# statusCode and gate on the presence of their domain key. Gating a gate action
# on statusCode was the 2026-07-12 v351 false-canary-fail: a healthy image whose
# gate actions all returned valid domain dicts (statusCode absent → parsed as 0)
# was refused promotion.


# ── statusCode mode (predict) — unchanged HTTP-shaped semantics ──────────────

@pytest.mark.parametrize("status", ["200", "204", "skipped"])
def test_accept_statuscode_mode_allowlisted(status):
    assert _canary_accept("statusCode", status, "statusCode body") is True


@pytest.mark.parametrize("status", ["400", "500", "0", ""])
def test_accept_statuscode_mode_rejects_errors(status):
    assert _canary_accept("statusCode", status, "") is False


# ── domain-key mode (SF gate actions) — the regression this PR fixes ─────────

@pytest.mark.parametrize(
    "expect,keys",
    [
        # Each gate action's real return shape (statusCode deliberately absent).
        ("is_trading_day", "is_trading_day check_date day_name marker next_trading_day"),
        ("is_weekly_run_day", "is_weekly_run_day check_date day_name marker reason"),
        ("status", "date status severity alerts alert_details skipped_checks n_alerts"),
        ("has_violation", "has_violation reason boundary_count violations"),
    ],
)
def test_accept_domain_key_present_promotes(expect, keys):
    # A well-formed domain dict WITHOUT statusCode must be accepted — this is
    # exactly the response that v351 falsely rolled back. statusCode is "0"
    # (the parser's default for a missing key); it must be ignored in this mode.
    assert _canary_accept(expect, "0", keys) is True


@pytest.mark.parametrize(
    "expect",
    ["is_trading_day", "is_weekly_run_day", "status", "has_violation"],
)
def test_accept_domain_key_absent_rolls_back(expect):
    # Missing the declared key (wrong branch / malformed response / empty
    # payload) must still roll back — key-presence is the wiring invariant.
    assert _canary_accept(expect, "0", "some other unrelated keys") is False
    assert _canary_accept(expect, "200", "") is False  # even a 200 can't rescue it


def test_domain_mode_ignores_statuscode_allowlist():
    # A gate action must NOT be accepted merely because statusCode happens to be
    # allowlisted, nor rejected because it is an error code — the domain key is
    # the sole criterion in this mode.
    assert _canary_accept("has_violation", "500", "has_violation reason") is True
    assert _canary_accept("has_violation", "200", "reason boundary_count") is False


# ── run_canary_action must not hard-require the <expect> arg (set -u guard) ───
#
# deploy.sh runs under `set -euo pipefail`. When run_canary_action gained the
# 5th <expect> parameter, the HTTP-shaped regime canary call sites still passed
# only 4 args, so `local expect="$5"` tripped `unbound variable` and crashed the
# deploy AFTER the (fixed) inference canary had already promoted the alias
# (2026-07-12 regime-canary crash). The parameter must default so an
# HTTP-shaped caller may omit it and no 4-arg caller can ever `set -u` fault.
def test_run_canary_action_defaults_expect_arg():
    body = DEPLOY_SH.read_text()
    assert 'local expect="${5:-statusCode}"' in body, (
        "run_canary_action must default <expect> to statusCode so a 4-arg "
        "call site cannot trip `set -u` — see the 2026-07-12 regime-canary crash"
    )
