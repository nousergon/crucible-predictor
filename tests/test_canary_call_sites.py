"""Static validation of every `run_canary_action` call site (config#2384).

Two canary-wiring defects reached `main` on 2026-07-12 and only surfaced on
the post-merge `Deploy` workflow (real ECR build + Lambda invoke), never in
PR CI, because PR CI's `test_canary_status_allowlist.py` only unit-tests the
pure `canary_accept`/`canary_status_ok` *helpers* — it never inspects the
call *sites* in `infrastructure/deploy.sh`:

1. PR #362 gated four Step-Function GATE actions (`check_drift`,
   `check_trading_day`, `check_weekly_run_day`, `check_pipeline_contract`) on
   `<expect>="statusCode"` — an HTTP-shaped invariant those handlers never
   emit (they return raw domain dicts). False FAILED; refused to promote a
   healthy image.
2. PR #366's fix (a new 5th `<expect>` arg) then broke the 4 HTTP-shaped
   regime/regime-eval call sites, which still passed only 4 args — `local
   expect="$5"` tripped `set -u`'s unbound-variable check and crashed the
   deploy mid-promotion.

Both were mechanically detectable by reading `deploy.sh` text — no live AWS
needed. This suite parses every `run_canary_action` call site statically and
asserts arity, payload well-formedness, and (for the SF-gate actions) that
the declared `<expect>` token matches the domain key the handler source
actually returns for that action — so a renamed/typo'd contract fails PR CI
instead of the post-merge Deploy.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SH = REPO_ROOT / "infrastructure" / "deploy.sh"
HANDLER_PY = REPO_ROOT / "inference" / "handler.py"

# run_canary_action <function_name> <version> <action_label> <payload> <expect>
CALL_SITE_RE = re.compile(r"run_canary_action\s+(.+?);\s*then")

# The Step-Function gate actions dispatched by inference/handler.py, and the
# domain key their handler returns (SF Choice states consume these — see
# handler.py's `action ==` branches and trading_day_gate.py /
# drift_detector.py / pipeline_contract_check.py). Mirrors the contract
# already pinned by tests/test_canary_status_allowlist.py's
# test_accept_domain_key_present_promotes parametrization.
SF_GATE_ACTION_EXPECT_KEY = {
    "check_drift": "status",
    "check_trading_day": "is_trading_day",
    "check_weekly_run_day": "is_weekly_run_day",
    "check_pipeline_contract": "has_violation",
}

_BARE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_call_sites(deploy_sh_text: str) -> list[list[str]]:
    """Extract each `run_canary_action` call site as a list of raw args.

    Uses ``shlex`` so single- and double-quoted args (incl. the JSON payload,
    which itself contains double quotes) split the same way bash would,
    without expanding `${VAR}` references — we only need arg count/content,
    not their runtime values.
    """
    sites = []
    for match in CALL_SITE_RE.finditer(deploy_sh_text):
        sites.append(shlex.split(match.group(1)))
    return sites


def _validate_call_site(args: list[str]) -> None:
    """Raise AssertionError describing the first contract violation found."""
    assert len(args) == 5, (
        f"run_canary_action must be called with exactly 5 args "
        f"(func, version, action label, payload, expect); got {len(args)}: {args!r}"
    )
    _func, _version, action_label, payload, expect = args

    try:
        payload_obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"payload arg for {action_label!r} is not valid JSON: {payload!r} ({exc})"
        ) from None

    assert expect and _BARE_TOKEN_RE.match(expect), (
        f"<expect> for {action_label!r} must be a bare token (statusCode or a "
        f"domain key), got {expect!r}"
    )

    payload_action = payload_obj.get("action") if isinstance(payload_obj, dict) else None
    if payload_action in SF_GATE_ACTION_EXPECT_KEY:
        want = SF_GATE_ACTION_EXPECT_KEY[payload_action]
        assert expect == want, (
            f"call site for SF-gate action {payload_action!r} declares "
            f"<expect>={expect!r}, but the handler returns its result under "
            f"the {want!r} key (not a statusCode) — this is exactly the "
            f"2026-07-12 v351 false-canary-fail (PR #362): gating a "
            f"domain-dict response on statusCode always fails."
        )


# ── Live deploy.sh: every current call site must pass ────────────────────────


def _live_call_sites() -> list[list[str]]:
    return _parse_call_sites(DEPLOY_SH.read_text())


def test_deploy_sh_has_the_expected_call_sites():
    # 5 inference (predict + 4 SF-gate) + 2 regime + 2 regime-eval, per the
    # issue's own "Approach" anchor. A count drift means a site was added or
    # removed without updating this suite's assumptions.
    assert len(_live_call_sites()) == 9


@pytest.mark.parametrize("args", _live_call_sites(), ids=lambda a: a[2])
def test_live_call_site_is_well_formed(args):
    _validate_call_site(args)


def test_sf_gate_actions_are_all_covered_by_the_contract_map():
    # Every SF-gate action label present in deploy.sh must have a known
    # expected-key mapping above — an action added to the handler without a
    # corresponding entry here would silently skip the cross-check.
    seen_actions = set()
    for args in _live_call_sites():
        payload_obj = json.loads(args[3])
        action = payload_obj.get("action") if isinstance(payload_obj, dict) else None
        if action in ("dry_run", "produce", None):
            continue  # HTTP-shaped / non-gate actions, not in the map
        seen_actions.add(action)
    assert seen_actions == set(SF_GATE_ACTION_EXPECT_KEY), (
        "deploy.sh's SF-gate call sites and SF_GATE_ACTION_EXPECT_KEY have "
        "drifted apart — update the map alongside any new/renamed gate action"
    )


def test_handler_source_still_returns_the_mapped_keys():
    # Binds the map itself to the handler source, so a renamed domain key in
    # inference/handler.py (or the modules it dispatches to) fails this test
    # instead of silently invalidating the cross-check above.
    handler_src = HANDLER_PY.read_text()
    for action, key in SF_GATE_ACTION_EXPECT_KEY.items():
        assert action in handler_src, (
            f"inference/handler.py no longer dispatches {action!r} — update "
            f"SF_GATE_ACTION_EXPECT_KEY (config#2384)"
        )
    # trading_day_gate / drift_detector / pipeline_contract_check are the
    # modules handler.py imports for these actions (see handler.py's
    # `from ... import ...` lines inside each `action ==` branch); grep their
    # source for the literal key so a rename there is caught too.
    key_to_module = {
        "is_trading_day": REPO_ROOT / "inference" / "trading_day_gate.py",
        "is_weekly_run_day": REPO_ROOT / "inference" / "trading_day_gate.py",
        "status": REPO_ROOT / "monitoring" / "drift_detector.py",
        "has_violation": REPO_ROOT / "inference" / "pipeline_contract_check.py",
    }
    for key, module_path in key_to_module.items():
        src = module_path.read_text()
        assert f'"{key}"' in src or f"'{key}'" in src, (
            f"expected domain key {key!r} not found (as a dict key literal) "
            f"in {module_path.relative_to(REPO_ROOT)} — the handler's real "
            f"return contract may have drifted from SF_GATE_ACTION_EXPECT_KEY"
        )


# ── Regression: this suite must fail on the actual historical bad states ─────


def test_would_have_caught_pr362_statuscode_mismatch():
    # PR #362's state: all 4 SF-gate actions wired with <expect>="statusCode"
    # instead of their real domain key.
    bad_deploy_sh = """
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_drift" '{"action": "check_drift"}' "statusCode"; then
  CANARY_FAILED=1
fi
"""
    (site,) = _parse_call_sites(bad_deploy_sh)
    with pytest.raises(AssertionError, match="false-canary-fail"):
        _validate_call_site(site)


def test_would_have_caught_pr366_arity_regression():
    # PR #366-merge state: the HTTP-shaped regime call sites still passed
    # only 4 args (no <expect>), which is exactly what tripped `set -u` once
    # run_canary_action gained its 5th positional parameter.
    bad_deploy_sh = """
if ! run_canary_action "${REGIME_LAMBDA_FUNCTION}" "${REGIME_VERSION}" "dry_run" '{"action": "dry_run"}'; then
  CANARY_FAILED=1
fi
"""
    (site,) = _parse_call_sites(bad_deploy_sh)
    with pytest.raises(AssertionError, match="exactly 5 args"):
        _validate_call_site(site)


def test_rejects_malformed_json_payload():
    bad_deploy_sh = """
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_drift" '{action: check_drift}' "status"; then
  CANARY_FAILED=1
fi
"""
    (site,) = _parse_call_sites(bad_deploy_sh)
    with pytest.raises(AssertionError, match="not valid JSON"):
        _validate_call_site(site)


def test_rejects_malformed_expect_token():
    bad_deploy_sh = """
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_drift" '{"action": "check_drift"}' "not a token"; then
  CANARY_FAILED=1
fi
"""
    (site,) = _parse_call_sites(bad_deploy_sh)
    with pytest.raises(AssertionError, match="bare token"):
        _validate_call_site(site)
