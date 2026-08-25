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
# set of domain keys whose presence proves the invoke dispatched to that
# action's branch and returned its contract (SF Choice states consume these —
# see handler.py's `action ==` branches and trading_day_gate.py /
# drift_detector.py / pipeline_contract_check.py). Mirrors the contract
# already pinned by tests/test_canary_status_allowlist.py's
# test_accept_domain_key_present_promotes parametrization.
#
# A set, not a single key, because a gate action's contract may have more than
# one legitimate SHAPE. alpha-engine-config-I7048 gave check_lib_pin_drift and
# check_pipeline_contract a fail-open branch that OMITS the verdict key and
# emits `reason=fetch_failed` in its place, so an unmeasured gate cannot be
# read as measured-clean. This suite pinned only the measured shape and so
# passed on every commit while the deploy canary rejected the degraded one:
# three consecutive deploys published a version and were refused promotion,
# freezing the `live` alias at v455 while main advanced — including past the
# commit adding action=check_market_hours, which the market-hours gate (by then
# the first state of both trading pipelines) invokes. The 2026-08-13 preopen
# run hit a live Lambda with no such action, fell through to the default
# predict branch, and failed States.Runtime on the absent verdict. No orders
# were placed. See DEGRADED_SHAPE_ACTIONS below for the invariant that keeps a
# fail-open branch and its canary from drifting apart again.
SF_GATE_ACTION_EXPECT_KEYS = {
    "check_drift": {"status"},
    "check_trading_day": {"is_trading_day"},
    "check_market_hours": {"is_market_hours"},
    "check_weekly_run_day": {"is_weekly_run_day"},
    "check_pipeline_contract": {"has_violation", "reason"},
    "check_coverage": {"missing_count"},
    "check_lib_pin_drift": {"has_drift", "reason"},
}

# Gate actions with an I7048 fail-open branch: the verdict key is ABSENT and
# `reason` carries why. Both are asserted live in
# test_degraded_shape_actions_really_omit_their_verdict_key, so this map cannot
# quietly widen a canary for an action that has no such branch.
DEGRADED_SHAPE_ACTIONS = {
    "check_pipeline_contract": ("has_violation", "reason"),
    "check_lib_pin_drift": ("has_drift", "reason"),
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

    # `|` separates alternative acceptable keys (canary_accept accepts on any).
    tokens = expect.split("|")
    assert expect and all(_BARE_TOKEN_RE.match(t) for t in tokens), (
        f"<expect> for {action_label!r} must be a bare token, or several "
        f"separated by '|' (statusCode or domain keys), got {expect!r}"
    )

    payload_action = payload_obj.get("action") if isinstance(payload_obj, dict) else None
    if payload_action in SF_GATE_ACTION_EXPECT_KEYS:
        want = SF_GATE_ACTION_EXPECT_KEYS[payload_action]
        assert set(tokens) == want, (
            f"call site for SF-gate action {payload_action!r} declares "
            f"<expect>={expect!r}, but the handler's contract for that action "
            f"is the key set {sorted(want)!r} (not a statusCode) — a mismatch "
            f"is either the 2026-07-12 v351 false-canary-fail (PR #362: gating "
            f"a domain-dict response on statusCode always fails) or the "
            f"2026-08-13 promote freeze (asserting only the measured shape of "
            f"an I7048 two-shaped contract refuses promotion whenever the "
            f"probe's upstream fetch misses)."
        )


# ── Live deploy.sh: every current call site must pass ────────────────────────


def _live_call_sites() -> list[list[str]]:
    return _parse_call_sites(DEPLOY_SH.read_text())


def test_deploy_sh_has_the_expected_call_sites():
    # 8 inference (predict + 7 SF-gate) + 2 regime + 2 regime-eval. config#3025
    # dim8 added check_coverage + check_lib_pin_drift to the inference matrix
    # (previously excluded with no documented rationale), bringing inference
    # from 5 to 7 sites. alpha-engine-config-I7111 added check_market_hours —
    # the first state of both trading pipelines, so a broken contract there
    # stops a pipeline from starting rather than degrading one. A count drift
    # means a site was added or removed without updating this suite's
    # assumptions.
    assert len(_live_call_sites()) == 12


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
    assert seen_actions == set(SF_GATE_ACTION_EXPECT_KEYS), (
        "deploy.sh's SF-gate call sites and SF_GATE_ACTION_EXPECT_KEY have "
        "drifted apart — update the map alongside any new/renamed gate action"
    )


def test_handler_source_still_returns_the_mapped_keys():
    # Binds the map itself to the handler source, so a renamed domain key in
    # inference/handler.py (or the modules it dispatches to) fails this test
    # instead of silently invalidating the cross-check above.
    handler_src = HANDLER_PY.read_text()
    for action in SF_GATE_ACTION_EXPECT_KEYS:
        assert action in handler_src, (
            f"inference/handler.py no longer dispatches {action!r} — update "
            f"SF_GATE_ACTION_EXPECT_KEYS (config#2384)"
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


# ── The I7048 two-shaped contract, bound to real behavior ───────────────────


@pytest.mark.parametrize(
    ("action", "verdict_key", "degraded_key"),
    [(a, v, d) for a, (v, d) in DEGRADED_SHAPE_ACTIONS.items()],
)
def test_degraded_shape_actions_really_omit_their_verdict_key(
    action, verdict_key, degraded_key, monkeypatch
):
    """Drive each fail-open branch and assert the shape the canary must accept.

    This is the assertion whose absence let the 2026-08-13 promote freeze run
    for three deploys: `DEGRADED_SHAPE_ACTIONS` is only allowed to widen a
    canary for an action that demonstrably HAS a branch omitting its verdict
    key, and the widened key must be the one that branch actually emits.
    """
    if action == "check_lib_pin_drift":
        from inference import lib_pin_drift as mod

        # Every upstream pin unresolvable → the I7048 fail-open branch.
        # PinRead, not None, since alpha-engine-config-I7171 gave the read a
        # reason alongside the (absent) pin.
        monkeypatch.setattr(
            mod,
            "_fetch_repo_pin",
            lambda *a, **k: mod.PinRead(None, mod.UNREACHABLE, "patched"),
        )
        result = mod.check_lib_pin_drift()
    else:
        from inference import pipeline_contract_check as mod

        # Contract + registry unreadable → the I7048 fail-open branch.
        # (text, reason, last_modified), not None, since
        # alpha-engine-config-I7281 moved the read to S3 and gave it a reason.
        monkeypatch.setattr(
            mod, "_fetch_source", lambda *a, **k: (None, mod._REASON_MISSING, None)
        )
        result = mod.check_pipeline_contract()

    assert verdict_key not in result, (
        f"{action}: the fail-open branch must OMIT {verdict_key!r} (I7048 — an "
        f"unmeasured gate must not report a definite verdict). If this branch "
        f"now always emits it, remove {action!r} from DEGRADED_SHAPE_ACTIONS "
        f"and narrow its deploy.sh <expect> back to {verdict_key!r}."
    )
    assert degraded_key in result, (
        f"{action}: the fail-open branch must still emit {degraded_key!r} — it "
        f"is the only key the deploy canary can gate on when the verdict is "
        f"absent, and without it a degraded probe is indistinguishable from a "
        f"broken dispatch."
    )


def test_degraded_shape_actions_are_widened_at_their_call_site():
    """Every fail-open action's call site must accept the degraded key too."""
    for action, (verdict_key, degraded_key) in DEGRADED_SHAPE_ACTIONS.items():
        want = SF_GATE_ACTION_EXPECT_KEYS[action]
        assert want == {verdict_key, degraded_key}, (
            f"{action} has an I7048 fail-open branch, so its canary must "
            f"accept either {verdict_key!r} or {degraded_key!r}; "
            f"SF_GATE_ACTION_EXPECT_KEYS declares {sorted(want)!r}"
        )


# ── Regression: this suite must fail on the actual historical bad states ─────


def test_would_have_caught_the_20260813_promote_freeze():
    # main's state 2026-08-12..13: check_lib_pin_drift gated on `has_drift`
    # alone, which the I7048 fail-open branch does not emit — so every deploy
    # whose probe could not reach github.com published a version and was then
    # refused promotion, silently freezing the `live` alias.
    bad_deploy_sh = """
if ! run_canary_action "${LAMBDA_FUNCTION}" "${VERSION}" "check_lib_pin_drift" '{"action": "check_lib_pin_drift"}' "has_drift"; then
  CANARY_FAILED=1
fi
"""
    (site,) = _parse_call_sites(bad_deploy_sh)
    with pytest.raises(AssertionError, match="promote freeze"):
        _validate_call_site(site)


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


# ── I7954: a canary invocation must be marked synthetic ──────────────────────


def test_every_predictor_canary_payload_carries_dry_run():
    """`dry_run: true` on every `${LAMBDA_FUNCTION}` canary payload.

    alpha-engine-config-I7954. To this handler `dry_run` does not mean "writes
    nothing" — it means SYNTHETIC INVOCATION, and two behaviours hang off it:

      - `handler.py`'s `run_for_drift_gate(skip_deploy_drift=_dry_run)` /
        `_pf.run(skip_deploy_drift=_dry_run)`, so a canary does not assert its
        own freshly-built image SHA against live `main` HEAD — the merge-burst
        false-failure config#1073/#2731 removed.
      - `check_lib_pin_drift(probe=...)` / `check_pipeline_contract(probe=...)`,
        which drop a DETECTED condition from ERROR to WARNING. `handler.py`
        attaches flow-doctor at ERROR, so without this a probe finding from a
        synthetic invocation emails Brian as a production incident.

    Measured 2026-08-21T15:06:03Z: `check_lib_pin_drift` was the one site in
    this matrix with no `dry_run`, and the deploy canary following
    crucible-predictor#534 (14:59:41Z) paged a co-install parity break against
    crucible-backtester, whose matching lockstep bump merged at 15:13:07Z. The
    finding was true, useless, and self-cleared in 7 minutes: a cross-repo
    lockstep pin bump is two merges and can never be atomic, so ANY canary
    landing in that window reports a break.

    This asserts the whole matrix, not the one action that failed, because the
    next gate action added here would reintroduce the page otherwise. The
    regime / regime-eval Lambdas are deliberately excluded: they are separate
    functions with their own HTTP-shaped `dry_run`/`produce` actions.
    """
    offenders = []
    for args in _live_call_sites():
        func, _version, action_label, payload, _expect = args
        if func != "${LAMBDA_FUNCTION}":
            continue  # regime / regime-eval functions, own contract
        payload_obj = json.loads(payload)
        if payload_obj.get("dry_run") is not True:
            offenders.append(f"{action_label}: {payload}")
    assert not offenders, (
        "predictor canary payload(s) missing `\"dry_run\": true` — a canary "
        "invocation that is not marked synthetic asserts deploy-drift against "
        "live main and pages on any condition its probe detects "
        "(alpha-engine-config-I7954):\n  " + "\n  ".join(offenders)
    )


def test_probe_bearing_actions_thread_the_synthetic_marker_into_the_probe():
    """The handler must pass a synthetic-run `probe` to every probe-bearing action.

    A probe flag reaching the Lambda changes nothing about log severity unless
    the handler threads it down. `check_pipeline_contract` ALREADY carried
    `dry_run: true` before I7954 and still logged its violations at ERROR —
    which is why the fix was never "add dry_run" alone.

    alpha-engine-config-I8155: nor was `probe=dry_run` alone enough. This test
    pinned that expression while the OTHER half — `dry_run` present in the
    payload literal — was pinned nowhere, and `check_lib_pin_drift`'s canary
    payload never carried it. The guard passed for I7954's entire life on a
    call site where I7954's suppression did not apply. Both probe sites now
    read `invocation_kind`, which `run_canary_action` stamps centrally and no
    payload literal can drift out of; `dry_run` is kept as an OR so an
    explicit hand-invoked dry run still suppresses.
    """
    handler_src = HANDLER_PY.read_text()
    for call in ("check_lib_pin_drift(", "check_pipeline_contract("):
        index = handler_src.find(call)
        assert index != -1, f"inference/handler.py no longer calls `{call}`"
        window = handler_src[index : index + 200]
        assert "dry_run or is_synthetic_invocation(event)" in window, (
            f"`{call}` no longer threads the synthetic marker into its probe — "
            f"a probe invoked without it pages on every finding "
            f"(alpha-engine-config-I7954, I8155)"
        )


def test_the_canary_helper_stamps_the_synthetic_marker_on_every_payload():
    """The half I7954 left unpinned, pinned at the chokepoint this time.

    Asserting it on the helper rather than on each payload literal is the
    point: a per-site declaration is a per-site drift.
    """
    deploy_src = DEPLOY_SH.read_text()
    helper_start = deploy_src.index("run_canary_action() {")
    helper = deploy_src[helper_start : deploy_src.index("\n}", helper_start)]
    assert 'obj["invocation_kind"] = "canary"' in helper, (
        "run_canary_action no longer stamps invocation_kind — every canary "
        "then reaches the handlers indistinguishable from a real Step "
        "Functions invocation (alpha-engine-config-I8155)"
    )
