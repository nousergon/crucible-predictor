"""Every per-stage spot launcher makes a DELIBERATE predictor.yaml decision.

alpha-engine-config-I7216 (crucible-backtester mirror). The 2026-08-13 live
failure was crucible-backtester's PredictorBacktest hard-failing on a missing
predictor.yaml because the file's only producer was a THREE-`cp`-deep side
effect of a sibling SF state (PredictorTraining), invisible exactly on the
mechanical-rerun/skip-set path that recovery exists for. PR657 fixed it there
by moving the guarantee into `spot_common_resolve_predictor_config()` — the
layer that DECLARES the requirement — rather than adding a fourth `cp` to the
SF's command array.

**Why this repo carries the identical class, unfixed until now.**
`spot_predictor_training.sh`, `spot_model_zoo_select.sh` and
`spot_train_spec_dispatch.sh` all called plain `check_config_exists()` against
this repo's own `config/predictor.yaml` — a file gitignored (the `.example`
pattern) that only exists on the dispatch box because the weekly SF's command
array (`sudo -u ec2-user cp --remove-destination
~/alpha-engine-config/experiments/reference/predictor/predictor.yaml
~/alpha-engine-predictor/config/predictor.yaml`) put it there, run by
PredictorTraining or ModelZooSelect immediately before each script's own SSM
command.

Audited against the live `nousergon-data/infrastructure/step_function.json`
2026-08-13: every path reaching `spot_train_spec_dispatch.sh` or
`spot_model_zoo_select.sh --select-only` passes through PredictorTraining
first — `skip_predictor_training`'s validated-skip terminal
(`PredictorTrainingSkipped`) ends the WHOLE branch before
`ResolveZooSpecs`/`ModelZooTrainMap` is ever reached, so this was not observed
failing live. But that guarantee lived entirely in SF TOPOLOGY, asserted
nowhere in the script — exactly the shape that let the backtester-side
instance go undetected until a mechanical rerun's skip-set diverged from the
graph's assumption. A future skip flag scoped to only the model-zoo fan-out,
or a Map reordering, would reintroduce the identical defect with the identical
blast radius (this stage writes `predictor/research_free_backfill/`, the live
entry feed).

`resolve_or_stage_predictor_config()` in `_spot_common.sh` closes this the same
way PR657 did: stage from the config repo's reference copy before declaring
the file missing, at the layer that needs it.

This suite pins that EVERY per-stage launcher makes an explicit, reviewable
choice about this precondition, so a new stage cannot silently inherit the
gap by omission — the exact way this one did.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INFRA = REPO_ROOT / "infrastructure"

STAGING_CALL = "resolve_or_stage_predictor_config"
BARE_CHECK_CALL = "check_config_exists"

# Launchers that read config/predictor.yaml and must resolve it through the
# self-staging chokepoint rather than a bare existence check.
_REQUIRES_SELF_STAGING = {
    "spot_predictor_training.sh",
    "spot_model_zoo_select.sh",
    "spot_train_spec_dispatch.sh",
}

# Launchers deliberately left off the self-staging path. Listed explicitly so
# the choice is recorded and reviewable — moving one here is a decision
# someone made, not a line nobody wrote.
_DELIBERATELY_NOT_SELF_STAGING = {
    # Retired monolith (config-I4442/I4497, cutover 2026-08-09) — kept
    # unchanged as the rollback path. It carries its own historical config
    # resolution and is out of scope for this per-stage-launcher hardening.
    "spot_train.sh",
}


def _stage_launchers() -> list[Path]:
    return sorted(p for p in INFRA.glob("spot_*.sh") if p.name != "_spot_common.sh")


def test_every_launcher_is_classified():
    """A new stage script cannot silently inherit the bare-check gap."""
    found = {p.name for p in _stage_launchers()}
    classified = _REQUIRES_SELF_STAGING | _DELIBERATELY_NOT_SELF_STAGING
    unclassified = found - classified
    assert not unclassified, (
        f"spot launcher(s) {sorted(unclassified)} are in neither "
        f"_REQUIRES_SELF_STAGING nor _DELIBERATELY_NOT_SELF_STAGING. Decide "
        f"which, and record it — inheriting the bare check_config_exists() "
        f"path by omission is how spot_train_spec_dispatch.sh and "
        f"spot_model_zoo_select.sh carried this gap (alpha-engine-config-I7216)."
    )
    stale = classified - found
    assert not stale, f"classified launcher(s) {sorted(stale)} no longer exist"


@pytest.mark.parametrize("name", sorted(_REQUIRES_SELF_STAGING))
def test_predictor_config_launchers_use_the_self_staging_resolver(name):
    text = (INFRA / name).read_text()
    assert STAGING_CALL in text, (
        f"{name} must resolve config/predictor.yaml via {STAGING_CALL}(), not "
        f"a bare {BARE_CHECK_CALL}() — the bare check assumes an EARLIER SF "
        f"state's cp already staged the file, a guarantee this script cannot "
        f"see and cannot enforce (alpha-engine-config-I7216)."
    )
    assert f'{BARE_CHECK_CALL} "$(' not in text, (
        f"{name} still calls the bare {BARE_CHECK_CALL}() directly against "
        f"config/predictor.yaml instead of routing through {STAGING_CALL}()."
    )


def test_the_resolver_is_defined_exactly_once_in_spot_common():
    text = (INFRA / "_spot_common.sh").read_text()
    assert text.count(f"{STAGING_CALL}() {{") == 1, (
        f"{STAGING_CALL}() must be defined exactly once in _spot_common.sh — "
        "a second definition is a fork of the invariant, not a second copy of it."
    )


def test_the_resolver_stages_from_the_reference_config_repo_path():
    """Pins the SOURCE path, so a future edit cannot quietly change it.

    Matches crucible-backtester's spot_common_resolve_predictor_config source
    exactly (alpha-engine-config-I7216): the two repos must agree on where the
    canonical predictor.yaml lives, or one of them will stage the wrong file.
    """
    text = (INFRA / "_spot_common.sh").read_text()
    assert (
        '$HOME/alpha-engine-config/experiments/${experiment_id}/predictor/predictor.yaml'
        in text
    ), (
        "resolve_or_stage_predictor_config must stage from "
        "$HOME/alpha-engine-config/experiments/${experiment_id}/predictor/"
        "predictor.yaml — the same reference path crucible-backtester's "
        "spot_common_resolve_predictor_config uses."
    )


def test_the_resolver_uses_portable_rm_then_cp_not_gnu_remove_destination():
    """The GNU-only flag is why this class of bug shipped untested locally.

    Scoped to actual `cp` INVOCATIONS, never a substring search of the whole
    file: this module's own docstrings and comments explain the SF's existing
    `cp --remove-destination` steps in prose, and a naive `not in` would
    report that explanation as the defect (the exact class of checker bug
    named in crucible-predictor's own test_spot_bootstrap_invariants.py,
    test_watchdog_unit_is_never_oneshot).
    """
    text = (INFRA / "_spot_common.sh").read_text()
    cp_invocations = [
        line for line in text.splitlines()
        if line.strip().startswith("cp ") or " && cp " in line or "&& cp " in line
    ]
    offenders = [line for line in cp_invocations if "--remove-destination" in line]
    assert not offenders, (
        "resolve_or_stage_predictor_config must use `rm -f` then plain `cp` "
        "(GNU `cp --remove-destination` breaks BSD cp / local testability), "
        f"found: {offenders}"
    )
