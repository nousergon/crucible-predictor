"""Every value the render call needs must be one the launcher guarantees.

The failure this pins (2026-08-11, `ne-weekly-freshness-pipeline` execution
`watch-rerun-2026-08-10-7`)::

    Using: Python 3.12.12
    fatal: repository '' does not exist

The bootstrap heredoc that produced this is gone (alpha-engine-config-I4992/
I6922 cutover): ``bootstrap_spot()`` now builds a ``krepis.spot_bootstrap
render`` CLI invocation and hands the RENDERED script to ``run_ssm``. The
env-closure class this test guards moved with it — from "does the exported
launcher-side prefix cover every ``${VAR}`` the heredoc body reads" to "does
every ``$VAR``/``${VAR}`` the render call's OWN arguments reference resolve to
something guaranteed non-empty before ``bootstrap_spot()`` runs". A value the
renderer needs that is unset is exactly as fatal as before — it just fails
inside ``python -m krepis.spot_bootstrap render`` (a required ``--repo-url ''``
argument) instead of inside a spot-side ``git clone``, and this test still
catches it statically, before either happens.

This was the THIRD defect in this bootstrap in one day, each revealed by fixing
the one ahead of it (#461 watchdog hang, #462 missing python3.12, this). A step
that has never run to completion hides its next failure behind its current one,
so the sequence is expected rather than surprising — and the useful response is
a check on the whole class rather than one more literal.
"""

from __future__ import annotations

import pytest

from tests._spot_bootstrap_render_support import (
    bootstrap_spot_block,
    render_args,
    render_invocation,
    resolve,
    spot_common_text,
)

_TEXT = spot_common_text()
_BLOCK = bootstrap_spot_block(_TEXT)
_INVOCATION = render_invocation(_BLOCK)
_ARGS = render_args(_INVOCATION)


def test_the_render_invocation_exists():
    """The premise of this whole check: bootstrap_spot() calls the renderer."""
    assert _ARGS, "no krepis.spot_bootstrap render arguments found"


@pytest.mark.parametrize("flag,raw", _ARGS)
def test_every_render_argument_resolves_to_a_guaranteed_value(flag, raw):
    """Every ``$VAR``/``${VAR}`` the render call passes must be resolvable.

    ``resolve()`` raises when a referenced variable has neither a static
    top-of-file default nor a declared runtime-only source (``_S3_STAGING``,
    set by ``spot_launch()`` before ``bootstrap_spot()`` can run) — the same
    failure mode that shipped ``REPO_URL`` as the empty string on
    2026-08-11, just caught here instead of on a live spot.
    """
    resolve(raw, _TEXT)  # raises AssertionError with the gap named, on failure


def test_repo_url_specifically_is_passed_and_resolves():
    """Anchored assertion on the argument that broke (crucible-predictor#463)."""
    repo_url_args = [raw for flag, raw in _ARGS if flag == "repo-url"]
    assert repo_url_args, "--repo-url is not passed to krepis.spot_bootstrap render"
    resolved = resolve(repo_url_args[0], _TEXT)
    assert resolved, "the --repo-url argument resolves to an empty string"
    assert resolved.startswith("https://") or resolved.startswith("git@"), (
        f"--repo-url resolved to {resolved!r}, which does not look like a repo "
        "URL — this is the crucible-predictor#463 class: a value that reaches "
        "the renderer as an empty or garbage string"
    )


def test_s3_staging_export_is_passed_and_resolves():
    """The staging export is load-bearing: the config-copy step reads it."""
    export_args = [raw for flag, raw in _ARGS if flag == "export"]
    staging = [raw for raw in export_args if raw.startswith("S3_STAGING=")]
    assert staging, "no --export S3_STAGING=... argument passed to the renderer"
    resolved = resolve(staging[0], _TEXT)
    assert resolved != "S3_STAGING=", "S3_STAGING resolves to an empty value"


def test_max_runtime_seconds_is_passed_and_resolves():
    """The spot-side hard-timeout timer must not be silently un-shipped.

    krepis 0.59.6 added --max-runtime-seconds (alpha-engine-config-I7372):
    bootstrap_spot() now passes it through so the hard-timeout timer this
    file's bootstrap previously had NO equivalent for at all is armed
    alongside the SSM-liveness watchdog — a separate guarantee, neither
    replaces the other. MAX_RUNTIME_SECONDS is declared empty at the top of
    _spot_common.sh on purpose (per-stage identity, never defaulted there),
    so its real value is only guaranteed by spot_launch()'s precondition
    check — exactly like _S3_STAGING — which is why both are declared
    runtime-only stand-ins rather than resolved off a static default.
    """
    runtime_args = [raw for flag, raw in _ARGS if flag == "max-runtime-seconds"]
    assert runtime_args, (
        "no --max-runtime-seconds argument passed to krepis.spot_bootstrap "
        "render — the spot-side hard-timeout timer is unshipped"
    )
    resolved = resolve(runtime_args[0], _TEXT)
    assert resolved and resolved.isdigit() and int(resolved) > 0, (
        f"--max-runtime-seconds resolved to {resolved!r}, not a positive "
        "integer — the renderer's transient-timer arm will fail loud (by "
        "design) but the value must be real before it ever reaches a spot"
    )
