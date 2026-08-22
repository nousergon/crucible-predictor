"""The deploy must strip denied environment variables, and must do it where
the promotion can carry the change to traffic.

alpha-engine-config-I7925. `alpha-engine-predictor-inference`'s environment is
live-only state — no repo, no IaC and no script ever wrote it. It holds a STALE
COPY of `/alpha-engine/GITHUB_TOKEN`: measured 2026-08-21, the SSM parameter's
own value authenticates (`GET /user` -> 200) while the environment's copy is
rejected with a 401. Same name, different value, nothing detecting the drift. A
first-party dependency read that copy out of site-packages and the 401 halted
the 2026-08-21 preopen trading pipeline 3.4 seconds after start
(alpha-engine-config-I7924).

`infrastructure/deploy.sh` now converges the environment against a deny-list.
These tests pin the two properties that make that convergence real, because
both fail SILENTLY: a removal placed after `publish-version` never reaches the
published version, and a removal that promotes the alias itself would race the
deploy's own promotion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOY = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"
_CODE = _DEPLOY.read_text(encoding="utf-8")


def test_deploy_script_exists() -> None:
    assert _DEPLOY.is_file(), f"{_DEPLOY} is missing"


def test_github_token_is_on_the_deny_list() -> None:
    """The variable whose stale copy caused I7924 must be named, not implied."""
    assert "LAMBDA_ENV_DENIED_KEYS=(" in _CODE, (
        "the deploy no longer declares a denied-key set — a variable set by "
        "hand now outlives every deploy again (alpha-engine-config-I7925)"
    )
    declaration = _CODE.split("LAMBDA_ENV_DENIED_KEYS=(", 1)[1].split(")", 1)[0]
    assert "GITHUB_TOKEN" in declaration


def test_removal_uses_the_shared_cli_not_a_bare_aws_call() -> None:
    """`aws lambda update-function-configuration --environment` REPLACES the
    whole variable map, deleting every operator-set flag codified nowhere.
    The read-modify-write chokepoint is `krepis.aws remove-lambda-env`."""
    assert "krepis.aws remove-lambda-env" in _CODE


def test_removal_runs_before_the_version_is_published() -> None:
    """A removal after `publish-version` mutates $LATEST only. The published
    version — and therefore the `live` alias — would keep the variable, and the
    deploy would report success having changed nothing that serves traffic."""
    remove_at = _CODE.index("remove-lambda-env")
    publish_at = _CODE.index("aws lambda publish-version")
    assert remove_at < publish_at, (
        "the environment convergence must precede publish-version, or the "
        "published version keeps the denied variable (L4497)"
    )


def test_removal_defers_promotion_to_the_deploy() -> None:
    """The deploy publishes a version and moves the `live` alias itself. A
    removal that also promoted would publish a second version mid-deploy and
    race the alias move."""
    step = _CODE.split("krepis.aws remove-lambda-env", 1)[1].split("\n\n", 1)[0]
    assert "--defer-publish" in step
    assert "--promote-alias" not in step


def test_removal_is_idempotent_across_deploys() -> None:
    """Every deploy after the first finds the key already gone; without
    --missing-ok the CLI refuses and `set -euo pipefail` aborts the deploy."""
    step = _CODE.split("krepis.aws remove-lambda-env", 1)[1].split("\n\n", 1)[0]
    assert "--missing-ok" in step


@pytest.mark.parametrize("req", ["requirements.txt", "requirements-lambda.txt"])
def test_krepis_pin_can_supply_the_subcommand(req: str) -> None:
    """`remove-lambda-env` ships in krepis 0.59.23. An older pin makes the
    deploy step exit 2 on an unknown subcommand."""
    line = next(
        ln
        for ln in (Path(__file__).resolve().parents[1] / req)
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.startswith("krepis[")
    )
    version = line.split("==", 1)[1].split()[0].strip()
    parts = tuple(int(p) for p in version.split("."))
    assert parts >= (0, 59, 23), (
        f"{req} pins krepis {version}; remove-lambda-env needs >= 0.59.23"
    )


def test_krepis_pin_does_not_need_the_deploy_role_to_list_aliases() -> None:
    """Deploy run 32509752554 (2026-08-21 17:48 UTC) failed here:

        ERROR: could not list aliases of alpha-engine-predictor-inference:
        AccessDeniedException ... github-actions-lambda-deploy is not
        authorized to perform: lambda:ListAliases

    `remove_lambda_environment_keys` enumerated aliases even under
    `--defer-publish`, which already asserts the caller publishes and promotes
    itself. The image was pushed and `$LATEST` updated, then this raised, so
    `publish-version` and the alias move never ran — a PARTIAL deploy, with
    the `live` alias serving a stale image while main had moved on. That is
    the SHA drift the preopen `DeployDriftGate` halts on.

    krepis 0.59.24 skips the enumeration under `defer_publish` (krepis#176).
    An older pin reintroduces the partial deploy, so the floor is pinned here
    rather than left to memory.
    """
    from pathlib import Path as _P

    for req in ("requirements.txt", "requirements-lambda.txt"):
        line = next(
            ln
            for ln in (_P(__file__).resolve().parents[1] / req)
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.startswith("krepis[")
        )
        version = line.split("==", 1)[1].split()[0].strip()
        parts = tuple(int(x) for x in version.split("."))
        assert parts >= (0, 59, 24), (
            f"{req} pins krepis {version}; --defer-publish needs >= 0.59.24 "
            f"or the deploy fails on lambda:ListAliases and leaves a PARTIAL "
            f"deploy (alpha-engine-config-I7925)"
        )
