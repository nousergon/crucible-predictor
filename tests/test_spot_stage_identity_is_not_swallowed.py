"""A per-stage script's identity must survive sourcing ``_spot_common.sh``.

Each per-stage launcher sources ``_spot_common.sh`` and then declares who it
is::

    source "$SCRIPT_DIR/_spot_common.sh"
    _SSM_SLUG="${_SSM_SLUG:-spot-full-training}"

``_spot_common.sh`` used to assign the same parameters itself, with generic
values (``spot-training``, ``predictor-training``). ``${VAR:-default}`` expands
to ``default`` only when ``VAR`` is unset or empty — so by the time the stage
script ran, every one of its assignments was a no-op and it silently kept the
shared value. The comment above those lines even said "Must happen AFTER source
so they take effect", which is exactly backwards.

Measured 2026-08-11 on ``watch-rerun-2026-08-10-9``: the heartbeat emitted under
``spot-spot-training`` although ``spot_predictor_training.sh`` asks for
``spot-full-training``, and ``cleanup``'s log-pointer loop probed
``_ssm_logs/spot-training/<date>/`` — a prefix nothing writes — so it printed no
pointer at all while the real log sat under another slug. The same no-op gave
every model-zoo spot instance the ``predictor-training`` Name tag and published
``AlphaEngine/SpotInterruptionRetry`` under the wrong ``Process`` dimension.

This is the rename-blindness class: the identity a tool searches by is not the
identity the run recorded, and the resulting silence reads as "nothing
happened" rather than "you looked in the wrong place".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"

#: Every launcher that sources `_spot_common.sh`, with the identity it declares
#: for the default invocation (no flags). These are read from the scripts, not
#: hard-coded twice — the expected values here are the ASSERTION.
_STAGES = {
    "spot_predictor_training.sh": {
        "_SPOT_NAME": "predictor-training",
        "_SSM_SLUG": "spot-full-training",
        "_PROCESS_NAME": "predictor-training",
    },
    "spot_train_spec_dispatch.sh": {
        "_SPOT_NAME": "train-spec-dispatch",
        "_SSM_SLUG": "spot-model-zoo-spec",
        "_PROCESS_NAME": "predictor-model-zoo-spec",
    },
}

_IDENTITY_VARS = ("_SPOT_NAME", "_SSM_SLUG", "_PROCESS_NAME", "MAX_RUNTIME_SECONDS")

#: `_SPOT_NAME="${_SPOT_NAME:-value}"` at the start of a line.
_ASSIGN_RE = re.compile(
    r"^(?P<var>_SPOT_NAME|_SSM_SLUG|_PROCESS_NAME|MAX_RUNTIME_SECONDS)="
    r'"\$\{(?P=var):-(?P<value>[^}]*)\}"',
    re.MULTILINE,
)


def test_common_does_not_default_the_stage_identity() -> None:
    """`_spot_common.sh` must declare the identity empty, never populate it.

    A non-empty value here is invisible at the stage script's own assignment —
    it is the whole defect, and it reappears the moment someone "restores a
    sensible default".
    """
    source = _COMMON.read_text()
    for var in _IDENTITY_VARS:
        matches = _ASSIGN_RE.finditer(source)
        for match in matches:
            if match.group("var") != var:
                continue
            assert match.group("value") == "", (
                f"_spot_common.sh defaults {var} to "
                f"{match.group('value')!r}. Any non-empty value makes every "
                f"per-stage `${{{var}:-...}}` a no-op, and the stage silently "
                f"runs under the shared identity."
            )


@pytest.mark.parametrize("script_name", sorted(_STAGES))
def test_stage_identity_wins_over_the_shared_defaults(
    script_name: str, tmp_path: Path
) -> None:
    """Sourcing then declaring must yield the STAGE's identity."""
    script = _INFRA / script_name
    stage_lines = [
        match.group(0) for match in _ASSIGN_RE.finditer(script.read_text())
    ]
    assert stage_lines, f"{script_name}: declares no identity at all"

    harness = tmp_path / "identity.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        f"source {_COMMON}\n"
        + "\n".join(stage_lines)
        + "\n"
        + "".join(f'echo "{var}=${var}"\n' for var in _IDENTITY_VARS)
    )

    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    observed = dict(
        line.split("=", 1) for line in proc.stdout.strip().splitlines() if "=" in line
    )

    for var, expected in _STAGES[script_name].items():
        assert observed.get(var) == expected, (
            f"{script_name}: {var} resolved to {observed.get(var)!r}, expected "
            f"{expected!r}. The stage's own assignment was swallowed by a "
            f"default in _spot_common.sh."
        )
    assert observed.get("MAX_RUNTIME_SECONDS"), (
        f"{script_name}: MAX_RUNTIME_SECONDS is empty — spot_launch will refuse"
    )


def test_spot_launch_refuses_an_unset_identity(tmp_path: Path) -> None:
    """A stage that forgets to declare itself must fail BEFORE it is billable.

    Removing the defaults trades a silent wrong identity for an unset one. That
    is only an improvement if the unset case is loud, so the assertion lives in
    `spot_launch`, ahead of the instance request.
    """
    harness = tmp_path / "no_identity.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        f"source {_COMMON}\n"
        # Every identity parameter left as _spot_common.sh declared it: empty.
        "spot_launch\n"
    )
    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2, (
        "spot_launch should exit 2 on an unset stage identity, got "
        f"{proc.returncode}.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    for var in _IDENTITY_VARS:
        assert var in combined, f"the refusal does not name {var}"
    assert "Requesting spot instance" not in combined, (
        "spot_launch reached the instance request before refusing — the whole "
        "point of asserting here is that nothing is billable yet."
    )
