"""Pins ``spot_train.sh``'s default INSTANCE_TYPES priority — closes the
2026-05-24 OOM forensic finding from ROADMAP (f).

The 2026-05-24 11:02 PT Saturday SF recovery's ``BranchBFailed`` was
root-caused via the SSM log at
``s3://alpha-engine-research/_ssm_logs/predictor-training/2026-05-24/
ip-172-31-73-124.ec2.internal-181932Z.log``: the kernel OOM-killed PID
26675 (``$PY - <<PYEOF`` invocation at SSM script line 56) during the
volatility-with-macros training stage (``train=1503471 val=322172
n_features=19``). The c5.large default (4 GB RAM) ran out of headroom;
``set -eo pipefail`` propagated SIGKILL/137; SSM returned non-Success;
SF cascaded to BranchBFailed.

The fix is small: reorder the capacity-resilient ``INSTANCE_TYPES``
fallback so ``m5.large`` (8 GB) is the first preference. The lib
``ec2_spot`` rotator only escalates on capacity errors — never on
OOM-class failures — so the priority order *is* the memory-budget
guard. ``m5.large`` at ~$0.10/hr is ~$0.015 more than ``c5.large``
per launch; a few cents per Saturday SF buys the headroom.

These tests pin the order so a future edit can't silently revert to
c5.large-first and re-introduce the OOM class.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_train.sh"


def _instance_types_default() -> str:
    """Return the right-hand side of ``INSTANCE_TYPES="${INSTANCE_TYPES:-...}"`` ."""
    text = _SCRIPT.read_text()
    match = re.search(
        r'^INSTANCE_TYPES="\$\{INSTANCE_TYPES:-([^}]+)\}"',
        text,
        re.MULTILINE,
    )
    assert match, "INSTANCE_TYPES default assignment not found — spot_train.sh structure changed"
    return match.group(1)


def test_default_first_is_m5_large():
    default = _instance_types_default()
    first = default.split(",")[0].strip()
    assert first == "m5.large", (
        f"first INSTANCE_TYPES entry must be m5.large (8 GB) — c5.large "
        f"(4 GB) OOM'd on 2026-05-24 Saturday SF recovery. Got: {first!r}"
    )


def test_default_includes_capacity_resilient_set():
    default = _instance_types_default()
    types = {t.strip() for t in default.split(",")}
    # All 4 types: 2 vCPU classes that can rotate capacity in default-VPC
    # subnets across us-east-1{a..f}. Lift / removal needs an explicit
    # ROADMAP entry.
    assert types == {"m5.large", "c5.large", "c6i.large", "c5a.large"}, (
        f"INSTANCE_TYPES fallback set drifted from the canonical 4-type "
        f"capacity-resilient set: {sorted(types)}"
    )


def test_oom_root_cause_comment_present():
    """Memory-budget rationale must accompany the priority order so
    a future editor doesn't revert to c5.large-first without seeing
    the prior incident.
    """
    text = _SCRIPT.read_text()
    assert "OOM" in text, (
        "the OOM rationale comment must remain attached to the "
        "INSTANCE_TYPES assignment — closes ROADMAP (f) forensic dive"
    )
    assert "ROADMAP (f)" in text, (
        "the spot_train.sh INSTANCE_TYPES comment must reference "
        "ROADMAP (f) so the audit trail is greppable"
    )
