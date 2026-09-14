"""Guard: no IAM policy document or apply script under `infrastructure/`.

alpha-engine-config-I8143: this repo used to carry its own copy of
`alpha-engine-predictor-role`'s IAM policy
(`infrastructure/iam/alpha-engine-predictor-role.json`) plus an `apply.sh`
that could push it live. Nothing wired the copy to anything — it drifted
from the live role, and on 2026-09-12 it was applied to production AWS by
hand, overwriting the policy actually owned by the private `nous-ergon-ops`
repo and holding that repo's IAM drift check red for two days.

IAM for this service is codified and applied from `nous-ergon-ops` only
(see `infrastructure/iam/README.md`, a pointer, not a copy). This test
fails the moment either shape reappears anywhere under `infrastructure/`
in this repo, regardless of filename, so the class doesn't need to be
rediscovered file-by-file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INFRA = REPO_ROOT / "infrastructure"

PUT_ROLE_POLICY_RE = re.compile(r"\bput-role-policy\b")


def _iter_json_files():
    if not INFRA.exists():
        return
    yield from INFRA.rglob("*.json")


def _iter_shell_files():
    if not INFRA.exists():
        return
    yield from INFRA.rglob("*.sh")


def test_no_iam_policy_document_json_under_infrastructure():
    offenders = []
    for path in _iter_json_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and (
            "Statement" in data or "PolicyDocument" in data
        ):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "An IAM policy document reappeared under infrastructure/: "
        f"{offenders}. IAM for alpha-engine-predictor-role is owned by the "
        "private nous-ergon-ops repo (alpha-engine-config-I8143) — do not "
        "codify a policy document here, even as a snapshot. See "
        "infrastructure/iam/README.md."
    )


def test_no_apply_script_calls_put_role_policy_under_infrastructure():
    offenders = []
    for path in _iter_shell_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if PUT_ROLE_POLICY_RE.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "A script under infrastructure/ calls `put-role-policy`: "
        f"{offenders}. This repo must never apply IAM directly — that is "
        "the exact failure mode alpha-engine-config-I8143 fixed (a stray "
        "apply.sh overwrote the owned nous-ergon-ops policy on live AWS "
        "on 2026-09-12). See infrastructure/iam/README.md."
    )
