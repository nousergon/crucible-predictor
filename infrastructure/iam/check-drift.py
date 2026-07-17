#!/usr/bin/env python3
"""check-drift.py — Diff codified IAM inline policies against live AWS state.

Mirrors the SOTA drift-check pattern from `crucible-executor/infrastructure/iam`
(alpha-engine-config#2340), adapted to THIS repo's flat, one-inline-policy-per-role
convention that `apply.sh` already establishes:

    infrastructure/iam/<role-name>.json      # a single inline policy document
        role name:   <role-name>             # the filename minus ".json"
        policy name: derive_policy_name()     # "<role minus -role>-policy"

For every codified `<role>.json` the check compares, across two axes:

  1. Presence — the derived inline-policy name is attached to the live role
     (`aws iam list-role-policies`). A codified policy missing on AWS means an
     un-applied change (run apply.sh); an inline policy on AWS with no codified
     source means an out-of-band manual edit.
  2. Content — for the policy present on both sides, the codified document vs
     the live document (`aws iam get-role-policy`), compared after canonical
     normalization (sorted keys, no incidental whitespace) so cosmetic-only
     formatting differences don't trip the check.

Any drift exits non-zero. A genuine AWS CLI/auth failure exits 2 (distinct from
"drift found" = 1), so an OIDC/permission problem is never silently read as
"clean".

Usage:
  ./infrastructure/iam/check-drift.py            # check every codified role
  ./infrastructure/iam/check-drift.py --role R   # check one role

Requires AWS creds with iam:ListRolePolicies + iam:GetRolePolicy on the codified
roles. Locally: any admin profile. In CI: the OIDC role
github-actions-iam-drift-check (read-only, scoped to the codified roles only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def derive_policy_name(role: str) -> str:
    """Inline-policy name for a role — must match apply.sh's derivation.

    Strips a trailing "-role" suffix (if present) and appends "-policy", so the
    convention matches the historical inline-policy names already attached to
    the alpha-engine roles (e.g. alpha-engine-predictor-role ->
    alpha-engine-predictor-policy).
    """
    return f"{role[:-len('-role')] if role.endswith('-role') else role}-policy"


def _aws_iam(*args: str) -> dict | list | str:
    """Call `aws iam ...` and return parsed JSON. Exit 2 on a CLI/auth failure."""
    result = subprocess.run(
        ["aws", "iam", *args, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"AWS CLI failed: aws iam {' '.join(args)}\n"
            f"stderr: {result.stderr}\n"
        )
        sys.exit(2)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _canonical_json(doc: dict) -> str:
    """Canonical JSON for byte-stable comparison: sorted keys, no extra ws."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def _check_role(role_file: Path) -> list[str]:
    """Return drift findings for one `<role>.json`. Empty list means clean."""
    role_name = role_file.stem
    policy_name = derive_policy_name(role_name)
    findings: list[str] = []

    try:
        source_doc = json.loads(role_file.read_text())
    except json.JSONDecodeError as exc:
        return [f"{role_name}: source JSON invalid ({exc})"]

    aws_resp = _aws_iam("list-role-policies", "--role-name", role_name)
    aws_policies = set(aws_resp.get("PolicyNames", []))

    if policy_name not in aws_policies:
        findings.append(
            f"{role_name}/{policy_name}: codified in source but not on AWS role "
            f"(run apply.sh to push)"
        )
    for extra in sorted(aws_policies - {policy_name}):
        findings.append(
            f"{role_name}/{extra}: present on AWS role but not codified "
            f"(add a JSON file or delete from AWS)"
        )

    if policy_name in aws_policies:
        aws_resp = _aws_iam(
            "get-role-policy",
            "--role-name", role_name,
            "--policy-name", policy_name,
        )
        aws_doc = aws_resp.get("PolicyDocument", {})
        if _canonical_json(source_doc) != _canonical_json(aws_doc):
            findings.append(
                f"{role_name}/{policy_name}: source document differs from "
                f"AWS document (content drift — run apply.sh or reconcile)"
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", help="Check one role (default: every codified role)"
    )
    args = parser.parse_args()

    if args.role:
        role_files = [SCRIPT_DIR / f"{args.role}.json"]
        if not role_files[0].is_file():
            sys.stderr.write(f"ERROR: {role_files[0]} not found\n")
            return 2
    else:
        role_files = sorted(SCRIPT_DIR.glob("*.json"))

    if not role_files:
        print(f"No codified role policies (*.json) under {SCRIPT_DIR} — "
              "nothing to check.")
        return 0

    total_findings: list[str] = []
    for role_file in role_files:
        total_findings.extend(_check_role(role_file))

    if total_findings:
        print(f"IAM drift detected ({len(total_findings)} finding(s)):")
        for f in total_findings:
            print(f"  - {f}")
        return 1

    role_names = ", ".join(f.stem for f in role_files)
    print(f"OK: no IAM drift for {role_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
