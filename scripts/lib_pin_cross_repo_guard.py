#!/usr/bin/env python3
"""Lib-pin cross-repo parity guard — decision logic (alpha-engine-config-I10762).

MIRROR: this file must stay byte-identical to crucible-backtester's
`scripts/lib_pin_cross_repo_guard.py`, aside from the OWN/SIBLING constants
below and this header's repo names. Update both when changing behaviour.

Extracted out of `.github/workflows/lib-pin-cross-repo-guard.yml`'s inline
bash so the parity decision (equal pins, or the alpha-engine-config-I7934
lookaside for an in-flight lockstep partner) can be unit-tested without a
network call — see `tests/test_lib_pin_cross_repo_guard.py`, which exercises
both merge orders of a coordinated pin bump. The workflow still does all the
fetching (this PR's own pin, the sibling's main pin, and the sibling's open
PRs with their resolved pins); this script only decides pass/fail from
already-resolved inputs. Behaviour — including failing CLOSED when the
GitHub API is unreachable ("no answer, not no partner") — is unchanged from
the inline bash it replaces.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

# The other repo's name, as used in messages and the API/raw URLs the
# workflow fetches from. crucible-backtester's copy of this file swaps these.
OWN = "crucible-predictor"
SIBLING = "crucible-backtester"


@dataclass
class Verdict:
    ok: bool
    lines: list[str] = field(default_factory=list)


def evaluate(
    own_pin: str,
    sibling_main_pin: str,
    sibling_open_prs: Optional[list[dict]],
) -> Verdict:
    """own_pin / sibling_main_pin: version strings like 'v0.124.77'.

    sibling_open_prs: a list of {"number": int, "pin": str | None} — one
    entry per open PR on the sibling repo, "pin" resolved from that PR's
    head `requirements.txt` (None if it could not be parsed) — or None
    itself if the sibling's open-PR list could not be fetched at all
    (fail-closed, alpha-engine-config-I7934).
    """
    if own_pin == sibling_main_pin:
        return Verdict(True, [f"co-install parity OK: both pin {own_pin}"])

    if sibling_open_prs is None:
        return Verdict(
            False,
            [
                "::error::co-install parity break: this PR would pin "
                f"{OWN} to {own_pin} while {SIBLING}@main is pinned to "
                f"{sibling_main_pin}, AND the GitHub API could not be "
                "reached to check for an in-flight lockstep partner. "
                "Failing closed — this is 'no answer', not 'no partner'."
            ],
        )

    for pr in sibling_open_prs:
        if pr.get("pin") == own_pin:
            num = pr["number"]
            return Verdict(
                True,
                [
                    f"::notice::co-install parity DEFERRED: {SIBLING}@main "
                    f"is still {sibling_main_pin}, but {SIBLING}#{num} is "
                    f"open and already pins {own_pin}. This is the other "
                    "half of the lockstep arc. Merge both — the weekly "
                    "SF's LibPinDriftGate fails if the second half never "
                    "lands.",
                    "co-install parity OK (lockstep partner in flight): "
                    f"{SIBLING}#{num} pins {own_pin}",
                ],
            )

    return Verdict(
        False,
        [
            f"::error::co-install parity break: this PR would pin {OWN} to "
            f"{own_pin} while {SIBLING}@main is pinned to "
            f"{sibling_main_pin}, and NO open {SIBLING} PR pins {own_pin}. "
            "spot_backtest.sh co-installs both into one venv "
            f"(alpha-engine-config-I7835) — open the matching "
            f"{SIBLING} pin PR and this check will pass on re-run."
        ],
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-pin", required=True)
    parser.add_argument("--sibling-pin", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sibling-prs-json",
        help="JSON list of {number, pin} for the sibling's open PRs",
    )
    group.add_argument(
        "--sibling-api-unreachable",
        action="store_true",
        help="the sibling repo's open-PR list could not be fetched",
    )
    args = parser.parse_args(argv)

    sibling_open_prs: Optional[list[dict]]
    if args.sibling_api_unreachable:
        sibling_open_prs = None
    else:
        sibling_open_prs = json.loads(args.sibling_prs_json)

    verdict = evaluate(args.own_pin, args.sibling_pin, sibling_open_prs)
    for line in verdict.lines:
        print(line)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
