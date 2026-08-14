"""Cross-repo `alpha-engine-lib` pin-drift probe (L4517).

Preventive companion to the reactive `spot_backtest.sh` co-install guard:
asserts the cross-repo lib-pin invariant BEFORE the Saturday Step Function
spends on any spot launch, so a drift fails the run in seconds (with a named
offender) instead of breaking a co-install ~12 min into a spot.

The invariant (Brian, L4517):
  (a) **co-install parity** — `backtester pin == predictor pin`. `spot_backtest.sh`
      installs both into ONE venv (backtester first, predictor second); a lagging
      predictor silently downgrades the lib and breaks `quant.stats` (2026-05-12).
      This is the only multi-repo co-install site.
  (b) **floor** — every Saturday-SF repo's pin >= `MIN_LIB_VERSION`. The floor sits
      at the LOWEST INTENTIONAL pin (data v0.39.0) — data/research lag on purpose
      and do NOT co-install, so strict fleet lockstep is explicitly NOT required;
      the floor only catches a regression BELOW today's fleet, not currency.

Exposed as `action=check_lib_pin_drift` on the predictor Lambda handler (mirrors
`check_deploy_drift`) so the SF invokes it as an early Task → Choice gate. Returns
a JSON-serializable dict the SF Choice consumes via `has_drift`.

Fail-open on the checker's own fragility (mirrors `deploy_drift.py` degraded mode):
a GitHub-fetch or parse failure for any needed repo → `has_drift` is OMITTED from
the result (not set to `false`) + WARN, so the probe never false-halts the
weekly run. It only halts on a CONFIRMED drift (all needed pins fetched + parsed
+ a parity mismatch or below-floor pin).

alpha-engine-config-I7048 (2026-08-12): previously this branch returned
`has_drift=False` — a definite, present-key "no drift" verdict — indistinguishable
downstream from a real pass. The SF's `LibPinDriftGate` Choice (nousergon-data
infrastructure/step_function.json) already has a correctly-designed
IsPresent-guarded branch for exactly this case — an ABSENT `has_drift` routes to
`LibPinGateDegraded` (visible, SNS-alerted, non-blocking) rather than the
silent-pass Default. Omitting the key here reuses machinery that already existed
and was already correct. Do NOT default to `has_drift=True` on a fetch failure —
inverting the lie does not make it true.

alpha-engine-config-I7301 (2026-08-13) draws the one line that rule does NOT
cover. A fetch failure means the pin is unknown. A pin that was READ and is
permanently uncomparable — `sha_pinned`, `unrecognised_pin` — on a member of
the CO-INSTALL PAIR is a different thing: invariant (a) is structurally
unverifiable, on this run and on every future run, until a human edits that
requirements.txt. That is a measured defect and it HALTS. Floor-only repos
(data/research) keep the fail-open, because the floor is the advisory half.
Reporting the permanent case as UNKNOWN is what let a real parity break
(backtester v0.124.5 vs a predictor SHA pin ~v0.124.16) run unnoticed from
2026-07-31 to 2026-08-13 behind a degraded alert that fired on every run and
could not be cleared by waiting.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from typing import NamedTuple

from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

# Floor = data's current (lowest INTENTIONAL) pin. Raising this above any
# intentional laggard (data v0.39.0 / research v0.42.0) would false-halt the
# Saturday SF. Bump deliberately only when the whole Saturday-SF fleet clears it.
MIN_LIB_VERSION = "0.39.0"

# The co-install pair that MUST match (spot_backtest.sh installs both into one
# venv). Order is (first-installed, second-installed) for the message only.
_CO_INSTALL_PAIR = ("nousergon/crucible-backtester", "nousergon/crucible-predictor")

# Every repo that participates in the Saturday SF must clear the floor.
_FLOOR_REPOS = (
    "nousergon/nousergon-data",
    "nousergon/crucible-predictor",
    "nousergon/crucible-backtester",
    "nousergon/crucible-research",
)

# Lifted from {predictor,backtester}/tests/test_lib_pin_lockstep.py (the pin
# format is `alpha-engine-lib[extras] @ git+https://.../alpha-engine-lib@vX.Y.Z`).
# TODO(L4517 follow-up, P3): lift this + `_fetch_repo_pin` into
# `alpha_engine_lib.preflight` alongside `_fetch_origin_main_sha` — the regex is
# already duplicated across repo test suites (chokepoint rule). Deferred: needs
# a lib release + cross-repo re-pin.
# Distribution name renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
# (config#1245 / #1172). Accept either spelling so the cross-repo drift probe
# keeps parsing pins as the fleet crosses one repo at a time.
_LIB_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/"
    r"nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)

# The SAME pin line, ending in a raw commit SHA instead of a version tag. This
# is a legitimate, in-use pin form — `crucible-predictor` has pinned this way
# since crucible-predictor#422 — and it is NOT comparable to MIN_LIB_VERSION or
# to another repo's tag without resolving the commit to a release.
#
# It exists as its own pattern so the probe can say WHICH problem it hit.
# Before alpha-engine-config-I7171 a SHA pin fell through `_LIB_PIN_RE` and was
# reported identically to an unreachable GitHub — `reason=fetch_failed`. That
# is a permanent condition wearing a transient's name, and it held on every
# invocation from 2026-07-31 onward (59 of 68 measured) while reading as
# intermittent flakiness. It is also what turned alpha-engine-config-I7048's
# correct omit-the-verdict-key change into a three-deploy promote freeze, which
# took out the 2026-08-13 preopen run.
_LIB_SHA_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/"
    r"nousergon/nousergon-lib@([0-9a-f]{7,40})\b"
)

_RAW_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/{repo}/{branch}/requirements.txt"
)

# Why a pin could not be resolved. `None` means it was.
# Measurement status (alpha-engine-config-I7277, sf-pipeline-policy.md §2.3a
# rule 2), shared vocabulary with pipeline_contract_check.py. UNKNOWN is not a
# verdict: that payload omits BOTH `has_drift` and `offenders`.
STATUS_UNKNOWN = "UNKNOWN"
STATUS_MEASURED = "MEASURED"

UNREACHABLE = "unreachable"        # the fetch itself failed — genuinely transient
SHA_PINNED = "sha_pinned"          # fetched and recognised; not comparable to a floor
UNRECOGNISED = "unrecognised_pin"  # fetched, and nothing in it looks like a lib pin


class PinRead(NamedTuple):
    """One repo's pin, or a named reason it is not available.

    Tri-state deliberately: a caller that collapses "could not reach GitHub"
    and "read the file and did not understand it" into one bucket cannot tell
    an outage from a contract change, and will wait out the wrong one.
    """

    pin: str | None
    problem: str | None
    detail: str | None = None


def _parse_pin(text: str) -> str | None:
    """Extract the `vX.Y.Z` lib pin from a requirements.txt body, or None."""
    match = _LIB_PIN_RE.search(text)
    return match.group(1) if match else None


def _fetch_repo_pin(repo: str, branch: str = "main", timeout: float = 5.0) -> PinRead:
    """Fetch `repo@branch`'s pinned lib version from GitHub.

    Reads the raw `requirements.txt` (public, no auth). Returns a `PinRead`
    naming which of the three outcomes occurred; every non-pin outcome is
    still fail-open at the caller, but it is fail-open with the reason
    attached (alpha-engine-config-I7171).
    """
    url = _RAW_REQUIREMENTS_URL.format(repo=repo, branch=branch)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except OSError as exc:  # URLError/HTTPError + bare read-phase TimeoutError
        log.warning("Lib-pin drift: requirements.txt unreachable for %s (%s)", repo, exc)
        return PinRead(None, UNREACHABLE, str(exc))

    pin = _parse_pin(text)
    if pin is not None:
        return PinRead(pin, None)

    sha_match = _LIB_SHA_PIN_RE.search(text)
    if sha_match is not None:
        sha = sha_match.group(1)
        log.warning(
            "Lib-pin drift: %s pins nousergon-lib by commit SHA %s, not a vX.Y.Z "
            "tag — the pin was READ successfully but cannot be compared to the "
            "%s floor or to another repo's tag (alpha-engine-config-I7171)",
            repo, sha[:12], MIN_LIB_VERSION,
        )
        return PinRead(None, SHA_PINNED, sha)

    log.warning(
        "Lib-pin drift: no nousergon-lib pin found in %s/requirements.txt "
        "(file fetched successfully — this is a contract mismatch, not an outage)",
        repo,
    )
    return PinRead(None, UNRECOGNISED, None)


def _ge_floor(pin: str) -> bool:
    """True if `pin` (vX.Y.Z) >= MIN_LIB_VERSION. Unparseable → False (caller
    only reaches this with a regex-matched pin, so this is belt-and-suspenders)."""
    try:
        return Version(pin.lstrip("v")) >= Version(MIN_LIB_VERSION)
    except InvalidVersion:
        return False


def check_lib_pin_drift(branch: str = "main") -> dict:
    """Assert the cross-repo lib-pin invariant; return a dict the SF Choice reads.

    `has_drift` is present (`true`/`false`) ONLY when every needed pin was
    fetched + parsed. Any fetch/parse miss OMITS `has_drift` entirely
    (fail-open, alpha-engine-config-I7048: unmeasured is not the same as
    measured-clean). `has_drift=true` only when a parity mismatch or
    below-floor pin is confirmed. Shape mirrors `check_deploy_drift`.

    alpha-engine-config-I7171: `reason` now NAMES which miss occurred rather
    than reporting every one of them as `fetch_failed`. `unresolved` carries
    the per-repo detail. The distinction is operational, not cosmetic — a
    transient outage is waited out, a SHA pin or a contract change never
    resolves itself, and reporting the second as the first is why this gate
    sat unmeasured from 2026-07-31 to 2026-08-13 while looking like GitHub
    flakiness. Reasons, most-actionable first:

      ``sha_pinned``     a repo pins the lib by commit SHA, which is a real
                         pin the probe cannot compare to a version floor
      ``unrecognised_pin`` the file was read and carries no lib pin at all
      ``fetch_failed``   GitHub was genuinely unreachable — the only
                         transient of the three, and the only one worth a
                         retry

    alpha-engine-config-I7277: the unresolved branch also carries an explicit
    ``status=UNKNOWN`` and OMITS ``offenders``. It previously emitted
    ``offenders: []`` — the same "could not measure rendered as a clean check"
    defect I7277 fixed on ``check_pipeline_contract``'s ``violations: []``.
    The measured branch carries ``status=MEASURED``.
    """
    repos = tuple(dict.fromkeys(_CO_INSTALL_PAIR + _FLOOR_REPOS))  # de-dup, ordered
    reads: dict[str, PinRead] = {r: _fetch_repo_pin(r, branch=branch) for r in repos}
    pins: dict[str, str | None] = {r: read.pin for r, read in reads.items()}

    # Fail-open: if any needed pin is unknown, do NOT halt the weekly run.
    # has_drift is OMITTED (not set False) — see module docstring I7048 note.
    unresolved = {
        r: {"problem": read.problem, "detail": read.detail}
        for r, read in reads.items()
        if read.pin is None
    }

    # alpha-engine-config-I7301: a PERMANENT problem on a CO-INSTALL-PAIR member
    # is a measured defect, not a failed measurement, and it halts.
    #
    # This is NOT the `has_drift=True`-on-fetch-failure inversion the module
    # docstring forbids (I7048), and the difference is the whole point. A fetch
    # failure means we do not know what the repo pins. `sha_pinned` and
    # `unrecognised_pin` mean we READ the repo's requirements.txt successfully
    # and it carries a pin that cannot be compared to the other half of the
    # co-install pair — permanently, on every future invocation, until a human
    # edits that file. Parity on that pair is invariant (a) — the one this gate
    # exists to protect, and the shape of the 2026-05-12 silent-downgrade
    # incident. A repo state that makes it structurally unverifiable IS the
    # finding.
    #
    # Reporting it as UNKNOWN instead is what let the real drift measured on
    # 2026-08-13 (backtester v0.124.5 vs a predictor SHA pin ~v0.124.16) sit
    # unnoticed from 2026-07-31 onward behind a fail-open that fired every run
    # and could never be cleared by waiting.
    #
    # Deliberately scoped to the pair: `nousergon-data` and `crucible-research`
    # are floor-only repos that lag ON PURPOSE and do not co-install, so an
    # uncomparable pin there weakens an advisory check, not a load-bearing one —
    # it stays fail-open below.
    unverifiable_pair = {
        r: d
        for r, d in unresolved.items()
        if r in _CO_INSTALL_PAIR and d["problem"] in (SHA_PINNED, UNRECOGNISED)
    }
    if unverifiable_pair:
        bt, pred = _CO_INSTALL_PAIR
        offenders = [
            f"co-install parity UNVERIFIABLE: {r} pins nousergon-lib by "
            f"{d['problem']} ({d['detail']}), which is not comparable to the "
            f"other half of the co-install pair "
            f"({bt if r == pred else pred}={pins[bt if r == pred else pred]})"
            for r, d in unverifiable_pair.items()
        ]
        log.error(
            "Lib-pin drift HALT: co-install parity cannot be verified — %s",
            "; ".join(offenders),
        )
        return {
            "status": STATUS_MEASURED,
            "has_drift": True,
            # Still None, not False: we did not measure a MISMATCH, we measured
            # that the comparison cannot be made. Asserting False here would be
            # the same fabrication this change exists to remove.
            "parity_ok": None,
            "floor_ok": None,
            "min_lib_version": MIN_LIB_VERSION,
            "pins": pins,
            "offenders": offenders,
            "unresolved": unresolved,
            "reason": "co_install_pin_unverifiable",
        }

    if unresolved:
        problems = {v["problem"] for v in unresolved.values()}
        # Report the most actionable problem present. A permanent condition
        # outranks a transient one: if any repo is SHA-pinned or carries no
        # recognisable pin, saying "fetch_failed" because some OTHER repo also
        # happened to be unreachable would bury the finding that will still be
        # here tomorrow.
        for candidate in (SHA_PINNED, UNRECOGNISED, UNREACHABLE):
            if candidate in problems:
                reason = "fetch_failed" if candidate is UNREACHABLE else candidate
                break
        log.warning(
            "Lib-pin drift: %d repo pin(s) unresolved (%s) — proceeding "
            "(fail-open). Detail: %s",
            len(unresolved), reason, unresolved,
        )
        # alpha-engine-config-I7277: `offenders: []` used to sit here beside
        # parity_ok/floor_ok=None — the same defect I7277 fixed on the
        # pipeline-contract probe. `offenders` is the evidence list a consumer
        # reads, and an empty one reads as "checked, nobody offended". Omitted
        # entirely now, with an explicit status=UNKNOWN in its place, so the
        # payload has no field that can be mistaken for a clean verdict.
        return {
            "status": STATUS_UNKNOWN,
            "parity_ok": None,
            "floor_ok": None,
            "min_lib_version": MIN_LIB_VERSION,
            "pins": pins,
            "unresolved": unresolved,
            "reason": reason,
        }

    bt, pred = _CO_INSTALL_PAIR
    parity_ok = pins[bt] == pins[pred]

    below_floor = [r for r in _FLOOR_REPOS if not _ge_floor(pins[r])]
    floor_ok = not below_floor

    offenders: list[str] = []
    if not parity_ok:
        offenders.append(
            f"co-install parity: {bt}={pins[bt]} != {pred}={pins[pred]}"
        )
    offenders.extend(
        f"below floor: {r}={pins[r]} < {MIN_LIB_VERSION}" for r in below_floor
    )

    has_drift = not (parity_ok and floor_ok)
    if has_drift:
        log.error("Lib-pin drift DETECTED: %s", "; ".join(offenders))

    return {
        "status": STATUS_MEASURED,
        "has_drift": has_drift,
        "parity_ok": parity_ok,
        "floor_ok": floor_ok,
        "min_lib_version": MIN_LIB_VERSION,
        "pins": pins,
        "offenders": offenders,
        "reason": "drift_detected" if has_drift else "in_sync",
    }
