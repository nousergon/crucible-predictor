"""Unit tests for inference.lib_pin_drift — cross-repo lib-pin drift probe (L4517).

Exercises the pure compare logic (parity + floor) + the fail-open degraded mode.
``_fetch_repo_pin`` (the GitHub read) is mocked so tests are hermetic.
"""

from __future__ import annotations

from unittest.mock import patch

import inference.lib_pin_drift as lpd


# Today's aligned fleet: co-install pair matched, all >= floor (v0.39.0).
_ALIGNED = {
    "nousergon/crucible-backtester": "v0.53.0",
    "nousergon/crucible-predictor": "v0.53.0",
    "nousergon/nousergon-data": "v0.39.0",      # == floor → passes
    "nousergon/crucible-research": "v0.42.0",
}


def _patch_pins(mapping):
    """Patch _fetch_repo_pin to resolve from `mapping`.

    Values may be a pin string, `None` (shorthand for an unreachable fetch —
    the historical meaning, preserved so existing cases read unchanged), or a
    ready-made `PinRead` when a test cares WHICH miss occurred
    (alpha-engine-config-I7171).
    """

    def _resolve(repo, **_):
        value = mapping.get(repo)
        if isinstance(value, lpd.PinRead):
            return value
        if value is None:
            return lpd.PinRead(None, lpd.UNREACHABLE, "patched: no pin")
        return lpd.PinRead(value, None)

    return patch.object(lpd, "_fetch_repo_pin", side_effect=_resolve)


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_parse_pin_happy():
    line = (
        "alpha-engine-lib[arcticdb,flow_doctor,quant-xs] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.53.0"
    )
    assert lpd._parse_pin(line) == "v0.53.0"


def test_parse_pin_renamed_dist_nousergon_lib():
    # Dist renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
    # (config#1245). The drift probe must parse the new spelling so it keeps
    # working as the fleet crosses one repo at a time.
    line = (
        "nousergon-lib[arcticdb,flow-doctor,quant-xs] @ "
        "git+https://github.com/nousergon/nousergon-lib@v0.60.2"
    )
    assert lpd._parse_pin(line) == "v0.60.2"


def test_parse_pin_miss_returns_none():
    assert lpd._parse_pin("requests==2.31.0\nnumpy>=1.26") is None


def test_ge_floor():
    assert lpd._ge_floor("v0.39.0") is True       # == floor
    assert lpd._ge_floor("v0.53.0") is True
    assert lpd._ge_floor("v0.38.9") is False       # below floor
    assert lpd._ge_floor("v0.17.0") is False


# ── check_lib_pin_drift ──────────────────────────────────────────────────────

def test_aligned_fleet_no_drift():
    with _patch_pins(_ALIGNED):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is False
    assert out["parity_ok"] is True
    assert out["floor_ok"] is True
    assert out["reason"] == "in_sync"
    assert out["offenders"] == []


def test_co_install_parity_mismatch_halts():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = "v0.52.1"  # predictor lags backtester
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["parity_ok"] is False
    assert any("co-install parity" in o for o in out["offenders"])
    # the offending repos/versions are named
    assert any("v0.53.0" in o and "v0.52.1" in o for o in out["offenders"])


def test_below_floor_halts_and_names_offender():
    pins = dict(_ALIGNED)
    pins["nousergon/nousergon-data"] = "v0.38.0"  # regressed below floor
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["floor_ok"] is False
    assert any("below floor" in o and "nousergon-data" in o and "v0.38.0" in o
               for o in out["offenders"])


def test_fetch_failure_fails_open():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-research"] = None  # GitHub unreachable / parse miss
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    # The checker's own fragility must NEVER halt the weekly run — but must
    # also never report a definite verdict it never measured.
    # alpha-engine-config-I7048: has_drift is OMITTED, not False, so the
    # SF's IsPresent-guarded Choice routes to the visible degraded path.
    assert "has_drift" not in out
    assert out["reason"] == "fetch_failed"


def test_parity_mismatch_below_floor_combined():
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": "v0.49.0",   # parity break
        "nousergon/nousergon-data": "v0.30.0",        # below floor
        "nousergon/crucible-research": "v0.42.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["parity_ok"] is False and out["floor_ok"] is False
    assert len(out["offenders"]) >= 2


# ── I7171: a miss is named, not lumped into "fetch_failed" ───────────────────
#
# The 2026-08-13 incident: crucible-predictor has pinned nousergon-lib by
# commit SHA since crucible-predictor#422, which _LIB_PIN_RE does not match.
# The probe reported that as reason=fetch_failed — a permanent contract
# mismatch wearing a transient's name — on 59 of 68 measured invocations from
# 2026-07-31 onward, reading as intermittent GitHub flakiness the whole time.


_SHA_PIN_LINE = (
    "nousergon-lib[arcticdb,flow-doctor,quant-xs,quant-stats,contracts] @ "
    "git+https://github.com/nousergon/nousergon-lib"
    "@c907a044bb1553815225327bc56644050543b6f2"
)


def test_a_sha_pin_is_recognised_not_treated_as_a_parse_miss():
    # The real line from crucible-predictor/requirements.txt:70.
    assert lpd._parse_pin(_SHA_PIN_LINE) is None          # not a vX.Y.Z pin
    match = lpd._LIB_SHA_PIN_RE.search(_SHA_PIN_LINE)     # but IS recognised
    assert match is not None
    assert match.group(1) == "c907a044bb1553815225327bc56644050543b6f2"


def test_sha_pinned_repo_reports_sha_pinned_not_fetch_failed():
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": lpd.PinRead(
            None, lpd.SHA_PINNED, "c907a044bb1553815225327bc56644050543b6f2"
        ),
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert "has_drift" not in out          # still unmeasured, still fail-open
    assert out["reason"] == "sha_pinned"   # but the reason is now true
    assert out["unresolved"]["nousergon/crucible-predictor"]["problem"] == "sha_pinned"
    assert out["unresolved"]["nousergon/crucible-predictor"]["detail"].startswith("c907a04")


def test_a_permanent_problem_outranks_a_transient_one_in_the_reason():
    # A SHA pin and an unreachable repo at once. Reporting "fetch_failed"
    # would bury the condition that is still here tomorrow behind the one
    # that resolves itself.
    pins = {
        "nousergon/crucible-backtester": None,   # unreachable
        "nousergon/crucible-predictor": lpd.PinRead(None, lpd.SHA_PINNED, "c907a04"),
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["reason"] == "sha_pinned"
    # ...and the transient one is still recorded, just not the headline.
    assert out["unresolved"]["nousergon/crucible-backtester"]["problem"] == "unreachable"


def test_a_file_with_no_lib_pin_at_all_is_its_own_reason():
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": lpd.PinRead(None, lpd.UNRECOGNISED, None),
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["reason"] == "unrecognised_pin"


def test_a_genuine_outage_still_reads_as_fetch_failed():
    # The one transient of the three keeps its historical reason string, so
    # nothing downstream keying on it regresses.
    pins = {
        "nousergon/crucible-backtester": None,
        "nousergon/crucible-predictor": "v0.53.0",
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["reason"] == "fetch_failed"
    assert "has_drift" not in out
