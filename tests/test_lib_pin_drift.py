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
    # A FLOOR-ONLY repo (research) is SHA-pinned. The floor is the advisory
    # half of the invariant, so this stays fail-open — the case this test was
    # written for. The co-install-pair equivalent halts instead: see the
    # I7301 block below.
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": "v0.53.0",
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": lpd.PinRead(
            None, lpd.SHA_PINNED, "c907a044bb1553815225327bc56644050543b6f2"
        ),
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert "has_drift" not in out          # still unmeasured, still fail-open
    assert out["reason"] == "sha_pinned"   # but the reason is now true
    assert out["unresolved"]["nousergon/crucible-research"]["problem"] == "sha_pinned"
    assert out["unresolved"]["nousergon/crucible-research"]["detail"].startswith("c907a04")


def test_a_permanent_problem_outranks_a_transient_one_in_the_reason():
    # A SHA pin and an unreachable repo at once. Reporting "fetch_failed"
    # would bury the condition that is still here tomorrow behind the one
    # that resolves itself. Both on non-pair repos, so the fail-open path is
    # the one under test.
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": "v0.53.0",
        "nousergon/nousergon-data": None,        # unreachable
        "nousergon/crucible-research": lpd.PinRead(None, lpd.SHA_PINNED, "c907a04"),
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["reason"] == "sha_pinned"
    # ...and the transient one is still recorded, just not the headline.
    assert out["unresolved"]["nousergon/nousergon-data"]["problem"] == "unreachable"


def test_a_file_with_no_lib_pin_at_all_is_its_own_reason():
    pins = {
        "nousergon/crucible-backtester": "v0.53.0",
        "nousergon/crucible-predictor": "v0.53.0",
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": lpd.PinRead(None, lpd.UNRECOGNISED, None),
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


# ── I7301: a permanent problem on the CO-INSTALL PAIR is a finding, not a gap ─
#
# The 2026-08-13 arc. `crucible-predictor` pinned by SHA since #422, so
# `parity_ok` came back None on every invocation and the SF fail-open fired
# every run. Underneath it, real drift: backtester v0.124.5 against a predictor
# SHA sitting ~v0.124.16 — the exact co-install shape the gate exists to catch,
# invisible for 13 days behind an alert that could not be cleared by waiting.
#
# The distinction this section pins down: an unreachable GitHub means we do not
# know the pin (fail open). A SHA or unrecognised pin ON A PAIR MEMBER means we
# read the file and the invariant is structurally unverifiable until a human
# edits it (halt). Floor-only repos keep the fail-open — see the tests above.


def test_sha_pinned_co_install_member_halts_with_a_named_offender():
    pins = {
        "nousergon/crucible-backtester": "v0.124.5",
        "nousergon/crucible-predictor": lpd.PinRead(
            None, lpd.SHA_PINNED, "c907a044bb1553815225327bc56644050543b6f2"
        ),
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["status"] == lpd.STATUS_MEASURED
    assert out["reason"] == "co_install_pin_unverifiable"
    # The offender names the repo, the problem, the SHA, and the other half of
    # the pair it could not be compared against — everything the FAILED SNS
    # alert must carry for a human to act without opening the execution.
    joined = " ".join(out["offenders"])
    assert "crucible-predictor" in joined
    assert "sha_pinned" in joined
    assert "c907a044bb1553815225327bc56644050543b6f2" in joined
    assert "v0.124.5" in joined


def test_the_halt_does_not_claim_a_parity_mismatch_it_never_measured():
    # has_drift=True says the invariant does not hold as VERIFIED. It must not
    # be accompanied by parity_ok=False, which asserts a measured mismatch —
    # the same fabrication (in the opposite direction) that this whole arc
    # exists to remove.
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = lpd.PinRead(None, lpd.SHA_PINNED, "abc1234")
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["parity_ok"] is None
    assert out["floor_ok"] is None


def test_unrecognised_pin_on_a_co_install_member_also_halts():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-backtester"] = lpd.PinRead(None, lpd.UNRECOGNISED, None)
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["reason"] == "co_install_pin_unverifiable"
    assert "crucible-backtester" in " ".join(out["offenders"])


def test_an_unreachable_co_install_member_still_fails_open():
    # The line I7048 drew and I7301 does not cross: a transient outage on a
    # pair member is an unknown pin, not a defect. Inverting it would let a
    # GitHub blip halt the weekly run.
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = None   # unreachable
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert "has_drift" not in out
    assert out["status"] == lpd.STATUS_UNKNOWN
    assert out["reason"] == "fetch_failed"


def test_a_permanent_pair_problem_outranks_a_transient_pair_outage():
    # Backtester unreachable AND predictor SHA-pinned. The permanent one wins:
    # even once GitHub recovers, the comparison still cannot be made.
    pins = {
        "nousergon/crucible-backtester": None,
        "nousergon/crucible-predictor": lpd.PinRead(None, lpd.SHA_PINNED, "abc1234"),
        "nousergon/nousergon-data": "v0.53.0",
        "nousergon/crucible-research": "v0.53.0",
    }
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert out["has_drift"] is True
    assert out["reason"] == "co_install_pin_unverifiable"
    # the transient one is still on the record
    assert out["unresolved"]["nousergon/crucible-backtester"]["problem"] == "unreachable"


def test_a_permanent_floor_only_problem_does_not_halt_the_pair_check():
    # research SHA-pinned, pair intact and matching. Nothing load-bearing is
    # unverifiable, so this must stay fail-open — a halt here would block the
    # weekly on the advisory half of the invariant.
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-research"] = lpd.PinRead(None, lpd.SHA_PINNED, "abc1234")
    with _patch_pins(pins):
        out = lpd.check_lib_pin_drift()
    assert "has_drift" not in out
    assert out["status"] == lpd.STATUS_UNKNOWN


# ── probe=True: a synthetic invocation records, it does not page ─────────────
# alpha-engine-config-I7954. `handler.py` attaches flow-doctor at ERROR, so
# the log LEVEL of a detected condition is the difference between an entry in
# CloudWatch and an email. `infrastructure/deploy.sh`'s canary invokes this
# action purely to exercise its wiring and gates on the PRESENCE of
# `has_drift`, never on its value — so a true finding from that invocation is
# noise. It fired for real on 2026-08-21T15:06:03Z, in the 13.5-minute window
# between the two halves of a cross-repo lockstep pin bump.


def _drifting_pins():
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = "v0.52.1"
    return pins


def test_probe_downgrades_a_detected_drift_to_warning(caplog):
    with caplog.at_level("WARNING", logger=lpd.log.name), _patch_pins(_drifting_pins()):
        lpd.check_lib_pin_drift(probe=True)
    records = [r for r in caplog.records if "Lib-pin drift DETECTED" in r.getMessage()]
    assert records, "the finding must still be recorded, only at a lower level"
    assert [r.levelname for r in records] == ["WARNING"]


def test_default_invocation_still_logs_a_detected_drift_at_error(caplog):
    # The Step Function's own pre-spend invocation passes no `probe`. A real
    # drift there is spend about to be wasted: it must still halt AND page.
    with caplog.at_level("WARNING", logger=lpd.log.name), _patch_pins(_drifting_pins()):
        lpd.check_lib_pin_drift()
    records = [r for r in caplog.records if "Lib-pin drift DETECTED" in r.getMessage()]
    assert [r.levelname for r in records] == ["ERROR"]


def test_probe_leaves_the_payload_identical():
    # The canary gates on the payload; changing it would change what the
    # deploy promotes on. `probe` may only move the log level.
    with _patch_pins(_drifting_pins()):
        as_probe = lpd.check_lib_pin_drift(probe=True)
    with _patch_pins(_drifting_pins()):
        as_gate = lpd.check_lib_pin_drift()
    assert as_probe == as_gate
    assert as_probe["has_drift"] is True


def test_probe_downgrades_the_unverifiable_parity_halt_too(caplog):
    # The second ERROR site: a co-install-pair member pinned by SHA is a
    # permanently unverifiable parity invariant (I7301). Same reasoning — the
    # canary is not the consumer of that verdict.
    pins = dict(_ALIGNED)
    pins["nousergon/crucible-predictor"] = lpd.PinRead(
        None, lpd.SHA_PINNED, "patched: 1a2b3c4d"
    )
    with caplog.at_level("WARNING", logger=lpd.log.name), _patch_pins(pins):
        out = lpd.check_lib_pin_drift(probe=True)
    assert out["has_drift"] is True
    records = [r for r in caplog.records if "Lib-pin drift HALT" in r.getMessage()]
    assert [r.levelname for r in records] == ["WARNING"]
