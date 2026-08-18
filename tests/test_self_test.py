"""Tests for `inference/self_test.py` — the predictor's known-answer self-test
(alpha-engine-config-I7262).

Four layers, and the last three are the load-bearing ones:

1. **The battery agrees on THIS runner.** Every case passes here too, so a CI
   failure and an in-Lambda failure mean the same thing and can be compared.
2. **The expectations are re-derived here, independently.** Each closed form is
   recomputed from the metric's definition. If the module's own arithmetic were
   ever quietly changed to match the implementation, this layer is what notices.
3. **The runner's outcome taxonomy holds.** Disagreed => FAIL, could-not-run =>
   UNKNOWN, over-budget => FAIL (Brian ruling 2026-08-13). This is the part that
   decides whether a harness fault gets reported as a correctness regression.
4. **The battery can actually FAIL.** A self-test never shown to fail is not
   evidence. `test_a_perturbed_implementation_is_caught` is the standing,
   automated form of the manual perturbation recorded in the PR body.
"""

from __future__ import annotations

import json
import math

import pytest
from nousergon_lib.quant.selftest_perturbation import assert_perturbation_caught

from inference import self_test as st
from training import self_test_cases as tsc


# ── layer 1: the real battery ───────────────────────────────────────────────

#: CI can reach `training/`, so it runs the FULL composed battery. The Lambda
#: cannot (`tests/test_dockerfile_import_closure.py` pins that `training/` is
#: outside its image), so it runs the inference scope alone. Both are exercised
#: below, and the scope is asserted — an inference-scope PASS must never be
#: readable as full coverage.
_ALL_SCOPES = {"training": tsc.build_cases}


@pytest.fixture(scope="module")
def body():
    return st.run_self_test(run_date="2026-08-15", extra_case_providers=_ALL_SCOPES)


def test_every_case_passes_on_this_runner(body):
    failures = [c for c in body["cases"] if c["verdict"] != st.PASS]
    assert not failures, json.dumps(failures, indent=2, default=str)
    assert body["verdict"] == st.PASS
    assert body["n_cases"] == len(st.build_cases()) + len(tsc.build_cases())


def test_all_four_case_classes_are_represented(body):
    """A case class silently dropped in a refactor is a coverage regression
    nothing else would notice: the artifact would still say PASS, on fewer
    questions."""
    names = {c["case"] for c in body["cases"]}
    assert any(n.endswith("_closed_form") for n in names)
    assert any(n.endswith("_metamorphic") for n in names)
    assert any(n.endswith("_degenerate") for n in names)
    assert any(n.endswith("_convention") for n in names)
    assert any(n.endswith("_agreement") for n in names)


def test_the_named_numeric_surfaces_are_all_covered(body):
    names = {c["case"] for c in body["cases"]}
    assert {
        "regime_zscore_closed_form",
        "composite_intensity_closed_form",
        "veto_adjustment_closed_form",
        "hmm_posterior_expectation_closed_form",
        "intensity_z_to_score_closed_form",
        "calibrator_p_up_closed_form",
        "combined_rank_closed_form",
        "predictor_ic_closed_form",
        "ic_cross_implementation_agreement",
    } <= names


def test_every_closed_form_case_asserts_to_1e_9(body):
    for case in body["cases"]:
        if case["case"].endswith("_closed_form"):
            assert case["tolerance"] <= 1e-9, case["case"]
            assert case["abs_error"] <= case["tolerance"], case["case"]


#: The ONLY cases allowed a tolerance looser than 1e-9, each with its reason.
#: A tolerance wide enough to hide a convention change is a case that cannot
#: fail, so the exemptions are enumerated rather than left to judgement.
_TOLERANCE_EXEMPTIONS = {
    # `downside_ic_stats` rounds its output to 6dp, which caps the achievable
    # agreement on a ratio built from it at ~1e-6. Tightening below that would
    # assert precision the production artifact does not carry.
    "sortino_convention_agreement": 1e-5,
}


def test_no_case_buys_itself_an_unjustified_tolerance(body):
    for case in body["cases"]:
        allowed = _TOLERANCE_EXEMPTIONS.get(case["case"], 1e-9)
        assert case["tolerance"] <= allowed, (
            f"{case['case']} carries tolerance {case['tolerance']} — justify it "
            "in _TOLERANCE_EXEMPTIONS or tighten it"
        )


def test_the_battery_completes_well_inside_its_budget(body):
    """It runs beside the weekly predictions; a battery that costs minutes would
    be the reason someone disables it."""
    assert body["wall_clock_seconds"] < 30.0


# ── the scope split — a PASS must state what it is a PASS OVER ──────────────

def test_the_artifact_declares_which_scopes_ran(body):
    """"A number published without its members" is its own failure class. A
    battery that silently asks fewer questions still reports PASS, so the scope
    is not optional metadata — it is what the verdict MEANS."""
    assert body["scope"] == ["inference", "training"]
    assert body["scope_note"]


def test_the_inference_scope_alone_is_labelled_as_such():
    """What the Lambda runs. `training/` is outside its image, so this is the
    honest maximum there — and the artifact must say so rather than let an
    inference-scope PASS read as full coverage."""
    out = st.run_self_test(run_date="2026-08-15")
    assert out["scope"] == ["inference"]
    assert out["verdict"] == st.PASS
    assert out["n_cases"] == len(st.build_cases())
    names = {c["case"] for c in out["cases"]}
    # The training-scope cases must be ABSENT, not silently passing.
    assert "ic_cross_implementation_agreement" not in names
    assert "downside_ic_sortino_closed_form" not in names


def test_training_scope_is_not_reachable_from_the_inference_module():
    """The seam this split exists to respect. `inference/self_test.py` must not
    import `training` at any depth, or
    tests/test_dockerfile_import_closure.py::
    test_training_package_is_not_reachable_from_any_lambda_entrypoint fails and
    the Lambda image ships a module it cannot resolve at runtime.

    Asserted on the SOURCE rather than by import, because a deferred import
    inside a function would not show up at import time — which is exactly the
    shape that made this a real failure the first time.
    """
    import pathlib

    source = pathlib.Path(st.__file__).read_text()
    offenders = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith(("import training", "from training"))
    ]
    assert not offenders, offenders


def test_the_training_scope_carries_the_cross_implementation_cases():
    """The IC agreement check and the I7271 divergence are the highest-value
    cases in the battery; a refactor that dropped them would leave a green
    artifact asking materially less."""
    names = {c.name for c in tsc.build_cases()}
    assert {
        "ic_cross_implementation_agreement",
        "sortino_convention_agreement",
        "downside_ic_sortino_closed_form",
        "predictor_ic_closed_form",
    } <= names


# ── layer 2: the expectations, re-derived from first principles ─────────────

_HISTORY = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
_VALUE = 8.0


def test_expected_zscore_uses_the_population_denominator():
    """The load-bearing convention: ddof=0, not ddof=1. The sample-sd alternative
    differs by sqrt(6/5) = 1.0954, so this assertion is what stops the
    expectation drifting to match a changed implementation."""
    n = len(_HISTORY)
    mean = sum(_HISTORY) / n
    pop = math.sqrt(sum((v - mean) ** 2 for v in _HISTORY) / n)
    sample = math.sqrt(sum((v - mean) ** 2 for v in _HISTORY) / (n - 1))
    assert st._expected_zscore() == pytest.approx((_VALUE - mean) / pop, rel=0, abs=1e-15)
    assert st._expected_zscore() != pytest.approx((_VALUE - mean) / sample, rel=1e-6)


def test_expected_spearman_is_the_rank_correlation_definition():
    """rho = 1 - 6*sum(d^2)/(n(n^2-1)); one adjacent transposition => sum(d^2)=2."""
    n = tsc.IC_N
    assert tsc.expected_spearman_ic() == pytest.approx(
        1.0 - 12.0 / (n * (n * n - 1)), rel=0, abs=1e-15)


def test_expected_hmm_discount_is_one_halflife_past_grace():
    assert st._expected_hmm_staleness_discount() == pytest.approx(0.625, rel=0, abs=1e-15)


def test_expected_calibrator_p_up_is_the_linear_fallback():
    assert st._expected_calibrator_p_up() == pytest.approx(0.60, rel=0, abs=1e-15)


def test_the_two_sortino_conventions_now_agree():
    """alpha-engine-config-I7271, FIXED 2026-08-13 (Brian ruling 'use sota'):
    the predictor now divides the squared downside by N, matching
    nousergon_lib.quant.riskstats's convention, so the two agree exactly
    (ratio 1.0). The pre-fix factor and its direction — the predictor's OLD
    convention was the smaller, more conservative number, which is why it
    never looked implausible — is preserved for the historical record."""
    n = len(tsc.DOWNSIDE_IC)
    n_down = sum(1 for v in tsc.DOWNSIDE_IC if v < 0.0)
    assert tsc.expected_downside_deviation_ratio() == pytest.approx(
        math.sqrt(n / n_down), rel=0, abs=1e-15)
    assert tsc.expected_sortino_convention_ratio() == pytest.approx(1.0, rel=0, abs=1e-15)
    assert tsc.expected_sortino_pre_fix_convention_ratio() == pytest.approx(
        math.sqrt(n_down / n), rel=0, abs=1e-15)
    assert tsc.expected_sortino_pre_fix_convention_ratio() < 1.0


def test_the_composite_intensity_fixture_is_not_self_cancelling():
    """A two-feature blend at weights (+1, -1) with the SAME value for both
    features has expectation 0.0 for ANY pair of z-scores — the case would pass
    against an arbitrarily broken function. Guards the fixture against
    regressing to that shape."""
    assert st._Z_VALUE != st._Z_VALUE_B
    assert abs(st._expected_composite_intensity()) > 1.0


def test_the_ic_fixture_is_not_a_perfect_correlation():
    """A rho of exactly 1.0 passes under any implementation that merely preserves
    order, including a broken one."""
    assert 0.9 < tsc.expected_spearman_ic() < 1.0


# ── layer 3: the artifact and the outcome taxonomy ─────────────────────────

def test_artifact_carries_the_provenance_header(body):
    """The resolved library versions are the point — they are what makes this an
    instrument check rather than a code check."""
    assert body["schema"] == st.SCHEMA
    assert body["component"] == "predictor"
    assert body["run_date"] == "2026-08-15"
    assert body["python"]
    assert body["code_sha"]
    for dist in st._TRACKED_DISTRIBUTIONS:
        assert dist in body["libraries"], dist


def test_every_case_publishes_enough_to_re_derive_it(body):
    for case in body["cases"]:
        assert case["description"]
        assert case["inputs"]
        assert "units" in case["inputs"], case["case"]
        assert case["expected"] is not None


def test_known_gap_cases_say_so_in_words(body):
    """A pinned-wrong case must never read as an endorsement. The artifact has to
    carry that in words, not leave a reader to infer it from a green row.

    config-I7272 fixed the two regime-layer known gaps (zero-variance z-score,
    regime_score with no heads); config-I7271 (Sortino denominator, fixed
    2026-08-13) was the last remaining one. Zero known gaps is the CURRENT
    state, not a structural guarantee — the assertion loop below still holds
    if a future known gap is pinned, so this test does not go stale silently."""
    gaps = [c for c in body["cases"] if c.get("known_gap")]
    assert len(gaps) == body["n_known_gaps"] == 0
    for case in gaps:
        assert case["gap_issue"].startswith("alpha-engine-config-I")
        assert "NOT" in case["known_gap_note"]
        assert "KNOWN GAP" in case["description"]
        assert "PINNED NOT FIXED" in case["description"]


def test_the_artifact_is_strict_json(body):
    """alpha-engine-config-I7237's third finding was a metrics.json that was not
    strict JSON. NaN/Infinity are not JSON, so a consumer using a strict parser
    would fail on the artifact rather than on the numbers."""
    text = json.dumps(body, default=str)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict by default


def test_a_disagreeing_case_is_FAIL_not_UNKNOWN():
    def cases():
        return [st.Case(name="wrong", description="d", inputs={"units": "u"},
                        expected=1.0, compute=lambda: 2.0)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["verdict"] == st.FAIL
    assert out["n_failed"] == 1


def test_a_case_that_could_not_run_is_UNKNOWN_not_FAIL():
    """Collapsing these two would make a broken image read as a correctness
    regression."""
    def cases():
        return [st.Case(name="boom", description="d", inputs={"units": "u"},
                        expected=1.0, compute=_raise)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["cases"][0]["verdict"] == st.UNKNOWN
    assert out["cases"][0]["error_class"] == "RuntimeError"
    assert out["verdict"] == st.UNKNOWN


def _raise():
    raise RuntimeError("the instrument could not be measured")


def test_a_battery_that_could_not_be_built_is_UNKNOWN_and_never_raises():
    def cases():
        raise ImportError("no numpy in this image")

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    assert out["verdict"] == st.UNKNOWN
    assert out["status"] == "error"
    assert out["n_cases"] == 0


def test_an_over_budget_case_is_FAIL(monkeypatch):
    """Brian ruling 2026-08-13: a timeout is FAIL, never UNKNOWN."""
    def slow():
        import time
        time.sleep(0.05)
        return 1.0

    def cases():
        return [st.Case(name="slow", description="d", inputs={"units": "u"},
                        expected=1.0, compute=slow)]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases,
                           case_timeout_seconds=0.001)
    assert out["cases"][0]["verdict"] == st.FAIL
    assert out["cases"][0]["timed_out"] is True
    assert out["verdict"] == st.FAIL


def test_run_self_test_never_raises_whatever_the_provider_does():
    """The contract that keeps an accuracy instrument from taking down the
    pipeline it measures."""
    for provider in (lambda: None, lambda: [1, 2, 3], _raise):
        out = st.run_self_test(run_date="2026-08-15", case_provider=provider)
        assert out["verdict"] in (st.PASS, st.FAIL, st.UNKNOWN)


def test_verdict_is_pass_is_strict():
    assert st.verdict_is_pass("PASS")
    for other in (None, "ok", "pass", "OK", "UNKNOWN", ""):
        assert not st.verdict_is_pass(other)


# ── layer 4: the battery can FAIL — the acceptance criterion for I7262 ──────

@pytest.mark.parametrize(
    "module_path,attr,perturbed",
    [
        # The tanh scale: a 2.0 -> 2.0001 change is far smaller than any
        # plausible edit, and must still be caught.
        ("regime.drawdown", "INTENSITY_Z_TANH_SCALE", 2.0001),
        # The staleness floor.
        ("regime.drawdown", "HMM_STALENESS_FLOOR", 0.2501),
        # The neutral band.
        ("regime.drawdown", "REGIME_SCORE_NEUTRAL_BAND", 0.2001),
    ],
)
def test_a_perturbed_implementation_is_caught(monkeypatch, module_path, attr, perturbed):
    """THE acceptance criterion (alpha-engine-config-I7262): a self-test never
    shown to fail is not evidence.

    Each parameter perturbs ONE production constant by ~1e-4 relative — far below
    any realistic edit — and asserts the battery goes FAIL. This fleet has shipped
    several detectors that could not fail; this is the standing guard against
    adding another.

    Delegates to the LIFTED helper (alpha-engine-config-I7238/I7262) — the same
    monkeypatch/rerun/assert-FAIL shape crucible-research's test suite carries
    independently; both now import one proven-correct implementation from
    ``nousergon_lib.quant.selftest_perturbation`` instead of each keeping its
    own copy.
    """
    out = assert_perturbation_caught(
        monkeypatch,
        module_path=module_path,
        attr=attr,
        perturbed=perturbed,
        run=lambda: st.run_self_test(run_date="2026-08-15", extra_case_providers=_ALL_SCOPES),
    )
    assert out["n_failed"] >= 1


def test_perturbing_the_z_score_denominator_is_caught(monkeypatch):
    """The ddof convention, perturbed at the production function rather than at a
    constant — the shape the sqrt(365)-vs-sqrt(252) defect (I7236) actually took."""
    from regime import composite

    def ddof_one(value, history):
        if len(history) == 0:
            return 0.0
        mu = float(history.mean())
        sigma = float(history.std(ddof=1))  # the perturbation
        if sigma == 0.0:
            return 0.0
        return (float(value) - mu) / sigma

    out = assert_perturbation_caught(
        monkeypatch,
        module_path="regime.composite",
        attr="_zscore",
        perturbed=ddof_one,
        run=lambda: st.run_self_test(run_date="2026-08-15", extra_case_providers=_ALL_SCOPES),
        case_name="regime_zscore_closed_form",
    )
    failed = {c["case"] for c in out["cases"] if c["verdict"] == st.FAIL}
    # The metamorphic case must NOT fire: a ddof change preserves affine
    # invariance, which is exactly why closed-form and metamorphic cases are
    # both needed and neither substitutes for the other.
    assert "regime_zscore_affine_invariance_metamorphic" not in failed


# ── the console surface ─────────────────────────────────────────────────────

def test_console_envelope_matches_the_published_contract(body):
    """FORK-DETECTION BACKSTOP (shared-code-policy.md §5.1).

    The envelope is built here rather than imported from
    `nousergon_lib.fleet_check_result` because that module first ships in lib
    v0.124.29 and this repo pins older (see the module docstring). This test is
    what holds the interim duplication legitimate: every field the console's
    checks-envelope adapter reads must be present and correctly typed.
    Migration is tracked in alpha-engine-config-I7274.
    """
    env = st.console_envelope(body)
    for field in ("schema_version", "check_id", "label", "ran_at", "status",
                  "summary", "cadence_minutes", "deep_link", "findings"):
        assert field in env, field
    assert env["schema_version"] == 1
    assert env["check_id"] == st.CHECK_ID
    assert "/" not in env["check_id"], "check_id must be a single path segment"
    assert env["status"] in ("ok", "attention", "error")
    assert isinstance(env["cadence_minutes"], int) and env["cadence_minutes"] > 0
    assert isinstance(env["findings"], list)
    json.loads(json.dumps(env, default=str))


def test_console_envelope_key_is_where_the_adapter_looks(body):
    assert st.console_envelope_key() == f"ops/checks/{st.CHECK_ID}/latest.json"


def test_console_status_never_renders_unknown_as_green():
    """`principles.md` §2.7: no data is never rendered as green."""
    assert st.console_envelope({"verdict": st.PASS})["status"] == "ok"
    assert st.console_envelope({"verdict": st.FAIL})["status"] == "error"
    assert st.console_envelope({"verdict": st.UNKNOWN})["status"] == "attention"
    assert st.console_envelope({})["status"] == "attention"


def test_console_envelope_lists_every_non_passing_case():
    def cases():
        return [
            st.Case(name="ok", description="d", inputs={"units": "u"},
                    expected=1.0, compute=lambda: 1.0),
            st.Case(name="bad", description="d", inputs={"units": "u"},
                    expected=1.0, compute=lambda: 9.0),
        ]

    out = st.run_self_test(run_date="2026-08-15", case_provider=cases)
    env = st.console_envelope(out)
    assert [f["key"] for f in env["findings"]] == ["bad"]
    assert "expected=1.0" in env["findings"][0]["detail"]


def test_publish_console_row_never_raises_on_a_broken_client(body):
    """A check must not go red because its telemetry did."""
    class Boom:
        def put_object(self, **kwargs):
            raise RuntimeError("s3 is down")

    assert st.publish_console_row(body, s3_client=Boom()) is None


def test_publish_console_row_dry_run_writes_nothing(body):
    assert st.publish_console_row(body, dry_run=True) is None


def test_publish_console_row_writes_the_expected_key(body):
    captured = {}

    class Capture:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    uri = st.publish_console_row(body, s3_client=Capture())
    assert uri == f"s3://{st.RESEARCH_BUCKET}/ops/checks/{st.CHECK_ID}/latest.json"
    assert captured["Bucket"] == st.RESEARCH_BUCKET
    assert captured["ContentType"] == "application/json"
    assert json.loads(captured["Body"])["check_id"] == st.CHECK_ID


# ── the S3 artifact ─────────────────────────────────────────────────────────

def test_self_test_key_is_beside_the_run(body):
    assert st.self_test_key("2026-08-15") == "predictor/2026-08-15/self_test.json"


def test_write_self_test_writes_the_body_verbatim(body):
    captured = {}

    class Capture:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    key = st.write_self_test("bucket", "2026-08-15", body, s3_client=Capture())
    assert key == "predictor/2026-08-15/self_test.json"
    assert json.loads(captured["Body"])["verdict"] == body["verdict"]


# ── the handler action ──────────────────────────────────────────────────────

@pytest.fixture
def stubbed_preflight(monkeypatch):
    """Replace PredictorPreflight with a no-op so tests don't hit S3.

    Mirrors the fixture in tests/test_inference_handler.py rather than inventing
    a second isolation mechanism for the same dependency.
    """
    import sys
    from unittest.mock import MagicMock

    pf = MagicMock()
    pf.return_value.run = MagicMock()
    pf.return_value.run_for_drift_gate = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.preflight",
                        MagicMock(PredictorPreflight=pf))
    return pf


def test_handler_check_self_test_returns_the_verdict(stubbed_preflight, monkeypatch):
    from unittest.mock import MagicMock

    import inference.handler as h

    monkeypatch.setattr(st, "write_self_test", lambda *a, **k: "predictor/2026-08-15/self_test.json")
    monkeypatch.setattr(st, "publish_console_row",
                        lambda *a, **k: "s3://alpha-engine-research/ops/checks/x/latest.json")

    result = h.handler({"action": "check_self_test", "date": "2026-08-15"}, MagicMock())
    assert result["verdict"] == st.PASS
    assert result["n_failed"] == 0
    assert result["run_date"] == "2026-08-15"
    assert result["self_test_key"] == "predictor/2026-08-15/self_test.json"
    assert result["expected_key"] == "predictor/2026-08-15/self_test.json"


def test_handler_check_self_test_survives_both_emissions_failing(
    stubbed_preflight, monkeypatch,
):
    """The verdict is the deliverable; the artifact and the console row are
    EVIDENCE. Losing the evidence must degrade the evidence, never turn a
    measured PASS into a stage error — and must never raise."""
    from unittest.mock import MagicMock

    import inference.handler as h

    def boom(*a, **k):
        raise RuntimeError("s3 is down")

    monkeypatch.setattr(st, "write_self_test", boom)
    monkeypatch.setattr(st, "publish_console_row", lambda *a, **k: None)

    result = h.handler({"action": "check_self_test", "date": "2026-08-15"}, MagicMock())
    assert result["verdict"] == st.PASS
    assert result["self_test_key"] is None
    assert result["console_row"] is None
    # The key it WOULD have written is still reported, so an operator can check
    # for the artifact's absence rather than guess where it should have been.
    assert result["expected_key"] == "predictor/2026-08-15/self_test.json"


def test_handler_check_self_test_resolves_a_missing_date(stubbed_preflight, monkeypatch):
    """A key containing "None" is a key nothing can ever look up."""
    import datetime as dt
    from unittest.mock import MagicMock

    import inference.handler as h

    monkeypatch.setattr(st, "write_self_test", lambda *a, **k: "k")
    monkeypatch.setattr(st, "publish_console_row", lambda *a, **k: None)

    result = h.handler({"action": "check_self_test"}, MagicMock())
    assert result["run_date"] == dt.date.today().isoformat()
    assert "None" not in result["expected_key"]
