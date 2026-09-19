"""The promotion gate's ``xsec_sd`` must be measured on the SERVED population.

alpha-engine-config-I11106.

``XSEC_SD_ABSOLUTE_FLOOR`` is 0.015. alpha-engine-config-I9267 states its
provenance verbatim: *"an absolute floor alongside the relative one, derived
from the healthy population (2026-08-10..08-21 ranged 0.0200-0.0434)"* — served
``alpha_stdev`` over batches of 26-30 names. It was being compared against a
number pooled over the whole L2 design matrix, ~896 names per date. Measured on
the serving champion ``v3.0-meta-2026-08-14-119e069b``::

    served cut (attractiveness_top_20)   0.007408   0.49x floor   FAILS
    whole served batch (35 names)        0.016768   1.12x floor
    fit-time panel (what the veto read)  0.048588   3.24x floor   passed by 3x

6.56x. The gate was too LENIENT, not too strict.

Every test here is RED against the pre-fix tree: ``training/served_cut_mask.py``
does not exist there, so the module-level import fails and the whole file
errors — the strongest form of champion-challenger-policy.md §7.4 ("a guard must
be verified to fail without the fix").
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.served_cut_mask import (
    BASIS_FIT_PANEL,
    BASIS_SERVED_CUT,
    MIN_CUT_TICKERS,
    as_iso_date,
    list_membership_dates,
    resolve_served_cut_mask,
)

CUT = "attractiveness_top_20"


def _membership(run_date, tickers, *, cut=CUT):
    return {
        "run_date": run_date,
        "generated_at": f"{run_date}T12:00:00+00:00",
        "predictor_universe_cut": cut,
        "cuts": {cut: {"tickers": list(tickers)}},
    }


class _FakeS3:
    """Only the two calls this module makes: the paginated list and get_object."""

    def __init__(self, objects: dict):
        self.objects = objects
        self.gets: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class _P:
            def paginate(self, Bucket, Prefix, Delimiter):  # noqa: N803
                prefixes = sorted({
                    k[: k.index("/", len(Prefix)) + 1]
                    for k in outer.objects
                    if k.startswith(Prefix) and "/" in k[len(Prefix):]
                })
                yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}

        return _P()

    def get_object(self, Bucket, Key):  # noqa: N803
        self.gets.append(Key)
        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key]
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)

        class _B:
            def read(self_inner):
                return payload.encode() if isinstance(payload, str) else payload

        return {"Body": _B()}


def _s3_with(memberships: dict, *, holdings_csv: str | None = None):
    objects: dict = {
        f"universe_membership/{d}/membership.json": m
        for d, m in memberships.items()
    }
    if holdings_csv is not None:
        objects["trades/eod_pnl.csv"] = holdings_csv
    return _FakeS3(objects)


def _panel(rows):
    """``rows`` is ``[(date, ticker), ...]`` → two aligned lists."""
    return [d for d, _ in rows], [t for _, t in rows]


# ── the numbers this exists for ──────────────────────────────────────────────

def test_the_two_bases_have_distinct_declared_labels():
    """Mixing them inside one manifest block is the defect. A reader must never
    have to infer which population a figure came from."""
    assert BASIS_FIT_PANEL != BASIS_SERVED_CUT
    assert BASIS_SERVED_CUT == "served_cut"


def test_the_mask_selects_only_the_cut_and_the_book():
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", ["AAA", "BBB", "CCC"] + [f"T{i:02d}" for i in range(17)])})
    dates, tickers = _panel([
        ("2026-09-04", "AAA"),
        ("2026-09-04", "ZZZ"),   # outside the cut and not held
        ("2026-09-05", "BBB"),
        ("2026-09-05", "YYY"),   # outside
    ])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert report["status"] == "ok"
    assert list(mask) == [True, False, True, False]
    assert report["n_rows_selected"] == 2
    assert report["n_dates_resolved"] == 2
    assert report["cut_names"] == [CUT]


def test_a_held_name_outside_the_cut_is_still_served_and_therefore_measured():
    """The executor needs a prediction for a name the book holds even when the
    forward-looking cut has dropped it — ``resolve_universe_from_membership``
    unions it in, so the measurement must too."""
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([("2026-09-04", "HELD"), ("2026-09-04", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held={"HELD"})
    assert list(mask) == [True, True]
    assert report["n_held"] == 1


def test_the_held_set_is_read_through_the_inference_path_not_reimplemented():
    """One implementation of the EOD-surface read in the tree. The CSV shape
    here is the executor's; if ``_read_holdings`` changes, this fails."""
    csv = (
        "date,positions_snapshot\n"
        '2026-09-03,"{""OLD"": 1}"\n'
        '2026-09-04,"{""HELD"": 3}"\n'
    )
    s3 = _s3_with(
        {"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])},
        holdings_csv=csv,
    )
    dates, tickers = _panel([("2026-09-04", "HELD"), ("2026-09-04", "OLD")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers)
    assert report["n_held"] == 1
    assert list(mask) == [True, False]


def test_the_fit_panel_is_far_wider_than_the_served_cut():
    """The shape of the whole finding, in one assertion: a ~900-name panel and a
    20-name cut are not the same population, and the floor was derived from the
    second."""
    cut = [f"T{i:02d}" for i in range(20)]
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", cut)})
    board = cut + [f"X{i:03d}" for i in range(876)]
    dates, tickers = _panel([("2026-09-04", t) for t in board])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert report["n_rows_total"] == 896
    assert report["n_rows_selected"] == 20


# ── resolution is DATED and immutable, never the mutable pointer ─────────────

def test_latest_json_is_never_consulted():
    """``universe_membership/latest.json`` is a mutable pointer. Resolving a
    historical date through it would make the measurement irreproducible — and
    would stamp today's cut onto every row of the panel."""
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    s3.objects["universe_membership/latest.json"] = _membership(
        "2026-09-18", ["LATEST"] * 20)
    dates, tickers = _panel([("2026-09-04", "T00"), ("2026-09-04", "LATEST")])
    mask, _ = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert list(mask) == [True, False]
    assert "universe_membership/latest.json" not in s3.gets


def test_a_date_takes_the_nearest_PRIOR_artifact_never_a_later_one():
    """Look-ahead would put names into a date's cut that the producer had not
    yet ranked there."""
    s3 = _s3_with({
        "2026-09-04": _membership("2026-09-04", ["EARLY"] + [f"T{i:02d}" for i in range(19)]),
        "2026-09-11": _membership("2026-09-11", ["LATE"] + [f"T{i:02d}" for i in range(19)]),
    })
    dates, tickers = _panel([("2026-09-08", "EARLY"), ("2026-09-08", "LATE")])
    mask, _ = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert list(mask) == [True, False]


def test_each_artifact_is_fetched_once_however_many_dates_use_it():
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([(f"2026-09-{d:02d}", "T00") for d in (4, 5, 8, 9, 10)])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert report["n_dates_resolved"] == 5
    assert s3.gets.count("universe_membership/2026-09-04/membership.json") == 1


def test_as_iso_date_normalises_the_panels_date_types():
    import datetime as dt

    assert as_iso_date("2026-09-04") == "2026-09-04"
    assert as_iso_date(dt.date(2026, 9, 4)) == "2026-09-04"
    assert as_iso_date(dt.datetime(2026, 9, 4, 13, 30)) == "2026-09-04"
    assert as_iso_date("2026-09-04T13:30:00+00:00") == "2026-09-04"
    assert as_iso_date(None) is None
    assert as_iso_date("not-a-date") is None


def test_list_membership_dates_ignores_non_date_prefixes():
    s3 = _FakeS3({
        "universe_membership/2026-09-04/membership.json": {},
        "universe_membership/2026-09-11/membership.json": {},
        "universe_membership/scratch/notes.json": {},
    })
    assert list_membership_dates(s3, "bkt") == ["2026-09-04", "2026-09-11"]


# ── the edge cases, each a NAMED state and never a silent widening ───────────

def test_a_date_with_no_membership_artifact_is_named_and_excluded():
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([("2026-06-01", "T00"), ("2026-09-04", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert list(mask) == [False, True]
    assert report["date_states"]["no_membership_artifact"] == 1
    assert "2026-06-01" in report["excluded_dates"]["no_membership_artifact"]


def test_an_artifact_older_than_the_declared_max_age_is_STALE_not_used():
    """The same staleness rule the inference path enforces, imported from it.
    A 40-day-old cut describes a dead cycle, and scoring against it is the
    frozen-universe defect ``load_universe`` already records."""
    from inference.stages.load_universe import MEMBERSHIP_MAX_AGE_DAYS

    s3 = _s3_with({"2026-07-01": _membership("2026-07-01", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([("2026-08-10", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert not mask.any()
    assert report["date_states"] == {"membership_stale": 1}
    assert report["max_age_days"] == MEMBERSHIP_MAX_AGE_DAYS
    assert report["status"] == "unresolvable"


def test_a_date_just_inside_the_age_limit_still_resolves():
    from inference.stages.load_universe import MEMBERSHIP_MAX_AGE_DAYS

    import datetime as dt

    base = dt.date(2026, 9, 4)
    edge = (base + dt.timedelta(days=MEMBERSHIP_MAX_AGE_DAYS)).isoformat()
    over = (base + dt.timedelta(days=MEMBERSHIP_MAX_AGE_DAYS + 1)).isoformat()
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([(edge, "T00"), (over, "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert list(mask) == [True, False]
    assert report["date_states"]["membership_stale"] == 1


def test_an_artifact_that_names_no_predictor_cut_is_cut_unresolvable():
    broken = _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])
    broken.pop("predictor_universe_cut")
    s3 = _s3_with({"2026-09-04": broken})
    dates, tickers = _panel([("2026-09-04", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert not mask.any()
    assert report["date_states"] == {"cut_unresolvable": 1}


def test_an_artifact_naming_a_cut_it_does_not_contain_is_cut_unresolvable():
    broken = _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])
    broken["predictor_universe_cut"] = "attractiveness_top_25"
    s3 = _s3_with({"2026-09-04": broken})
    dates, tickers = _panel([("2026-09-04", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert report["date_states"] == {"cut_unresolvable": 1}


def test_a_cut_below_the_declared_minimum_width_is_cut_too_narrow():
    """A truncated artifact. Dispersion over a handful of names is noise wearing
    the gate's name — and widening back to the panel is the defect being fixed,
    so the date is dropped instead."""
    s3 = _s3_with({
        "2026-09-04": _membership(
            "2026-09-04", [f"T{i:02d}" for i in range(MIN_CUT_TICKERS - 1)]),
    })
    dates, tickers = _panel([("2026-09-04", "T00")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert not mask.any()
    assert report["date_states"] == {"cut_too_narrow": 1}
    assert report["min_cut_tickers"] == MIN_CUT_TICKERS


def test_a_date_whose_rows_are_all_outside_the_cut_is_no_rows_in_cut():
    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])})
    dates, tickers = _panel([("2026-09-04", "ZZZ")])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert not mask.any()
    assert report["date_states"] == {"no_rows_in_cut": 1}


@pytest.mark.parametrize("state_setup", ["missing", "stale", "unresolvable", "narrow"])
def test_NO_degraded_state_EVER_falls_back_to_the_full_panel(state_setup):
    """The single most load-bearing assertion in this file.

    A silent fallback to the whole design matrix would restore the exact 6.6x
    overstatement I11106 removes, while every surface still read green.
    """
    if state_setup == "missing":
        memberships = {}
        panel_date = "2026-09-04"
    elif state_setup == "stale":
        memberships = {"2026-07-01": _membership("2026-07-01", [f"T{i:02d}" for i in range(20)])}
        panel_date = "2026-08-10"
    elif state_setup == "unresolvable":
        m = _membership("2026-09-04", [f"T{i:02d}" for i in range(20)])
        m.pop("predictor_universe_cut")
        memberships = {"2026-09-04": m}
        panel_date = "2026-09-04"
    else:
        memberships = {"2026-09-04": _membership("2026-09-04", ["ONE", "TWO"])}
        panel_date = "2026-09-04"

    s3 = _s3_with(memberships)
    dates, tickers = _panel([(panel_date, f"T{i:02d}") for i in range(20)])
    mask, report = resolve_served_cut_mask(s3, "bkt", dates, tickers, held=set())
    assert not mask.any(), "a degraded date must contribute NO rows"
    assert report["n_rows_selected"] == 0
    assert report["status"] == "unresolvable"
    assert "not" in (report["reason"] or "").lower()


def test_an_s3_failure_is_an_error_state_with_an_empty_mask_never_a_raise():
    """A diagnostic must not take the weekly training run down — and must not
    quietly widen either."""

    class _Boom:
        def get_paginator(self, name):
            raise RuntimeError("s3 is down")

    mask, report = resolve_served_cut_mask(
        _Boom(), "bkt", ["2026-09-04"], ["AAA"], held=set())
    assert not mask.any()
    assert report["status"] == "error"
    assert "s3 is down" in report["reason"]


def test_a_misaligned_panel_is_refused_rather_than_guessed():
    mask, report = resolve_served_cut_mask(
        _s3_with({}), "bkt", ["2026-09-04", "2026-09-05"], ["AAA"], held=set())
    assert not mask.any()
    assert report["status"] == "error"
    assert "aligned" in report["reason"]


def test_the_report_counts_are_complete_even_when_the_enumeration_is_capped():
    from training.served_cut_mask import MAX_REPORTED_DATES_PER_STATE

    n = MAX_REPORTED_DATES_PER_STATE + 5
    dates = [f"2026-01-{i + 1:02d}" for i in range(n)]
    mask, report = resolve_served_cut_mask(
        _s3_with({}), "bkt", dates, ["AAA"] * n, held=set())
    assert report["date_states"]["no_membership_artifact"] == n
    assert len(report["excluded_dates"]["no_membership_artifact"]) == (
        MAX_REPORTED_DATES_PER_STATE
    )


# ── the end-to-end consequence: the floor now judges the right population ────

def test_a_model_passing_the_floor_on_the_panel_can_FAIL_it_on_the_served_cut():
    """The finding itself, reproduced arithmetically.

    A fitted predictor whose dispersion is wide across the whole ~900-name board
    but narrow within the 20 names the executor can enter clears the 0.015 floor
    on the panel and fails it on the cut. Before I11106 only the first number
    reached the veto.
    """
    from training.promotion_behavioral_veto import XSEC_SD_ABSOLUTE_FLOOR
    from training.xsec_variance_share import cross_sectional_variance_share

    rng = np.random.default_rng(11106)
    cut = [f"C{i:02d}" for i in range(20)]
    rest = [f"R{i:03d}" for i in range(876)]
    dates_u = ["2026-09-04", "2026-09-05", "2026-09-08"]

    rows_date, rows_tkr, feature = [], [], []
    for d in dates_u:
        for t in cut:
            rows_date.append(d)
            rows_tkr.append(t)
            feature.append(rng.normal(0.0, 0.004))   # tight inside the cut
        for t in rest:
            rows_date.append(d)
            rows_tkr.append(t)
            feature.append(rng.normal(0.0, 0.060))   # wide across the board

    X = np.array(feature, dtype=float).reshape(-1, 1)
    coefs = {"f": 1.0}

    panel = cross_sectional_variance_share(X, rows_date, ["f"], coefs)
    assert panel["xsec_sd"] > XSEC_SD_ABSOLUTE_FLOOR

    s3 = _s3_with({"2026-09-04": _membership("2026-09-04", cut)})
    mask, report = resolve_served_cut_mask(
        s3, "bkt", rows_date, rows_tkr, held=set())
    assert report["status"] == "ok"
    served = cross_sectional_variance_share(
        X[mask], [d for d, k in zip(rows_date, mask) if k], ["f"], coefs)
    assert served["xsec_sd"] < XSEC_SD_ABSOLUTE_FLOOR
    assert panel["xsec_sd"] / served["xsec_sd"] > 5.0


def test_an_unresolvable_served_cut_yields_an_unmeasurable_summary_which_VETOES():
    """The chain end to end: no served cut → no ``xsec_sd`` → refusal.

    ``unmeasurable`` is never a pass (champion-challenger-policy.md §§7.2, 5.1),
    and since alpha-engine-config-I11106 the magnitude leg implements that
    instead of merely asserting it in prose.
    """
    from training.promotion_behavioral_veto import evaluate_behavioral_veto

    mask, report = resolve_served_cut_mask(
        _s3_with({}), "bkt", ["2026-09-04"], ["AAA"], held=set())
    assert not mask.any()

    manifest = {"behavioral_metrics": {
        "xsec_sd": None,
        "xsec_variance_share": None,
        # The fit-panel figure is present and healthy — and must NOT rescue it.
        "xsec_sd_fit_panel": 0.048588,
    }}
    verdict = evaluate_behavioral_veto(manifest, manifest)
    assert verdict["status"] == "veto"
    assert [v for v in verdict["vetoes"] if v["metric"] == "xsec_sd"]


# ── structural guard: the wiring itself, which no unit test can reach ────────

def test_meta_trainer_wires_the_served_cut_and_persists_BOTH_bases():
    """``run_meta_training`` is not callable in a unit test, and the regression
    this guards is a DELETION — the mask call being dropped and ``xsec_sd``
    silently reverting to the fitted panel, which reads green everywhere.

    alpha-engine-config-I6495 is the precedent: a "consolidation" PR deleted the
    universe-membership chokepoint in ``load_universe`` and nothing failed.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "training" / "meta_trainer.py").read_text()

    assert "resolve_served_cut_mask" in src, (
        "the served-cut mask call has been removed — xsec_sd would revert to "
        "the ~896-name fitted panel and the 0.015 floor would again be 6.6x "
        "too lenient (alpha-engine-config-I11106)"
    )
    # Both bases persisted, under distinct keys, at BOTH manifest emission
    # sites (the S3 manifest and the returned run result).
    for key in ("xsec_variance_share_served_cut", "served_cut_mask",
                "xsec_incumbent_served_cut"):
        assert src.count(f'"{key}": ') >= 2, key
    # The gating key is fed from the SERVED-cut summary, never the fit panel.
    assert '_cand_xsec_sd = xsec_variance_share_served.get("xsec_sd")' in src
    assert '_fit_xsec_sd = xsec_variance_share_observe.get("xsec_sd")' in src
    assert '"xsec_sd_fit_panel": _fit_xsec_sd,' in src
