"""`inference_coverage`'s denominator is the set we were ASKED to score.

alpha-engine-config-I7648. `n_universe` is the research population, and the
predictor stopped scoring it at the champion cutover: measured 2026-08-18,
every one of the day's 24 predictions carried `watchlist_source` of
`attractiveness_top_20` (20) or `held` (4), while signals.json declared
`universe: 903`. The report card divided 23 by 903 and graded 2.5% against a
95% target — a CRITICAL component held permanently RED by a denominator
describing a design that no longer existed, and one of three stated reasons
live sizing was being kept de-risked.
"""
from __future__ import annotations

from unittest.mock import patch

from inference.stages.write_output import write_predictions


def _capture(**kwargs) -> dict:
    """Run write_predictions in dry-run and return the metrics body it built."""
    seen: dict = {}

    def _fake_dump(obj, *a, **k):
        if isinstance(obj, dict) and "n_predictions_today" in obj:
            seen.update(obj)
        return "{}"

    preds = [{"ticker": "AAPL", "prediction_confidence": 0.9}]
    with patch("inference.stages.write_output.json.dumps", side_effect=_fake_dump):
        write_predictions(preds, "2026-08-18", "bucket", {}, dry_run=True, **kwargs)
    return seen


def test_the_intended_denominator_is_written():
    m = _capture(
        n_universe=903, n_universe_covered=23,
        n_intended=24, n_intended_covered=23,
        intended_source=["attractiveness_top_20", "held"],
    )
    assert m["n_intended"] == 24
    assert m["n_intended_covered"] == 23
    assert m["intended_source"] == ["attractiveness_top_20", "held"]


def test_the_research_population_is_retained_not_replaced():
    """Funnel width is a real number — how narrow the funnel has become is
    worth seeing. It just is not the coverage denominator."""
    m = _capture(
        n_universe=903, n_universe_covered=23,
        n_intended=24, n_intended_covered=23,
        intended_source=["attractiveness_top_20"],
    )
    assert m["n_universe"] == 903
    assert m["n_universe_covered"] == 23


def test_absent_intended_is_none_and_never_falls_back_to_the_population():
    """A fallback here is how the metric would go on reporting the old number
    after the fix — a silent substitution invisible in the value."""
    m = _capture(n_universe=903, n_universe_covered=23)
    assert m["n_intended"] is None
    assert m["n_intended_covered"] is None
    assert m["intended_source"] is None
    assert m["n_universe"] == 903


# --------------------------------------------------------------------------
# intended_scoring_set — the derivation
# --------------------------------------------------------------------------

from inference.stages.write_output import intended_scoring_set  # noqa: E402


class _Ctx:
    def __init__(self, tickers=None, sources=None):
        self.tickers = tickers
        self.ticker_sources = sources


def test_the_live_2026_08_18_shape():
    """20 from the scanner's top-20, 4 held, 23 of the 24 scored."""
    tickers = [f"T{i}" for i in range(20)] + ["H1", "H2", "H3", "H4"]
    sources = {t: ("attractiveness_top_20" if t.startswith("T") else "held") for t in tickers}
    scored = {t.upper() for t in tickers[:-1]}
    n, covered, src = intended_scoring_set(_Ctx(tickers, sources), scored)
    assert (n, covered) == (24, 23)
    assert src == ["attractiveness_top_20", "held"]
    # The number the report card should have been grading all along.
    assert covered / n > 0.95


def test_a_genuine_miss_still_shows_as_a_miss():
    """The fix must not make the metric unable to go red. Handed 24, scored 5."""
    tickers = [f"T{i}" for i in range(24)]
    n, covered, _ = intended_scoring_set(
        _Ctx(tickers, {t: "attractiveness_top_20" for t in tickers}),
        {f"T{i}" for i in range(5)},
    )
    assert (n, covered) == (24, 5)
    assert covered / n < 0.8  # below the red-line, correctly


def test_case_and_blanks_are_normalized():
    n, covered, _ = intended_scoring_set(
        _Ctx(["aapl", "MSFT", "", None], {"aapl": "held"}), {"AAPL"}
    )
    assert (n, covered) == (2, 1)


def test_no_tickers_returns_all_none():
    for ctx in (_Ctx(None, None), _Ctx([], {}), _Ctx(["", None], None)):
        assert intended_scoring_set(ctx, {"AAPL"}) == (None, None, None)


def test_missing_sources_does_not_suppress_the_counts():
    """The counts are the load-bearing half; an unlabelled source degrades the
    description, not the denominator."""
    n, covered, src = intended_scoring_set(_Ctx(["AAPL", "MSFT"], None), {"AAPL"})
    assert (n, covered, src) == (2, 1, None)
