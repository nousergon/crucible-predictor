"""Tests for regime/retrospective_eval.py — Stage C.2 T1.

Covers:
- 4-class → 3-class agent label normalization (caution → bear)
- HMM smoother fit + labels at requested eval indices
- Agent ↔ retrospective pairing (skip unparseable/missing rows)
- Symmetric vs asymmetric agreement scoring (bear-miss weight)
- Rolling-window-N score
- Confusion matrix
- Payload assembly into canonical eval-artifact shape
- Pin: only the smoother is called retrospectively; the filter path
  stays untouched (look-ahead discipline)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("hmmlearn")


from regime.retrospective_eval import (
    DEFAULT_BEAR_MISS_WEIGHT,
    DEFAULT_LAG_WEEKS,
    DEFAULT_ROLLING_WEEKS,
    RegimePairing,
    _normalize_agent_label,
    assemble_t1_eval_payload,
    compute_retrospective_labels,
    pair_agent_calls_with_retrospective,
    score_agreement,
)


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


class TestNormalizeAgentLabel:
    @pytest.mark.parametrize("inp,expected", [
        ("bull", "bull"),
        ("BULL", "bull"),
        (" Bull ", "bull"),
        ("neutral", "neutral"),
        ("caution", "bear"),
        ("CAUTION", "bear"),
        ("bear", "bear"),
    ])
    def test_known_labels(self, inp, expected):
        assert _normalize_agent_label(inp) == expected

    @pytest.mark.parametrize("inp", [None, "", "unknown", "BULLISH", 42])
    def test_unparseable_returns_none(self, inp):
        assert _normalize_agent_label(inp) is None


# ---------------------------------------------------------------------------
# Retrospective label computation (HMM smoother)
# ---------------------------------------------------------------------------


def _synthetic_two_regime_features(n_per_half: int = 80, seed: int = 7) -> pd.DataFrame:
    """Two-regime synthetic feature panel — calm first half, stressed
    second half. Mirrors the predictor's test_regime_hmm + test_regime_substrate
    fixtures so the HMM has enough state separation."""
    rng = np.random.default_rng(seed)
    calm = pd.DataFrame({
        "spy_20d_return": rng.normal(0.02, 0.012, n_per_half),
        "vix_level": rng.normal(13.0, 1.8, n_per_half),
        "hy_oas_bps": rng.normal(280.0, 35.0, n_per_half),
    })
    stressed = pd.DataFrame({
        "spy_20d_return": rng.normal(-0.05, 0.018, n_per_half),
        "vix_level": rng.normal(32.0, 4.0, n_per_half),
        "hy_oas_bps": rng.normal(700.0, 80.0, n_per_half),
    })
    df = pd.concat([calm, stressed], ignore_index=True)
    df.index = pd.date_range("2020-01-03", periods=len(df), freq="W-FRI")
    return df


class TestComputeRetrospectiveLabels:
    def test_returns_labels_for_each_eval_index(self):
        df = _synthetic_two_regime_features()
        feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
        # Evaluate a few interior indices, well within the smoothing window
        eval_indices = [40, 80, 120]
        labels = compute_retrospective_labels(
            df, hmm_feature_columns=feat_cols, eval_indices=eval_indices,
        )
        assert set(labels.keys()) == set(eval_indices)
        for idx, label in labels.items():
            assert label in ("bear", "neutral", "bull"), (idx, label)

    def test_smoother_separates_calm_from_stressed_halves(self):
        """On a clean two-regime synthetic, the smoothed argmax in the
        calm half should be bull/neutral; in the stressed half bear/neutral.
        Pins that the smoother is actually fitting + classifying, not
        returning a degenerate single state."""
        df = _synthetic_two_regime_features(n_per_half=100)
        feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
        # Sample across both halves
        eval_indices = list(range(20, 180, 20))
        labels = compute_retrospective_labels(
            df, hmm_feature_columns=feat_cols, eval_indices=eval_indices,
        )
        # Indices < 100 → calm half (expect bull/neutral)
        calm_labels = [labels[i] for i in eval_indices if i < 100]
        stressed_labels = [labels[i] for i in eval_indices if i >= 100]
        bear_in_calm = sum(1 for l in calm_labels if l == "bear")
        bear_in_stressed = sum(1 for l in stressed_labels if l == "bear")
        # Smoother should put MORE bears in the stressed half
        assert bear_in_stressed > bear_in_calm

    def test_raises_when_history_too_short(self):
        df = _synthetic_two_regime_features(n_per_half=20)
        feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
        # 40 rows is below HMMRegimeClassifier's 60-row minimum for stable
        # EM convergence; we expect a ValueError on either guard. The
        # retrospective-eval module's own "needs at least N rows" check
        # is a defensive backstop after the HMM-internal threshold; both
        # indicate the same root cause (history too short for valid
        # smoothing) so we accept either message.
        with pytest.raises(ValueError, match="(needs at least|Insufficient observations)"):
            compute_retrospective_labels(
                df, hmm_feature_columns=feat_cols, eval_indices=[39],
            )

    def test_empty_eval_indices_returns_empty(self):
        df = _synthetic_two_regime_features()
        feat_cols = ["spy_20d_return", "vix_level", "hy_oas_bps"]
        assert compute_retrospective_labels(
            df, hmm_feature_columns=feat_cols, eval_indices=[],
        ) == {}


# ---------------------------------------------------------------------------
# Pairing agent calls with retrospective labels
# ---------------------------------------------------------------------------


def _agent_calls_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"trading_day": [pd.Timestamp(d) for d, _ in rows],
         "market_regime": [r for _, r in rows]}
    )


class TestPairAgentCallsWithRetrospective:
    def test_happy_path(self):
        agent_df = _agent_calls_df([
            ("2024-01-05", "bull"),
            ("2024-01-12", "caution"),
            ("2024-01-19", "bear"),
        ])
        retrospective = {
            pd.Timestamp("2024-01-05"): "bull",     # agree
            pd.Timestamp("2024-01-12"): "neutral",  # disagree (caution → bear vs neutral)
            pd.Timestamp("2024-01-19"): "bear",     # agree
        }
        pairings = pair_agent_calls_with_retrospective(
            agent_calls=agent_df, retrospective_labels=retrospective,
        )
        assert len(pairings) == 3
        # First pairing — bull/bull agreement
        assert pairings[0].agreed is True
        assert pairings[0].was_bear_miss is False
        assert pairings[0].was_bull_miss is False
        # Second pairing — caution agent (normalized to bear) vs neutral retro
        assert pairings[1].agent_call_normalized == "bear"
        assert pairings[1].agreed is False
        assert pairings[1].was_bear_miss is False  # truth is NOT bear
        assert pairings[1].was_bull_miss is True   # agent IS bear, truth NOT bear
        # Third pairing — bear/bear agreement
        assert pairings[2].agreed is True

    def test_skips_unparseable_agent_label(self):
        agent_df = _agent_calls_df([
            ("2024-01-05", "bull"),
            ("2024-01-12", "wat"),  # unparseable
        ])
        retrospective = {
            pd.Timestamp("2024-01-05"): "bull",
            pd.Timestamp("2024-01-12"): "neutral",
        }
        pairings = pair_agent_calls_with_retrospective(
            agent_calls=agent_df, retrospective_labels=retrospective,
        )
        assert len(pairings) == 1
        assert pairings[0].trading_day == pd.Timestamp("2024-01-05")

    def test_skips_missing_retrospective(self):
        agent_df = _agent_calls_df([
            ("2024-01-05", "bull"),
            ("2024-01-12", "bear"),  # no retrospective for this date
        ])
        retrospective = {pd.Timestamp("2024-01-05"): "bull"}
        pairings = pair_agent_calls_with_retrospective(
            agent_calls=agent_df, retrospective_labels=retrospective,
        )
        assert len(pairings) == 1

    def test_bear_miss_detected(self):
        """The asymmetric metric's key signal: agent called bull but
        retrospective truth is bear → was_bear_miss=True. This is the
        worst failure mode in the business-cost sense."""
        agent_df = _agent_calls_df([("2024-01-05", "bull")])
        retrospective = {pd.Timestamp("2024-01-05"): "bear"}
        pairings = pair_agent_calls_with_retrospective(
            agent_calls=agent_df, retrospective_labels=retrospective,
        )
        assert pairings[0].agreed is False
        assert pairings[0].was_bear_miss is True
        assert pairings[0].was_bull_miss is False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _pairing(td: str, agent_norm: str, retro: str) -> RegimePairing:
    return RegimePairing(
        trading_day=pd.Timestamp(td),
        agent_call=agent_norm,
        agent_call_normalized=agent_norm,
        retrospective_label=retro,
        agreed=(agent_norm == retro),
        was_bear_miss=(retro == "bear" and agent_norm != "bear"),
        was_bull_miss=(agent_norm == "bear" and retro != "bear"),
    )


class TestScoreAgreement:
    def test_empty_pairings_returns_none_metrics(self):
        result = score_agreement([])
        assert result["n_pairings"] == 0
        assert result["symmetric_agreement_rate"] is None
        assert result["asymmetric_weighted_agreement_rate"] is None
        assert result["rolling_window_score"] is None

    def test_perfect_agreement(self):
        pairings = [
            _pairing("2024-01-05", "bull", "bull"),
            _pairing("2024-01-12", "neutral", "neutral"),
            _pairing("2024-01-19", "bear", "bear"),
        ]
        result = score_agreement(pairings)
        assert result["symmetric_agreement_rate"] == pytest.approx(1.0)
        assert result["asymmetric_weighted_agreement_rate"] == pytest.approx(1.0)
        assert result["bear_miss_count"] == 0
        assert result["bull_miss_count"] == 0

    def test_bear_miss_penalized_more_than_other_miss(self):
        """Two pairings, one bear-miss and one bull-miss. Symmetric rate
        treats them equally (both 50%). Asymmetric rate penalizes
        bear-miss more heavily."""
        pairings_bear_miss = [_pairing("2024-01-05", "bull", "bear")]
        pairings_bull_miss = [_pairing("2024-01-05", "bear", "bull")]
        r_bear = score_agreement(pairings_bear_miss, bear_miss_weight=2.0)
        r_bull = score_agreement(pairings_bull_miss, bear_miss_weight=2.0)
        # Symmetric: both 0% agreement
        assert r_bear["symmetric_agreement_rate"] == 0.0
        assert r_bull["symmetric_agreement_rate"] == 0.0
        # Asymmetric: bear-miss costs 2.0 (max), bull-miss costs 1.0 (half max)
        assert r_bear["asymmetric_weighted_agreement_rate"] == pytest.approx(0.0)
        assert r_bull["asymmetric_weighted_agreement_rate"] == pytest.approx(0.5)

    def test_rolling_window_takes_most_recent_n(self):
        # 30 pairings, mix of agreement; rolling window 10 should only
        # see the most recent 10
        pairings = []
        for i in range(20):
            pairings.append(_pairing(f"2024-{(i % 12) + 1:02d}-{((i % 27) + 1):02d}",
                                     "bull", "bull"))  # all agree
        for i in range(10):
            # Bear-miss in the recent window
            pairings.append(_pairing(f"2025-{((i % 12) + 1):02d}-{((i % 27) + 1):02d}",
                                     "bull", "bear"))
        result = score_agreement(pairings, rolling_window_weeks=10)
        # Headline rolling score uses the recent 10 (all bear-misses)
        assert result["rolling_window_size"] == 10
        assert result["rolling_window_score"] == pytest.approx(0.0)
        # Symmetric over ALL 30 — 20/30 agreed
        assert result["symmetric_agreement_rate"] == pytest.approx(20 / 30)

    def test_confusion_matrix_counts_pairings_correctly(self):
        pairings = [
            _pairing("2024-01-05", "bull", "bull"),
            _pairing("2024-01-12", "bull", "neutral"),
            _pairing("2024-01-19", "bear", "bear"),
            _pairing("2024-01-26", "bear", "bear"),
        ]
        result = score_agreement(pairings)
        cm = result["confusion_matrix"]
        assert cm["bull"]["bull"] == 1
        assert cm["bull"]["neutral"] == 1
        assert cm["bear"]["bear"] == 2

    def test_bear_miss_weight_at_one_collapses_to_symmetric(self):
        """When bear_miss_weight=1.0 the asymmetric metric should
        reduce to symmetric (all misses equally weighted)."""
        pairings = [
            _pairing("2024-01-05", "bull", "bear"),
            _pairing("2024-01-12", "bear", "bull"),
            _pairing("2024-01-19", "bull", "bull"),
        ]
        result = score_agreement(pairings, bear_miss_weight=1.0)
        # Symmetric: 1/3 agreement
        assert result["symmetric_agreement_rate"] == pytest.approx(1 / 3)
        # Asymmetric reduces to symmetric when weight=1.0
        assert result["asymmetric_weighted_agreement_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


class TestAssembleT1EvalPayload:
    def test_payload_shape(self):
        pairings = [
            _pairing("2024-01-05", "bull", "bull"),
            _pairing("2024-01-12", "bear", "bear"),
        ]
        score = score_agreement(pairings)
        payload = assemble_t1_eval_payload(
            pairings=pairings,
            score=score,
            run_id="2605170230",
            calendar_date="2026-05-17",
            trading_day="2026-05-15",
            feature_window_start="2020-01-03",
            feature_window_end="2024-01-12",
            hmm_feature_columns=["spy_20d_return", "vix_level", "hy_oas_bps"],
        )
        # Canonical eval-artifact fields
        assert payload["calendar_date"] == "2026-05-17"
        assert payload["trading_day"] == "2026-05-15"
        assert payload["run_id"] == "2605170230"
        assert payload["schema_version"] == 1
        assert payload["eval_tier"] == "T1_retrospective_hmm_smoothing"
        # Score block embedded
        assert payload["score"]["symmetric_agreement_rate"] == pytest.approx(1.0)
        # Pairings serialized in chronological order
        assert payload["pairings"][0]["trading_day"] == "2024-01-05T00:00:00"
        assert payload["pairings"][1]["trading_day"] == "2024-01-12T00:00:00"
        # Model metadata pins the look-ahead discipline + asymmetric convention
        md = payload["model_metadata"]
        assert md["smoother_algorithm"] == "forward_backward_hamilton_kim"
        assert "bear_miss=2.0" in md["asymmetric_loss_convention"]


# ---------------------------------------------------------------------------
# Default-value pins (cheap regression test for tunable thresholds)
# ---------------------------------------------------------------------------


class TestDefaultsPins:
    def test_bear_miss_weight_default_is_two(self):
        """Defaults are in the contract — if someone bumps this, the
        rolling score's headline number shifts, and the dashboard's
        threshold should move in lockstep."""
        assert DEFAULT_BEAR_MISS_WEIGHT == 2.0

    def test_lag_weeks_default_is_eight(self):
        """8-week lag is the smoother-credibility threshold per
        regime-v3-260514.md §5.3.3."""
        assert DEFAULT_LAG_WEEKS == 8

    def test_rolling_window_default_is_26(self):
        """~6 months — half-year baseline once the historical pairing
        set is populated."""
        assert DEFAULT_ROLLING_WEEKS == 26
