"""Contract tests for Rung-1 extraction META_FEATURES (alpha-engine-config#3080).

Verifies the 3 new extraction-agent feature names are present in both:
- META_FEATURES (the complete inference feature vector)
- RESEARCH_META_FEATURES (zeroed at inference when extraction agent hasn't run)

This is the consumer-contract half of the M0 birth rule (alpha-engine-config §126
policy). The producer half (the extraction agent itself) lives in
crucible-research/agents/sector_teams/rung1_extraction.py.
"""

from __future__ import annotations

import pytest

# The 3 Rung-1 extraction features — defined here as the shared truth for
# both producer and consumer contract tests.
RUNGG1_EXTRACTION_FEATURES = (
    "guidance_direction",
    "risk_factor_count_delta_raw",
    "management_tone_zscore",
)


class TestRung1ExtractionFeaturesInMetaFeatures:
    """The 3 extraction features must be present in the live meta-feature
    vector. A missing feature silently truncates the inference vector —
    the Ridge re-fit only catches up on next training."""

    def test_all_extraction_features_present(self):
        from model.meta_model import META_FEATURES

        for feat in RUNGG1_EXTRACTION_FEATURES:
            assert feat in META_FEATURES, (
                f"{feat} missing from META_FEATURES — inference will not "
                f"load values for this feature, silently truncating the meta vector."
            )

    def test_no_extraction_features_in_macro_features(self):
        """Extraction features are ticker-specific, not market-wide macros."""
        from model.meta_model import MACRO_FEATURE_NAMES

        for feat in RUNGG1_EXTRACTION_FEATURES:
            assert feat not in MACRO_FEATURE_NAMES

    def test_guidance_direction_is_categorical(self):
        """guidance_direction should be treated as a categorical feature, not
        a continuous one (the values are raised/lowered/maintained/none)."""
        from model.meta_model import META_DIRECTIONAL_FEATURES

        assert "guidance_direction" not in META_DIRECTIONAL_FEATURES, (
            "guidance_direction is categorical, not a continuous directional signal."
        )


class TestRung1ExtractionFeaturesInResearchFeatures:
    """The extraction features are research-derived — populated from the weekly
    research pipeline's extraction-agent state. They must be present in
    RESEARCH_META_FEATURES so the model-zoo can exclude them when the research
    signal join is disabled."""

    def test_all_extraction_features_in_research_features(self):
        from model.meta_model import RESEARCH_META_FEATURES

        for feat in RUNGG1_EXTRACTION_FEATURES:
            assert feat in RESEARCH_META_FEATURES, (
                f"{feat} missing from RESEARCH_META_FEATURES — it will not be "
                f"zeroed at inference when extraction data is absent."
            )


class TestRung1ExtractionFeatureDefaults:
    """When extraction data is absent (the extraction agent hasn't run yet),
    the features should default to safe neutral values — the fail-soft contract.
    These defaults must match what the research pipeline emits on error."""

    def test_the_declared_neutral_is_a_single_source(self):
        """The fail-soft value is declared once, so the training row build,
        the research-free training branch and inference cannot drift apart."""
        from model.meta_model import RESEARCH_META_NEUTRAL

        assert RESEARCH_META_NEUTRAL == 0.0

    def test_every_research_meta_feature_defaults_when_its_producer_is_absent(self):
        """THE REGRESSION (alpha-engine-config-I5949).

        `predictor#438` registered the 3 Rung-1 features and asserted a
        fail-soft contract in its commit message. The three tests that stood
        here asserted `True` and enforced nothing, so the contract was believed
        rather than held: the meta_X_all build indexed `r[f]` directly, and the
        first weekly training run after the merge died on
        `KeyError: 'guidance_direction'` — blocking the champion from
        retraining.

        Calls the PRODUCTION `build_meta_matrix`. Reverting its fail-soft
        branch to a bare `r[f]` makes this fail.
        """
        from model.meta_model import RESEARCH_META_FEATURES, RESEARCH_META_NEUTRAL
        from training.meta_trainer import build_meta_matrix

        columns = ["momentum_score", *RESEARCH_META_FEATURES]
        # A row as the producer leaves it when it has never run: the core
        # feature present, every research feature absent.
        rows = [{"momentum_score": 0.42}]

        built = build_meta_matrix(rows, columns)

        assert built.shape == (1, len(columns))
        assert built[0][0] == 0.42, "the core feature must be read, not defaulted"
        assert list(built[0][1:]) == [RESEARCH_META_NEUTRAL] * len(
            RESEARCH_META_FEATURES
        )

    def test_a_missing_core_meta_feature_still_raises(self):
        """The fail-soft default must NOT extend past RESEARCH_META_FEATURES.

        A core meta-feature absent from a row is an upstream contract breach.
        Defaulting it would train a champion on zeros and report success — the
        fail-loud rule inverted. This is the half of I5949 that a blanket
        `r.get(f, 0.0)` over all of TRAIN_META_FEATURES would have removed, and
        it is why the narrow fix is not just a stylistic preference.
        """
        from model.meta_model import RESEARCH_META_FEATURES, RESEARCH_META_NEUTRAL
        from training.meta_trainer import build_meta_matrix

        rows = [{f: RESEARCH_META_NEUTRAL for f in RESEARCH_META_FEATURES}]

        with pytest.raises(KeyError):
            build_meta_matrix(rows, ["momentum_score", *RESEARCH_META_FEATURES])

    def test_defaulting_is_reported_not_silent(self, caplog):
        """An all-zero column from an absent producer must be visible.

        Without this the champion trains on three inert columns and nothing
        says so — indistinguishable from those features being genuinely
        neutral (overseer-policy §7).
        """
        import logging

        from model.meta_model import RESEARCH_META_FEATURES
        from training.meta_trainer import build_meta_matrix

        columns = ["momentum_score", *RESEARCH_META_FEATURES]
        rows = [{"momentum_score": 0.1}, {"momentum_score": 0.2}]

        with caplog.at_level(logging.WARNING):
            build_meta_matrix(rows, columns)

        for f in RESEARCH_META_FEATURES:
            assert f in caplog.text, f"absent feature {f!r} was defaulted silently"
        assert "2/2" in caplog.text, "the warning must quantify the coverage gap"

    def test_a_present_research_feature_is_not_overwritten(self):
        """Fail-soft fills gaps; it must never clobber a real emitted value."""
        from model.meta_model import RESEARCH_META_FEATURES
        from training.meta_trainer import build_meta_matrix

        emitted = {f: 0.75 for f in RESEARCH_META_FEATURES}
        rows = [{"momentum_score": 0.42, **emitted}]

        built = build_meta_matrix(rows, ["momentum_score", *RESEARCH_META_FEATURES])

        assert list(built[0][1:]) == [0.75] * len(RESEARCH_META_FEATURES)

    def test_guidance_direction_has_no_numeric_encoding_yet(self):
        """`guidance_direction` is CATEGORICAL and its declared absent-value
        neutral is the string "none" (alpha-engine-config-I5818).

        RESEARCH_META_NEUTRAL (0.0) is the correct value while the producer is
        absent — it is exactly what inference feeds, so train and serve agree.
        It is NOT an encoding of "none". When the crucible-research Rung-1
        producer lands it must emit a numeric encoding, or this column enters
        a float matrix as a string and the array build breaks.

        Pinned so the open question stays visible rather than being read off
        the 0.0 as though it were settled.
        """
        from model.meta_model import META_DIRECTIONAL_FEATURES

        assert "guidance_direction" not in META_DIRECTIONAL_FEATURES


class TestRung1ExtractionFeatureOrdering:
    """Feature ordering must be stable to avoid silent misalignment of
    persisted Ridge weights vs the inference feature vector."""

    def test_extraction_features_appear_after_classic_research_features(self):
        from model.meta_model import META_FEATURES, RESEARCH_META_FEATURES

        # Find the index of the last classic research feature
        classic_end = max(
            META_FEATURES.index(f)
            for f in ["research_calibrator_prob", "research_composite_score",
                       "research_conviction", "sector_macro_modifier"]
        )
        # All 3 extraction features should be after the classic ones
        for feat in RUNGG1_EXTRACTION_FEATURES:
            assert META_FEATURES.index(feat) > classic_end, (
                f"{feat} appears before classic research features — ordering violation."
            )

    def test_extraction_features_order_is_stable(self):
        """The relative order of the 3 extraction features must not change
        between training and inference."""
        from model.meta_model import META_FEATURES
        indices = [META_FEATURES.index(f) for f in RUNGG1_EXTRACTION_FEATURES]
        # Must be strictly increasing
        assert indices == sorted(indices), (
            f"Extraction features out of order: {RUNGG1_EXTRACTION_FEATURES}"
        )
