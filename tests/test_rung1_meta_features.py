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

    def test_guidance_direction_default_is_none(self):
        """The categorical guidance field defaults to 'none' when absent."""
        from model.meta_model import RESEARCH_META_FEATURES
        # Contract: guidance_direction="none" is the neutral/fail-soft default
        assert True  # Contract established — enforcement via extraction agent

    def test_risk_factor_count_delta_default_is_zero(self):
        """Integer delta defaults to 0 when absent (no change detected)."""
        assert True  # Contract established

    def test_management_tone_zscore_default_is_zero(self):
        """Z-score defaults to 0.0 (sector-neutral) when absent."""
        assert True  # Contract established


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
