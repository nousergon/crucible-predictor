"""A degenerate arm must stop occupying the shadow rotation
(alpha-engine-config-I10180).

``SHADOW_VERSIONS_MAX_N`` is 5 and the five ``spec-sota-combine-2026-07-*``
arms — each emitting ONE predicted_alpha for 24 of 29 tickers, measured
2026-09-08 — filled it. Three consequences, all live: five rotation slots per
day not spent on an arm that could win; a realized-edge leaderboard scoring
constant-alpha output as if it were a model's; and (before
crucible-predictor-PR611) three ERROR pages a day attributed to the live path.

Two exclusion mechanisms, deliberately both:

* a DECLARED retirement (``model/retired_arms.py``) — the only thing that can
  reach a pre-2026-09-08 vintage, whose manifest carries no
  ``xsec_variance_share`` and for which the metric is not computable after the
  fact (its fitted panel is not persisted);
* a manifest-driven floor — the general rule, which arms itself with no code
  change the moment a vintage carries the metric.

Neither may raise ``SHADOW_VERSIONS_MAX_N``: the cap is the Lambda soft-timeout.
"""

from __future__ import annotations

import pytest

from inference.stages.shadow_versions import _select_challengers_for_cycle
from model.retired_arms import RETIRED_SHADOW_VERSIONS


def _v(vid):
    return {"version_id": vid, "date": "2026-09-04", "created_utc": "z"}


DEGENERATE = "spec-sota-combine-2026-07-24-8578f8ae"
HEALTHY = [_v(f"v3.0-meta-2026-09-0{i}-aaaa000{i}") for i in range(1, 6)]


def test_a_declared_retired_arm_is_never_selected():
    pool = [_v(vid) for vid in RETIRED_SHADOW_VERSIONS] + HEALTHY
    picked = {v["version_id"] for v in _select_challengers_for_cycle(
        pool, 5, "2026-09-08",
    )}
    assert not (picked & set(RETIRED_SHADOW_VERSIONS))
    assert len(picked) == 5


def test_the_five_retired_arms_free_the_whole_rotation():
    """The measured starting state: five degenerate arms and five healthy ones,
    a cap of five. Before this, the rotation could hand a whole day to the
    degenerate half."""
    pool = [_v(vid) for vid in RETIRED_SHADOW_VERSIONS] + HEALTHY
    for day in ("2026-09-08", "2026-09-09", "2026-09-10"):
        picked = {v["version_id"] for v in _select_challengers_for_cycle(pool, 5, day)}
        assert picked == {v["version_id"] for v in HEALTHY}


def test_a_below_floor_xsec_variance_share_manifest_is_not_selected():
    """The general rule, for vintages that DO carry the metric."""
    pool = HEALTHY + [_v(DEGENERATE)]
    manifests = {
        DEGENERATE: {"behavioral_metrics": {"xsec_variance_share": 0.0046}},
    }
    picked = {v["version_id"] for v in _select_challengers_for_cycle(
        pool, 5, "2026-09-08", manifests_by_vid=manifests,
    )}
    assert DEGENERATE not in picked


def test_an_above_floor_manifest_is_still_eligible():
    other = "v3.0-meta-2026-08-28-01cf7e1a"
    pool = HEALTHY[:2] + [_v(other)]
    manifests = {other: {"behavioral_metrics": {"xsec_variance_share": 0.98}}}
    picked = {v["version_id"] for v in _select_challengers_for_cycle(
        pool, 5, "2026-09-08", manifests_by_vid=manifests,
    )}
    assert other in picked


def test_an_absent_metric_does_not_exclude_an_arm():
    """Absent is uncomputable, not degenerate. Excluding on absence would empty
    the rotation of every pre-2026-09-08 vintage at once, including healthy
    ones — which is why the five known-bad arms are DECLARED instead."""
    other = "v3.0-meta-2026-08-28-01cf7e1a"
    pool = HEALTHY[:2] + [_v(other)]
    picked = {v["version_id"] for v in _select_challengers_for_cycle(
        pool, 5, "2026-09-08", manifests_by_vid={other: {}},
    )}
    assert other in picked


def test_an_unreadable_manifest_does_not_exclude_an_arm():
    """A read failure must not starve the rotation — that would be a second
    detector-blindness defect wearing a fix's name. It is logged instead."""
    other = "v3.0-meta-2026-08-28-01cf7e1a"
    pool = HEALTHY[:2] + [_v(other)]
    picked = {v["version_id"] for v in _select_challengers_for_cycle(
        pool, 5, "2026-09-08", manifests_by_vid={other: None},
    )}
    assert other in picked


def test_exclusion_happens_before_the_cap_not_after():
    """Filtering after the window would still lose the freed slots: the
    rotation would pick five, drop the retired ones, and shadow fewer arms than
    it could."""
    pool = [_v(vid) for vid in RETIRED_SHADOW_VERSIONS] + HEALTHY
    assert len(_select_challengers_for_cycle(pool, 5, "2026-09-08")) == 5


def test_rotation_is_still_idempotent_per_date():
    pool = HEALTHY + [_v(f"x-{i}") for i in range(8)]
    a = _select_challengers_for_cycle(pool, 5, "2026-09-08")
    b = _select_challengers_for_cycle(pool, 5, "2026-09-08")
    assert [v["version_id"] for v in a] == [v["version_id"] for v in b]


def test_the_cap_is_never_raised_to_absorb_a_degenerate_arm():
    import config as cfg

    assert int(getattr(cfg, "SHADOW_VERSIONS_MAX_N", 5)) <= 5, (
        "the cap is the Lambda soft-timeout, and the shadow stage already skips "
        "remaining challengers near it — raising it is explicitly not the fix"
    )
