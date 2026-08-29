"""alpha-engine-config-I9313 — the M-slot arm register.

The 2026-08-28 leaderboard is the fixture these are written against:
``horizon-60d`` scored CPCV IC 0.046929, passed the DSR gate, passed the
registry bar, and was refused with the bare reason ``non_canonical_horizon``.
``horizon-90d`` was separately dropped by ``selection_pbo`` as a
``dropped_misaligned_spec``. Two of four challengers were structurally
unable to win a promotion regardless of score, at the cost of a full weekly
training run each, and no artifact anywhere said why.

Each test below fails against `origin/main`.
"""
from __future__ import annotations

import pytest

from training.model_zoo_registry import (
    SLOT,
    applicable_spec_ids,
    arm_applicability,
    as_leaderboard_block,
    resolve_arms,
)

# The live 2026-08-28 zoo, as declared in predictor.yaml.
_LIVE_SPECS = [
    {"id": "residual-momentum", "model_version_label": "spec-residual-mom",
     "status": "active", "priority": 10,
     "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True, "MOMENTUM_L1_IN_META": False}},
    {"id": "horizon-60d", "model_version_label": "spec-60d", "status": "active",
     "overrides": {"FORWARD_DAYS": 60}},
    {"id": "horizon-90d", "model_version_label": "spec-90d", "status": "active",
     "overrides": {"FORWARD_DAYS": 90}},
    {"id": "sota-directional-combine", "model_version_label": "spec-sota-combine",
     "status": "active", "priority": 20,
     "overrides": {"META_STANDARDIZE_ENABLED": True}},
]


class TestTheStructuralExclusion:
    def test_the_two_horizon_arms_are_inapplicable_not_merely_ineligible(self):
        arms = {a.spec_id: a for a in resolve_arms(_LIVE_SPECS, canonical_horizon=21)}
        assert arms["horizon-60d"].applicability == "inapplicable"
        assert arms["horizon-90d"].applicability == "inapplicable"
        assert arms["residual-momentum"].applicability == "applicable"
        assert arms["sota-directional-combine"].applicability == "applicable"

    def test_an_inapplicable_arm_is_not_scheduled_at_all(self):
        """The cost half. Refused BEFORE training, not after scoring —
        champion-challenger-policy §4."""
        assert applicable_spec_ids(_LIVE_SPECS, canonical_horizon=21) == [
            "residual-momentum", "sota-directional-combine",
        ]

    def test_the_reason_travels_with_the_verdict(self):
        v = arm_applicability("horizon-60d", _LIVE_SPECS, canonical_horizon=21)
        assert v["applicability"] == "inapplicable"
        assert v["horizon_days"] == 60
        # the whole finding: `non_canonical_horizon` used to be a bare string
        assert "60-day forward horizon" in v["reason"]
        assert "canonical horizon is 21 days" in v["reason"]
        assert "horizon_battery" in v["reason"]

    def test_an_unregistered_arm_is_never_applicable_by_default(self):
        v = arm_applicability("some-shadow-writer", _LIVE_SPECS)
        assert v["applicability"] == "unregistered"
        assert "not in MODEL_SPECS" in v["reason"]

    def test_a_retired_arm_is_neither_scheduled_nor_applicable(self):
        specs = _LIVE_SPECS + [
            {"id": "old", "status": "retired", "retired_date": "2026-08-29"},
        ]
        arms = {a.spec_id: a for a in resolve_arms(specs, canonical_horizon=21)}
        assert arms["old"].applicability == "retired"
        assert arms["old"].retired_date == "2026-08-29"
        assert "old" not in applicable_spec_ids(specs, canonical_horizon=21)


class TestTheSlotContract:
    def test_the_slot_declares_every_field_policy_ss10_requires(self):
        for field in ("metric", "canonical_horizon_days", "benchmark",
                      "promote_margin_cfg", "promote_min_ic_cfg"):
            assert getattr(SLOT, field), field

    def test_the_horizon_ruling_is_recorded_with_its_reason(self):
        assert SLOT.horizon_policy == "refuse_non_canonical"
        assert "champion-challenger-policy §4" in SLOT.horizon_policy_reason
        assert "21-day hold" in SLOT.horizon_policy_reason
        # and the ruling names where the horizon question IS answered instead
        assert "horizon_battery" in SLOT.horizon_policy_alternative_measure

    def test_a_slot_wide_horizon_change_moves_every_arm_with_it(self):
        """The canonical horizon is resolved, not pinned: declaring the slot
        60d makes the 60d arm applicable and an explicitly-21d arm not.

        An arm with NO horizon override follows the slot by construction —
        that is why the absence of the key means canonical rather than 21.
        """
        specs = _LIVE_SPECS + [
            {"id": "pinned-21", "status": "active", "overrides": {"FORWARD_DAYS": 21}},
        ]
        arms = {a.spec_id: a for a in resolve_arms(specs, canonical_horizon=60)}
        assert arms["horizon-60d"].applicability == "applicable"
        assert arms["pinned-21"].applicability == "inapplicable"
        # the override-less arm follows the slot wherever it goes
        assert arms["residual-momentum"].horizon_days == 60
        assert arms["residual-momentum"].applicability == "applicable"


class TestTheAuditArtifact:
    def test_the_leaderboard_block_states_the_race_and_every_arm(self):
        block = as_leaderboard_block(_LIVE_SPECS, canonical_horizon=21)
        assert block["canonical_horizon_days"] == 21
        assert block["horizon_policy"] == "refuse_non_canonical"
        assert block["n_applicable"] == 2
        assert block["n_inapplicable"] == 2
        assert block["n_retired"] == 0
        by_id = {a["spec_id"]: a for a in block["arms"]}
        # every declared arm appears, including the ones that never ran —
        # "not in the race" must be distinguishable from "lost the race"
        assert set(by_id) == {a["id"] for a in _LIVE_SPECS}
        assert by_id["horizon-90d"]["trainable"] is False
        assert by_id["horizon-90d"]["reason"]
        assert by_id["residual-momentum"]["trainable"] is True
        assert by_id["residual-momentum"]["reason"] is None

    def test_the_block_survives_a_json_round_trip(self):
        import json

        block = as_leaderboard_block(_LIVE_SPECS, canonical_horizon=21)
        assert json.loads(json.dumps(block)) == block


class TestOneRegisterNotFour:
    def test_model_zoo_never_re_derives_membership_from_status(self):
        """The four independent membership decisions are down to one.

        ``model_zoo`` may still read ``status`` when building the parallel
        CANDIDATE pool (scoring a registered arm is policy §3's "every arm is
        scored every cycle"), but nothing that decides whether an arm is
        SCHEDULED or ELIGIBLE may re-derive it.
        """
        import inspect

        from training import model_zoo as mz

        for fn in (mz.train_all_active, mz.select_rotation_specs,
                   mz.train_weekly_rotation):
            src = inspect.getsource(fn)
            assert 'status", "active"' not in src, (
                f"{fn.__name__} re-derives slot membership from `status` "
                "instead of resolving it from the arm register (I9313)"
            )
            assert "model_zoo_registry" in src or "resolve_arms" in src \
                or "applicable_spec_ids" in src, fn.__name__

    def test_select_winner_resolves_the_horizon_verdict_from_the_register(self):
        import inspect

        from training import model_zoo as mz

        import ast
        import textwrap

        src = inspect.getsource(mz.select_winner)
        assert "arm_applicability" in src
        # AST, not substring: the comment explaining what was REMOVED names the
        # old expression, and a substring check would trip on the explanation.
        tree = ast.parse(textwrap.dedent(src))
        compares = [
            ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Compare)
        ]
        assert "fwd != champ_fwd" not in compares, (
            "the horizon exclusion is resolved from the register, not "
            "re-derived here (I9313)"
        )


class TestNoSilentDefaults:
    def test_a_spec_without_a_horizon_override_is_canonical(self):
        arms = resolve_arms(
            [{"id": "plain", "status": "active"}], canonical_horizon=21,
        )
        assert arms[0].horizon_days == 21
        assert arms[0].applicability == "applicable"

    def test_a_malformed_horizon_override_falls_back_to_canonical(self):
        """A non-integer FORWARD_DAYS is a config error, not licence to invent
        a horizon. It resolves to the slot's, which makes the arm APPLICABLE
        and therefore visible — never silently excluded."""
        arms = resolve_arms(
            [{"id": "bad", "status": "active", "overrides": {"FORWARD_DAYS": "sixty"}}],
            canonical_horizon=21,
        )
        assert arms[0].horizon_days == 21

    def test_a_spec_with_no_id_is_skipped_not_crashed(self):
        assert resolve_arms([{"status": "active"}]) == []


@pytest.mark.parametrize("spec_id", ["horizon-60d", "horizon-90d"])
def test_the_two_live_arms_that_could_never_win_now_cost_nothing(spec_id):
    """The bottom line of the 2026-08-28 leaderboard, asserted directly."""
    assert spec_id not in applicable_spec_ids(_LIVE_SPECS, canonical_horizon=21)
