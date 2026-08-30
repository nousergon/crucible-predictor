"""alpha-engine-config-I9319 / -I9322 — the model (M) slot on the shared arena.

Every test here is written against a defect that was live on 2026-08-29:

* ``PredictorTraining`` returned SSM ``Status: Success``, all four zoo specs
  reported OK, ``ModelZooSelect`` wrote a complete leaderboard and
  ``branch_b_degraded`` was ``false`` — over fits with SEVEN features hard
  zeroed. Every surface said healthy. ``TrainingIntegrityError`` must now make
  that week a FAILED task (``TestImproperTrainingFailsTheTask``).
* ``promoted_kind: champion-arch-refresh`` carried every predictor transition
  but one, ever. It must be structurally unwritable
  (``TestTheAbolishedPromotionKind``).
* The pointer was decided on promotion-time CPCV mean IC, measured to have no
  relationship with realized 21d rank-IC (Spearman +0.03, p=0.90). It now ranks
  on realized market-relative rank-IC (``TestTheRankingStatistic``).
* Minimum-week and minimum-cohort bars gated the decision path
  (``TestNoEvidenceBars``).

`champion-challenger-policy.md` §7.4 — a guard must be shown RED against
pre-fix code before it is trusted. Each class below names, in its docstring,
the pre-fix behaviour it was run against.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "nousergon_lib.arena",
    reason=(
        "nousergon_lib.arena landed in v0.124.98; this repo's pin is bumped to "
        "v0.124.99 in the same PR. A local env on an older lib skips."
    ),
)

from nousergon_lib.arena.arms import ArmRegister, derive_arm_id  # noqa: E402
from nousergon_lib.arena.engine import (  # noqa: E402
    ServingPrecondition,
    TrainingIntegrityError,
)
from nousergon_lib.arena.window import ArmSeries  # noqa: E402

from training import arena_model_slot as ams  # noqa: E402


# ── fixtures: SHAPES, never a restatement of the live roster ────────────────
#
# Three fixtures rotted this session by hardcoding a live registry as a literal
# and then drifting from it silently. These specs are deliberately synthetic
# and are exercised as INPUTS to the same `resolve_arms` derivation production
# uses; `TestTheRegisterIsDerivedNotRestated` separately asserts that the live
# `config.MODEL_SPECS` flows through the same path, so a roster change cannot
# leave these tests green against a roster that no longer exists.

_SPECS = [
    {"id": "alpha-arm", "model_version_label": "lbl-alpha", "status": "active",
     "overrides": {"META_STANDARDIZE_ENABLED": True}},
    {"id": "beta-arm", "model_version_label": "lbl-beta", "status": "active",
     "overrides": {"RESIDUAL_MOMENTUM_ENABLED": True}},
    {"id": "off-horizon", "model_version_label": "lbl-60", "status": "active",
     "overrides": {"FORWARD_DAYS": 60}},
]


def _arm(spec_id, label, horizon=21, overrides=None, cadence="weekly"):
    class _A:
        pass
    a = _A()
    a.spec_id, a.model_version_label, a.horizon_days = spec_id, label, horizon
    a.overrides = overrides or {}
    a.applicability, a.status, a.reason, a.retired_date = "applicable", "active", "", None
    return a


def _versions(*rows):
    return [
        {"version_id": vid, "model_version": label, "date": date,
         "created_utc": f"{date}T12:00:00Z", "stage": "challenger"}
        for vid, label, date in rows
    ]


class _FakeS3:
    """Enough S3 for the register + emitter. Records every PUT."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        body = self.objects[Key]

        class _B:
            def read(self_inner):
                return body if isinstance(body, bytes) else json.dumps(body).encode()

        return {"Body": _B()}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.puts[Key] = json.loads(Body)
        self.objects[Key] = Body


# ── the recipe boundary ─────────────────────────────────────────────────────

class TestTheRecipeBoundary:
    """RED before the fix: there was no register at all — the pointer was
    decided per ``version_id``, which changes on every fit because it hashes
    the contract files' S3 ETags. Every Saturday minted a brand-new identity
    with an empty record, which is why 100% of transitions were
    ``champion-arch-refresh``."""

    def test_a_refit_changes_no_id_and_resets_no_series(self):
        arm = _arm("alpha-arm", "lbl-alpha")
        spec = ams.recipe_spec(arm, cadence="weekly")
        first = derive_arm_id(ams.SLOT, "alpha-arm", spec)
        # A second week. Same recipe, different fitted bundle.
        second = derive_arm_id(ams.SLOT, "alpha-arm", ams.recipe_spec(arm, cadence="weekly"))
        assert first == second

    def test_a_changed_recipe_is_a_different_arm(self):
        base = _arm("alpha-arm", "lbl-alpha", overrides={"A": 1})
        changed = _arm("alpha-arm", "lbl-alpha", overrides={"A": 2})
        assert (
            derive_arm_id(ams.SLOT, "alpha-arm", ams.recipe_spec(base, cadence="weekly"))
            != derive_arm_id(ams.SLOT, "alpha-arm", ams.recipe_spec(changed, cadence="weekly"))
        )

    def test_the_refit_cadence_is_part_of_the_recipe(self):
        """§3.1 names cadence as a fixed field of the recipe. It is not
        persisted anywhere per-arm today, so it is resolved from live rotation
        config and hashed in — a cadence change must not inherit a record
        earned under a different one."""
        arm = _arm("alpha-arm", "lbl-alpha")
        assert (
            ams.recipe_spec(arm, cadence="weekly")["refit_cadence"]
            != ams.recipe_spec(arm, cadence="round-robin:every-2-weeks")["refit_cadence"]
        )
        assert (
            derive_arm_id(ams.SLOT, "alpha-arm", ams.recipe_spec(arm, cadence="weekly"))
            != derive_arm_id(ams.SLOT, "alpha-arm",
                             ams.recipe_spec(arm, cadence="round-robin:every-2-weeks"))
        )

    def test_the_recipe_carries_nothing_that_varies_between_refits(self):
        spec = ams.recipe_spec(_arm("alpha-arm", "lbl-alpha"), cadence="weekly")
        flat = json.dumps(spec)
        for forbidden in ("version_id", "created_utc", "cpcv", "date"):
            assert forbidden not in flat, forbidden

    def test_a_dict_ordering_change_cannot_mint_a_phantom_arm(self):
        a = _arm("x", "lbl", overrides={"A": 1, "B": 2})
        b = _arm("x", "lbl", overrides={"B": 2, "A": 1})
        assert (ams.recipe_spec(a, cadence="weekly")
                == ams.recipe_spec(b, cadence="weekly"))


class TestTheRegisterIsDerivedNotRestated:
    """RED before the fix: membership was re-derived in four places from
    ``status`` and a horizon comparison, and they disagreed (I9313)."""

    def test_the_slot_roster_comes_from_the_live_spec_register(self):
        arms = {a.spec_id for a in ams.slot_arms(_SPECS, canonical_horizon=21)}
        # the off-horizon arm is not in the M slot at all
        assert arms == {"alpha-arm", "beta-arm", ams.BASE_ARCH_ARM}

    def test_the_base_arch_is_an_ordinary_arm_of_the_slot(self):
        """The whole reason ``champion-arch-refresh`` can be abolished: the
        base architecture is an arm, and its weekly retrain is a refit."""
        arms = {a.spec_id: a for a in ams.slot_arms(_SPECS, canonical_horizon=21)}
        assert arms[ams.BASE_ARCH_ARM].applicability == "applicable"

    def test_the_live_roster_flows_through_the_same_derivation(self):
        """The anti-rot guard. Whatever ``predictor.yaml`` declares today must
        resolve through the same path these fixtures exercise — so a roster
        change cannot leave this file green against a roster that is gone."""
        import config as cfg
        live = getattr(cfg, "MODEL_SPECS", []) or []
        if not live:
            pytest.skip("no MODEL_SPECS in this environment (public checkout)")
        resolved = {a.spec_id for a in ams.slot_arms(live)}
        declared = {s.get("id") for s in live if s.get("id")}
        assert resolved - {ams.BASE_ARCH_ARM} <= declared
        assert ams.BASE_ARCH_ARM in resolved

    def test_a_declared_spec_with_no_bundle_is_not_yet_an_arm(self):
        s3 = _FakeS3()
        register, by_label = ams.build_register(
            s3, "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=[], register=ArmRegister(),
        )
        assert register.all_arms() == ()

    def test_every_bundle_after_the_first_is_recorded_as_a_refit(self):
        versions = _versions(
            ("v-a-1", "lbl-alpha", "2026-08-08"),
            ("v-a-2", "lbl-alpha", "2026-08-15"),
            ("v-a-3", "lbl-alpha", "2026-08-22"),
        )
        register, by_label = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=versions, register=ArmRegister(),
        )
        arm_id = by_label["lbl-alpha"]
        assert register.state(arm_id).record.created_date == "2026-08-08"
        assert register.refits(arm_id) == ("2026-08-15", "2026-08-22")

    def test_a_changed_recipe_supersedes_and_the_old_arm_keeps_its_record(self):
        versions = _versions(("v-a-1", "lbl-alpha", "2026-08-08"))
        register, by_label = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=versions, register=ArmRegister(),
        )
        old = by_label["lbl-alpha"]
        edited = [dict(s) for s in _SPECS]
        edited[0]["overrides"] = {"META_STANDARDIZE_ENABLED": False}
        register2, by_label2 = ams.build_register(
            _FakeS3(), "b", as_of="2026-09-05", specs=edited, canonical_horizon=21,
            versions=versions, register=register,
        )
        new = by_label2["lbl-alpha"]
        assert new != old
        assert register2.state(new).record.supersedes == old
        # the superseded arm is retired, but its record and series survive (§6.3)
        assert register2.state(old).retired_date == "2026-09-05"
        assert old in register2.all_arms()


# ── improper training fails the TASK ────────────────────────────────────────

class TestImproperTrainingFailsTheTask:
    """RED before the fix: the 2026-08-29 rotation ran to completion and wrote
    a full leaderboard over fits with seven features hard-zeroed. Nothing
    raised, and every surface reported success."""

    def _register(self):
        versions = _versions(
            ("v-a", "lbl-alpha", "2026-08-22"),
            ("v-b", "lbl-beta", "2026-08-22"),
            ("v-c", "v3.0-meta", "2026-08-22"),
        )
        return ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=versions, register=ArmRegister(),
        )

    def test_the_2026_08_29_condition_raises(self):
        """Seven constant input columns is exactly ``arm_validity``'s
        ``constant_input_column`` failure. The cycle must not run."""
        register, by_label = self._register()
        manifests = {
            by_label["lbl-alpha"]: {
                "arm_validity": {"failures": [
                    {"reason": "constant_input_column — 7 of 34 meta features "
                               "have ZERO variance across the whole panel"},
                ]},
            },
            by_label["lbl-beta"]: {"arm_validity": {"failures": []}},
            by_label["v3.0-meta"]: {"arm_validity": {"failures": []}},
        }
        statuses = ams.training_statuses(manifests)
        assert statuses[by_label["lbl-alpha"]].ok is False
        assert "constant_input_column" in statuses[by_label["lbl-alpha"]].reason

        with pytest.raises(TrainingIntegrityError) as exc:
            ams.run_model_slot_cycle(
                as_of="2026-08-29", register=register,
                series_by_arm={a: ArmSeries(arm_id=a, scores={}) for a in register.all_arms()},
                incumbent=None, preconditions={}, training=statuses,
            )
        assert "not trained properly" in str(exc.value)

    def test_an_unasserted_fit_is_treated_as_a_failed_fit(self):
        """The half that is easy to miss: an arm nobody vouched for."""
        register, by_label = self._register()
        statuses = ams.training_statuses({by_label["lbl-alpha"]: {"arm_validity": {}}})
        with pytest.raises(TrainingIntegrityError) as exc:
            ams.run_model_slot_cycle(
                as_of="2026-08-29", register=register,
                series_by_arm={a: ArmSeries(arm_id=a, scores={}) for a in register.all_arms()},
                incumbent=None, preconditions={}, training=statuses,
            )
        assert "no training status reported" in str(exc.value)

    def test_an_unreadable_manifest_yields_no_status_not_a_pass(self):
        assert ams.training_statuses({"M:x:abc": None}) == {}

    def test_l1_fit_failures_reach_the_status(self):
        st = ams.training_statuses({"M:x:abc": {
            "l1_fit_validity": {"failures": [{"reason": "research_gbm best_iteration=2"}]},
        }})
        assert st["M:x:abc"].ok is False
        assert "research_gbm" in st["M:x:abc"].reason

    def test_degraded_inputs_are_an_unsound_fit(self):
        st = ams.training_statuses({"M:x:abc": {"data_coverage_degraded": True}})
        assert st["M:x:abc"].ok is False


class TestTheStepFunctionSeesAFailure:
    """The ruling's teeth reach the pipeline. ``ModelZooSelect`` runs
    ``infrastructure/spot_model_zoo_select.sh``, which invokes
    ``run_select_only`` on an EC2 spot box — so what makes the SF state FAIL is
    the exception propagating out of the Python entrypoint to a non-zero exit,
    NOT a caught-and-recorded status field.

    RED before the fix: ``select_and_finalize`` had no arena call at all, so
    there was nothing to propagate and the 2026-08-29 run exited 0."""

    def test_training_integrity_is_not_caught_anywhere_on_the_select_path(self):
        import inspect

        from training import model_zoo as mz

        for fn in (mz.select_and_finalize, mz.run_select_only,
                   mz.run_rotation_and_select):
            src = inspect.getsource(fn)
            assert "TrainingIntegrityError" not in src, (
                f"{fn.__name__} names TrainingIntegrityError — the ruling is "
                "that it FAILS THE TASK (champion-challenger-policy §3, §11). "
                "Catching it, even to re-record it, is the escape hatch the "
                "ruling forbids."
            )

    def test_the_arena_is_invoked_before_the_promote(self):
        import inspect

        from training import model_zoo as mz

        src = inspect.getsource(mz.select_and_finalize)
        assert "arena_model_slot" in src and "run_slot" in src
        assert src.index("run_slot") < src.index("promote_to_champion")

    def test_the_spot_entrypoint_propagates_a_nonzero_exit(self):
        """The shell wrapper must not swallow the Python exit code."""
        from pathlib import Path

        script = (Path(__file__).resolve().parent.parent
                  / "infrastructure" / "spot_model_zoo_select.sh")
        if not script.exists():
            pytest.skip("spot_model_zoo_select.sh not present in this checkout")
        body = script.read_text()
        assert "|| true" not in body, (
            "the select entrypoint swallows its exit code, so a "
            "TrainingIntegrityError would surface as a SUCCEEDED SF state"
        )


# ── the abolished promotion kind ───────────────────────────────────────────

class TestTheAbolishedPromotionKind:
    """RED before the fix: ``promoted_kind`` was set to
    ``"champion-arch-refresh"`` by ``select_and_finalize`` whenever no
    challenger won, and was written to the leaderboard, the promotion marker,
    the operator alert and the digest email without objection."""

    def test_writing_the_kind_raises(self):
        from training.model_zoo import (
            AbolishedPromotionKindError, _assert_no_arch_refresh_kind,
        )
        with pytest.raises(AbolishedPromotionKindError) as exc:
            _assert_no_arch_refresh_kind("champion-arch-refresh")
        assert "REFIT" in str(exc.value)

    def test_the_legitimate_kinds_pass(self):
        from training.model_zoo import _assert_no_arch_refresh_kind
        for kind in ("arena-pointer", "refit", None):
            _assert_no_arch_refresh_kind(kind)

    def test_the_promotion_path_can_no_longer_produce_the_kind(self):
        import ast
        import inspect
        import textwrap

        from training import model_zoo as mz

        tree = ast.parse(textwrap.dedent(inspect.getsource(mz.select_and_finalize)))
        literals = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert "champion-arch-refresh" not in literals, (
            "select_and_finalize can still assign the abolished kind"
        )


# ── the ranking statistic (I9322) ──────────────────────────────────────────

class TestTheRankingStatistic:
    """RED before the fix: the pointer was decided on ``cpcv_mean_ic``, whose
    relationship with realized 21d rank-IC was measured at Spearman +0.03,
    permutation p=0.90 over n=6 attributed champion eras."""

    def test_the_series_is_realized_market_relative_rank_ic_per_date(self):
        from analysis import observe_leaderboard as ol

        pairs = []
        for date, sign in (("2026-08-03", 1.0), ("2026-08-04", -1.0)):
            for i in range(12):
                pairs.append({
                    "date": date, "ticker": f"T{i}", "champion_version_id": "v-a",
                    "predicted_alpha": float(i),
                    "realized_alpha": sign * float(i),
                })
        per_date = ol.per_date_rank_ic_by_version(pairs)
        assert set(per_date["v-a"]) == {"2026-08-03", "2026-08-04"}
        assert per_date["v-a"]["2026-08-03"] == pytest.approx(1.0)
        assert per_date["v-a"]["2026-08-04"] == pytest.approx(-1.0)

    def test_a_thin_cross_section_is_a_miss_not_a_zero(self):
        from analysis import observe_leaderboard as ol

        pairs = [{"date": "2026-08-03", "ticker": "A", "champion_version_id": "v",
                  "predicted_alpha": 1.0, "realized_alpha": 1.0}]
        assert ol.per_date_rank_ic_by_version(pairs)["v"]["2026-08-03"] is None

    def test_two_refits_of_one_recipe_form_one_continuous_series(self):
        """The recipe boundary paying for itself: the arm's line does not
        restart when its weights refresh."""
        series = ams.build_series(
            "b",
            arm_id_by_label={"lbl-alpha": "M:alpha-arm:hash"},
            version_labels={"v-week1": "lbl-alpha", "v-week2": "lbl-alpha"},
            pairs=[
                {"date": "2026-08-03", "ticker": f"T{i}", "champion_version_id": "v-week1",
                 "predicted_alpha": float(i), "realized_alpha": float(i)}
                for i in range(12)
            ] + [
                {"date": "2026-08-10", "ticker": f"T{i}", "champion_version_id": "v-week2",
                 "predicted_alpha": float(i), "realized_alpha": float(i)}
                for i in range(12)
            ],
        )
        assert sorted(series["M:alpha-arm:hash"].scores) == ["2026-08-03", "2026-08-10"]

    def test_an_unclaimed_version_is_never_folded_into_another_arm(self):
        """§7.5 — provenance true by construction. The 2026-07-28 defect was an
        artifact asserting something false about its own origin."""
        series = ams.build_series(
            "b", arm_id_by_label={"lbl-alpha": "M:alpha:h"},
            version_labels={"v-stray": "lbl-nobody"},
            pairs=[
                {"date": "2026-08-03", "ticker": f"T{i}", "champion_version_id": "v-stray",
                 "predicted_alpha": float(i), "realized_alpha": float(i)}
                for i in range(12)
            ],
        )
        assert series["M:alpha:h"].scores == {}

    def test_the_config_declares_the_ruled_statistic_and_a_justified_clip(self):
        cfg = ams.ARENA_CONFIG.resolve()
        assert cfg.slot_kind == "model"
        assert cfg.benchmark == ams.BENCHMARK
        assert cfg.diff_clip == 0.30
        assert cfg.variance_mode == "declared"
        # the justification is recorded WITH the number, not in a commit message
        assert "0.30" in ams.DIFF_CLIP_RATIONALE
        assert "clip_rate" in ams.DIFF_CLIP_RATIONALE

    def test_the_cycle_records_whether_the_declared_clip_still_holds(self):
        diag = ams._clip_diagnostics(
            {
                "a": ArmSeries(arm_id="a", scores={"2026-08-03": 0.5, "2026-08-04": 0.01}),
                "b": ArmSeries(arm_id="b", scores={"2026-08-03": 0.0, "2026-08-04": 0.0}),
            },
            clip=0.30,
        )
        assert diag["n_paired_diffs"] == 2
        assert diag["n_at_or_beyond_clip"] == 1
        assert diag["clip_rate"] == pytest.approx(0.5)


# ── no evidence bars (policy §5.0) ─────────────────────────────────────────

class TestNoEvidenceBars:
    """RED before the fix: ``analysis/observe_leaderboard.py`` carried
    ``MIN_SOAK_WEEKS = 4`` and a ``ready_for_full_promotion`` verdict gated on
    it plus ``MIN_REALIZED_OUTCOMES``. Those are exactly the minimum-week and
    minimum-cohort bars §5.0 abolishes."""

    def test_the_minimum_week_bar_is_gone(self):
        from analysis import observe_leaderboard as ol
        assert not hasattr(ol, "MIN_SOAK_WEEKS")

    def test_no_promotion_readiness_verdict_survives(self):
        import inspect

        from analysis import observe_leaderboard as ol

        src = inspect.getsource(ol.build_observe_leaderboard)
        assert "ready_for_full_promotion" not in src

    def test_min_paired_dates_is_a_well_formedness_check_only(self):
        assert ams.ARENA_CONFIG.resolve().min_paired_dates == 1

    def test_the_slot_declares_no_floor_that_could_withhold_a_decision(self):
        cfg = ams.ARENA_CONFIG.resolve()
        # grace_weeks and cap govern RETIREMENT, never serving; min_active_arms
        # is the floor that stops a slot being stranded at one arm.
        assert (cfg.cap, cfg.grace_weeks, cfg.min_active_arms) == (5, 4, 3)
        assert cfg.retire_evidence == "point"


# ── serving preconditions (policy §5.3) ────────────────────────────────────

class TestServingPreconditions:
    """RED before the fix: the behavioural veto was applied inside
    ``select_winner`` as one of several promotion gates, so an arm could be
    excluded from a ranking rather than from SERVING, and the incumbent failing
    it did not force the pointer to move."""

    def test_the_veto_reaches_the_engine_unmodified_and_scale_dependent(self):
        """A candidate whose dispersion collapsed to 39% of the incumbent's.
        A standardized ratio would read 0.943 and pass; the scale-DEPENDENT
        form refuses."""
        pre = ams.serving_preconditions(
            arm_ids=["M:cand:h"],
            manifests_by_arm={
                "M:cand:h": {"output_distribution_gate": {"metrics": {"stdev_p_up": 0.0437}}},
                "M:inc:h": {"output_distribution_gate": {"metrics": {"stdev_p_up": 0.1136}}},
            },
            incumbent_arm="M:inc:h",
        )
        veto = [p for p in pre["M:cand:h"] if p.name == "behavioral_veto"][0]
        assert veto.passed is False
        assert "stdev_p_up" in veto.reason

    def test_an_uncomputable_veto_is_non_blocking_and_never_a_pass_in_prose(self):
        pre = ams.serving_preconditions(
            arm_ids=["M:cand:h"], manifests_by_arm={"M:cand:h": {}},
            incumbent_arm=None,
        )
        veto = [p for p in pre["M:cand:h"] if p.name == "behavioral_veto"][0]
        assert veto.passed is True
        assert "insufficient" in veto.reason

    def test_input_completeness_is_its_own_precondition(self):
        pre = ams.serving_preconditions(
            arm_ids=["M:cand:h"],
            manifests_by_arm={"M:cand:h": {"data_coverage_degraded": True}},
            incumbent_arm=None,
        )
        comp = [p for p in pre["M:cand:h"] if p.name == "input_completeness"][0]
        assert comp.passed is False

    def test_neither_precondition_is_folded_into_the_ranking(self):
        import inspect
        src = inspect.getsource(ams.build_series)
        for name in ("behavioral", "veto", "completeness"):
            assert name not in src, (
                "a serving precondition leaked into the SCORE — §5.3 makes "
                "them gates on serving, never ranking inputs"
            )

    def test_an_unservable_slot_fails_loud(self):
        register, by_label = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=_versions(
                ("v-a", "lbl-alpha", "2026-08-01"),
                ("v-b", "lbl-beta", "2026-08-01"),
                ("v-c", "v3.0-meta", "2026-08-01"),
            ),
            register=ArmRegister(),
        )
        arms = register.active_arms()
        series = {
            a: ArmSeries(arm_id=a, scores={"2026-08-08": 0.01, "2026-08-15": 0.02})
            for a in arms
        }
        blocked = {
            a: (ServingPrecondition(name="behavioral_veto", passed=False,
                                    reason="dispersion collapsed"),)
            for a in arms
        }
        cycle, _doc = ams.run_model_slot_cycle(
            as_of="2026-08-29", register=register, series_by_arm=series,
            incumbent=arms[0], preconditions=blocked,
            training={a: __import__(
                "nousergon_lib.arena.engine", fromlist=["TrainingStatus"],
            ).TrainingStatus(arm_id=a, ok=True) for a in arms},
        )
        assert cycle.decision.status == "unservable"
        with pytest.raises(ams.ArenaSlotUnservable):
            ams.assert_servable(cycle)


# ── the emitted artifact (M0 contract discipline) ──────────────────────────

class TestTheEmittedCycle:
    """RED before the fix: the M slot emitted no per-cycle decision artifact at
    all — the leaderboard recorded a CPCV ranking, not a pointer decision with
    the window and bound it rested on. A slot that emits nothing is not
    healthy, it is unobserved (``principles.md`` §2.7)."""

    def _cycle(self):
        register, by_label = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=_versions(
                ("v-a", "lbl-alpha", "2026-07-04"),
                ("v-b", "lbl-beta", "2026-07-04"),
                ("v-c", "v3.0-meta", "2026-07-04"),
            ),
            register=ArmRegister(),
        )
        from nousergon_lib.arena.engine import TrainingStatus

        arms = register.active_arms()
        dates = [f"2026-07-{d:02d}" for d in (6, 13, 20, 27)]
        series = {
            a: ArmSeries(arm_id=a, scores={d: 0.01 * (i + 1) for d in dates})
            for i, a in enumerate(arms)
        }
        return ams.run_model_slot_cycle(
            as_of="2026-08-29", register=register, series_by_arm=series,
            incumbent=arms[0],
            preconditions={a: () for a in arms},
            training={a: TrainingStatus(arm_id=a, ok=True) for a in arms},
        )

    def test_the_cycle_validates_against_the_merged_contract(self):
        from nousergon_lib.contracts import validate

        _cycle, doc = self._cycle()
        validate("arena_cycle", doc)

    def test_it_is_written_to_a_dated_key_and_a_latest_mirror(self):
        _cycle, doc = self._cycle()
        s3 = _FakeS3()
        keys = ams.emit_cycle(s3, "b", doc, as_of="2026-08-29")
        assert keys == ["arena/model/2026-08-29.json", "arena/model/latest.json"]
        assert s3.puts["arena/model/latest.json"]["slot"] == "M"

    def test_a_malformed_cycle_is_never_written(self):
        s3 = _FakeS3()
        with pytest.raises(Exception):
            ams.emit_cycle(s3, "b", {"slot": "M"}, as_of="2026-08-29")
        assert s3.puts == {}

    def test_cpcv_is_retained_as_a_diagnostic_and_is_not_the_score(self):
        register, by_label = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=_versions(("v-a", "lbl-alpha", "2026-07-04"),
                               ("v-b", "lbl-beta", "2026-07-04"),
                               ("v-c", "v3.0-meta", "2026-07-04")),
            register=ArmRegister(),
        )
        from nousergon_lib.arena.engine import TrainingStatus

        arms = register.active_arms()
        _cycle, doc = ams.run_model_slot_cycle(
            as_of="2026-08-29", register=register,
            series_by_arm={a: ArmSeries(arm_id=a, scores={}) for a in arms},
            incumbent=None, preconditions={a: () for a in arms},
            training={a: TrainingStatus(arm_id=a, ok=True) for a in arms},
            diagnostics={"cpcv_leaderboard": {"candidates": [{"cpcv_mean_ic": 0.05}]}},
        )
        assert doc["arena_config"]["ranking_statistic"] == "realized-market-relative-rank-ic"
        assert doc["retained_diagnostics"]["cpcv_leaderboard"]["candidates"]
        # and it is nowhere near the score
        assert "cpcv" not in json.dumps(doc["ladders"])

    def test_the_register_is_persisted_as_an_append_only_log(self):
        register, _ = ams.build_register(
            _FakeS3(), "b", as_of="2026-08-29", specs=_SPECS, canonical_horizon=21,
            versions=_versions(("v-a", "lbl-alpha", "2026-07-04")),
            register=ArmRegister(),
        )
        s3 = _FakeS3()
        ams.persist_register(s3, "b", register)
        events = s3.puts[ams.REGISTER_KEY]
        assert [e["kind"] for e in events] == ["registered"]
        assert ArmRegister.from_dicts(events).all_arms() == register.all_arms()


class TestTheSlotDoesNotReimplementTheEngine:
    """policy §10: a slot re-implementing §§3–6 is a defect, not a variation."""

    def test_no_local_ranking_pointer_or_retirement_logic(self):
        import inspect
        src = inspect.getsource(ams)
        for forbidden in ("def rank_", "def decide_pointer", "def confidence_sequence",
                          "def evaluate_retirements", "def pair_on_common_window"):
            assert forbidden not in src, forbidden

    def test_the_decision_comes_from_run_cycle(self):
        import inspect
        assert "engine.run_cycle(" in inspect.getsource(ams.run_model_slot_cycle)
