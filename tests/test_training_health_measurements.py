"""alpha-engine-config-I12629: the training health record read keys the trainer never sets.

Measured on the 10-23 weekly: `health/predictor_training.json` had
`duration_seconds 0` and `ic_30d null` under `status ok`, while
`predictor/metrics/training_summary_latest.json` for the same run reported
`elapsed_s 896.1` and `test_ic 0.083252`.
"""

from __future__ import annotations

from training import train_handler


def test_the_trainers_own_keys_fill_the_record():
    dur, ic, warns = train_handler._training_health_measurements(
        {"elapsed_s": 896.1, "test_ic": 0.083252, "n_train": 1461381}
    )
    assert dur == 896.1 and ic == 0.083252 and warns == []


def test_legacy_keys_still_win_when_a_trainer_supplies_them():
    dur, ic, warns = train_handler._training_health_measurements(
        {"train_time_seconds": 12.0, "ic_30d": 0.04, "elapsed_s": 99.0, "test_ic": 0.09}
    )
    assert (dur, ic, warns) == (12.0, 0.04, [])


def test_an_unmeasured_ic_is_a_warning_never_a_silent_ok():
    dur, ic, warns = train_handler._training_health_measurements({"elapsed_s": 10.0})
    assert ic is None
    assert any("ic not measured" in w for w in warns)


def test_the_record_carries_the_measurements(monkeypatch):
    import data_manifest
    import nousergon_lib.health as health

    seen: dict = {}
    monkeypatch.setattr(health, "write_health", lambda **kw: seen.update(kw))
    monkeypatch.setattr(data_manifest, "write_data_manifest", lambda **kw: None)

    class _IO:
        write_side_artifacts = True

    train_handler._write_run_health_and_manifest(
        {"elapsed_s": 896.1, "test_ic": 0.083252, "n_train": 5}, "bkt", "2026-10-23", _IO()
    )
    assert seen["duration_seconds"] == 896.1
    assert seen["summary"]["ic_30d"] == 0.083252
    assert seen["summary"]["test_ic"] == 0.083252
    assert seen["warnings"] is None
