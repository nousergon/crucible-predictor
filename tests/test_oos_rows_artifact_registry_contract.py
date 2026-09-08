"""Producer/consumer contract test — alpha-engine-config-I10173.

``PredictorTraining``'s 2026-09-04 weekly-cycle stage-coverage verdict
reported ``predictor_oos_rows_dated`` as MISSING and ``predictor_oos_rows_latest``
as STALE, even though ``training/meta_trainer.py`` had written both keys minutes
into the cycle. Root cause, verified against live S3 (``ne-admin``):

    looked for : predictor/diagnostics/oos_rows/2026-09-04.parquet        (absent)
    exists at  : predictor/diagnostics/oos_rows/v3.0-meta/2026-09-04.parquet
    exists at  : predictor/diagnostics/oos_rows/v3.0-meta/latest.parquet

``TrainingIOSpec.oos_rows_key``/``oos_rows_latest_key`` were rescoped by
``{model_version}/`` under alpha-engine-config-I9378 (PR591, merged
2026-08-29) to stop a specialist run silently overwriting the champion's
diagnostic. ``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml``'s
``predictor_oos_rows_dated`` / ``predictor_oos_rows_latest`` rows were never
updated to the new ``{model_version}/`` segment — the coverage check kept
looking for the pre-I9378 flat key and reported the producer's real, correctly
-written output as missing/stale. This is a registry-declaration defect, not
a producer defect: the producer's per-arm keying is deliberate (I9378) and is
NOT reverted here.

This test pins the producer's key SHAPE so nobody can silently drop the
``{model_version}/`` segment again — reintroducing exactly the I9378 collision
— without also being told (via a failing assertion here) that the registry's
declared ``s3_key_template`` for both artifacts must move in lockstep. The
registry side of the contract is closed by hand in
``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml`` (private repo,
out of this repo's scope) — see alpha-engine-config-I10173.
"""

from __future__ import annotations

import re

from training.io_spec import TrainingIOSpec

# The exact templates this test asserts the registry must declare, mirrored
# here as plain strings (not imported — the registry lives in a private repo
# this test suite cannot see) so a producer-side format change is caught even
# though the registry itself cannot be read from CI.
_REGISTRY_DATED_TEMPLATE = "predictor/diagnostics/oos_rows/{model_version}/{date}.parquet"
_REGISTRY_LATEST_TEMPLATE = "predictor/diagnostics/oos_rows/{model_version}/latest.parquet"


def _spec() -> TrainingIOSpec:
    return TrainingIOSpec.live()


def test_oos_rows_key_matches_the_registry_declared_template():
    spec = _spec()
    key = spec.oos_rows_key("2026-09-04", "v3.0-meta")
    expected = _REGISTRY_DATED_TEMPLATE.format(model_version="v3.0-meta", date="2026-09-04")
    assert key == expected, (
        "producer's oos_rows_key() no longer matches the key shape "
        "ARTIFACT_REGISTRY.yaml's predictor_oos_rows_dated must declare "
        "(alpha-engine-config-I10173) — update the registry's "
        "s3_key_template in lockstep with this change."
    )


def test_oos_rows_latest_key_matches_the_registry_declared_template():
    spec = _spec()
    key = spec.oos_rows_latest_key("v3.0-meta")
    expected = _REGISTRY_LATEST_TEMPLATE.format(model_version="v3.0-meta")
    assert key == expected, (
        "producer's oos_rows_latest_key() no longer matches the key shape "
        "ARTIFACT_REGISTRY.yaml's predictor_oos_rows_latest must declare "
        "(alpha-engine-config-I10173) — update the registry's "
        "s3_key_template in lockstep with this change."
    )


def test_oos_rows_key_is_scoped_by_model_version_not_flat_by_date():
    """Regression guard for the pre-I9378 shape a stale registry entry
    silently reintroduces as the 'source of truth' if this ever regresses:
    a bare ``{prefix}{date}.parquet`` with no model_version segment."""
    spec = _spec()
    champion_key = spec.oos_rows_key("2026-09-04", "v3.0-meta")
    specialist_key = spec.oos_rows_key("2026-09-04", "spec-sota-combine")
    assert champion_key != specialist_key, (
        "two arms training on the same date must not resolve to the same "
        "oos_rows key — this is exactly the I9378 collision."
    )
    flat_unscoped = f"{spec.oos_rows_prefix}2026-09-04.parquet"
    assert champion_key != flat_unscoped
    assert re.match(
        r"^predictor/diagnostics/oos_rows/[^/]+/2026-09-04\.parquet$", champion_key
    )


def test_oos_rows_key_requires_a_nonempty_model_version():
    spec = _spec()
    for bad in ("", None):
        try:
            spec.oos_rows_key("2026-09-04", bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(
            "oos_rows_key must refuse an empty model_version rather than "
            "silently falling back to an unscoped key (alpha-engine-config-I9378)."
        )
