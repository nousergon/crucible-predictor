"""tests/test_dead_l1_observe_manifest.py — pin the W4.2 dead-L1 OBSERVE
cohort into the PERSISTED manifest (config#1994).

The dead-L1 monitor (predictor#351) computes ``dead_l1_observe`` and its
input ``meta_oos_ic_leakfree_per_l1_dropout`` at meta-training time and
returned them in the run-result dict — but they were DROPPED from the
``manifest`` dict literal that ``run_meta_training`` writes to
``s3://.../predictor/weights/meta/manifest.json``. Every sibling observe
diagnostic (``meta_oos_ic_leakfree_nonlinear``, ``_post2020``,
``_no_expected_move`` …) is persisted there, so the report card / console
reads the manifest — meaning the dead-L1 cohort silently never landed in
S3 and config#1994's "first observe cohort result recorded" closes-when
could never be satisfied, however many weekly runs succeeded.

Source-text invariant only (a behavioral test would require spinning up
the whole trainer with synthetic data — out of scope, per
test_meta_trainer_oos_ic_field.py). We assert the two keys live INSIDE
the persisted-manifest dict literal, not merely somewhere in the module
(they were always in the return dict).
"""
from __future__ import annotations

from pathlib import Path

import pytest


_META_TRAINER = (
    Path(__file__).resolve().parent.parent / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def manifest_block() -> str:
    """The ``manifest = { ... }`` literal that gets json.dump'd to S3.

    Sliced from ``manifest = {`` to the ``manifest["ic_reliability"]``
    augmentation that immediately precedes the ``put_object`` write — the
    dict literal is fully closed by then, so any key in this slice is a
    persisted manifest field.
    """
    src = _META_TRAINER.read_text()
    start = src.index("manifest = {")
    end = src.index('manifest["ic_reliability"]', start)
    return src[start:end]


@pytest.mark.parametrize(
    "key",
    ["dead_l1_observe", "meta_oos_ic_leakfree_per_l1_dropout"],
)
def test_dead_l1_observe_pair_persisted_in_manifest(manifest_block, key):
    assert f'"{key}":' in manifest_block, (
        f"{key} is missing from the persisted manifest dict — the W4.2 "
        "dead-L1 OBSERVE cohort will not land in S3 and config#1994 can "
        "never record its first cohort (regression of the predictor#351 "
        "wiring gap)."
    )
