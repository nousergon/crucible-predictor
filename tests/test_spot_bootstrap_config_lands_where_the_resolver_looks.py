"""The spot bootstrap must stage predictor.yaml where the resolver actually looks.

Bug class (config#6846, live failure ``watch-rerun-2026-08-10-4``, 2026-08-11):
in ``nousergon-data``'s twin of ``_spot_common.sh`` the bootstrap staged its
config to a path ``load_config`` never searches, so MorningEnrich died on::

    FileNotFoundError: Config data/config.yaml not found. Searched:
      ['/home/ec2-user/alpha-engine-config/experiments/reference/data/config.yaml',
       '/home/ec2-user/alpha-engine-config/data/config.yaml',
       'config.yaml']

That fix (``nousergon-data#1298``) was the one defect of the 2026-08-10/11 set
never carried to this repo, and alpha-engine-config-I6922 flagged it as
"not ported; unknown whether it applies". This test is the audit, made
permanent.

## The audit result: it does not apply here, and the reason is measurable

``config.py`` resolves via ``resolve_experiment_config("predictor",
"predictor.yaml", repo_root=<repo>, repo_local_fallback=<repo>/config/
predictor.yaml)``. On the spot box, with the clone at ``/home/ec2-user/
predictor``, the last candidate is ``/home/ec2-user/predictor/config/
predictor.yaml`` — exactly where the bootstrap stages it. The predictor was
never exposed to config#6846, because it stages onto its *repo-local
fallback* rather than onto a config-repo path.

## Subject moved: the rendered script, not this repo's source

``bootstrap_spot()`` no longer builds this ``aws s3 cp`` line as literal text
in ``_spot_common.sh`` (alpha-engine-config-I4992/I6922 cutover) — it passes
``--config-copy predictor.yaml:/home/ec2-user/predictor/config/predictor.yaml``
to ``krepis.spot_bootstrap render``, which emits the copy. This module now
derives that argument from the live source and asserts the destination
against the resolver using the script krepis actually renders.

## Why the assertion is EVERY destination, not ANY

A staged config at a path nothing searches is indistinguishable, from the
outside, from one that works — until the workload raises ``FileNotFoundError``
several minutes later, on a box that looked healthy through the whole
bootstrap. Allowing dead destinations alongside a live one is how the dead one
survives long enough for someone to rely on it. (The audit that motivated this
test found exactly one such dead write in this repo's history — a second copy
rooted inside the checkout rather than at the alpha-engine-config checkout the
resolver's experiment-package candidates actually live under — removed before
this cutover and not reintroduced by it.)

The candidate list here is DERIVED from
``nousergon_lib.config.resolve_experiment_config`` rather than hardcoded: if the
resolver's search order changes, this test fails instead of the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nousergon_lib.config import resolve_experiment_config

from tests._spot_bootstrap_render_support import rendered_script

# Where the bootstrap clones crucible-predictor on the spot box, and the working
# directory every stage's SSM heredoc cds into before invoking python.
_REMOTE_CHECKOUT = Path("/home/ec2-user/predictor")

#: The bootstrap hardcodes `reference` as the experiment slot it exports.
_EXPERIMENT_ID = "reference"

# `aws s3 cp "${S3_STAGING}/predictor.yaml" <dest>` in the RENDERED script.
_STAGE_CP = re.compile(
    r'aws\s+s3\s+cp\s+"\$\{S3_STAGING\}/predictor\.yaml"\s+"?(?P<dest>[^"\s]+)"?'
)


@pytest.fixture(scope="module")
def rendered() -> str:
    return rendered_script()


def _resolver_candidates() -> list[str]:
    """The exact paths ``config.py`` would try on the spot box.

    Mirrors ``config.py``: same subdir, filename, repo_root and
    repo_local_fallback, with ``resolve=False`` so we get the candidate list
    rather than a FileNotFoundError.
    """
    candidates = resolve_experiment_config(
        "predictor",
        "predictor.yaml",
        repo_root=_REMOTE_CHECKOUT,
        repo_local_fallback=_REMOTE_CHECKOUT / "config" / "predictor.yaml",
        experiment_id=_EXPERIMENT_ID,
        resolve=False,
    )
    return [
        str(_REMOTE_CHECKOUT / c) if not Path(c).is_absolute() else str(c)
        for c in candidates
    ]


def _staged_destinations(rendered: str) -> list[str]:
    return [m.group("dest") for m in _STAGE_CP.finditer(rendered)]


def test_bootstrap_stages_config_at_all(rendered: str):
    assert _staged_destinations(rendered), (
        "the rendered bootstrap no longer stages predictor.yaml — if that is "
        "deliberate (prebaked image), delete this test with the reason; "
        "otherwise every stage will fail on a fresh box."
    )


def test_bootstrap_stages_config_to_a_resolver_candidate(rendered: str):
    dests = _staged_destinations(rendered)
    candidates = _resolver_candidates()
    assert any(d in candidates for d in dests), (
        f"the bootstrap stages predictor.yaml to {dests}, none of which "
        f"config.py searches on a box with the checkout at {_REMOTE_CHECKOUT}. "
        f"Resolver candidates: {candidates}. This is config#6846: the copy "
        "succeeds, the box looks healthy, and the workload dies on "
        "FileNotFoundError several minutes later."
    )


def test_no_staged_destination_is_unreachable(rendered: str):
    """A config staged where nothing looks is dead weight that reads as coverage."""
    candidates = set(_resolver_candidates())
    dead = [d for d in _staged_destinations(rendered) if d not in candidates]
    assert not dead, (
        f"the bootstrap stages predictor.yaml to {dead}, which "
        f"resolve_experiment_config never searches. Either the destination is "
        f"wrong or the resolver call in config.py is. Candidates: "
        f"{sorted(candidates)}"
    )
