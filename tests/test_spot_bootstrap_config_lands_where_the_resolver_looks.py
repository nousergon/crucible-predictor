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
predictor.yaml`` — exactly where ``bootstrap_spot`` stages it. The predictor
was never exposed to config#6846, because it stages onto its *repo-local
fallback* rather than onto a config-repo path.

What the audit DID find is the same class pointing the other way: the bootstrap
also staged a second copy to ``/home/ec2-user/predictor/experiments/<id>/
predictor/predictor.yaml``, which matches no candidate at all. The resolver's
experiment-package candidates are rooted at the *alpha-engine-config* checkout
(``~/alpha-engine-config`` and ``<repo>/../alpha-engine-config``), never inside
this repo's own tree. That copy was a dead write: it cost an S3 GET per launch
and, worse, read as experiment-package coverage that did not exist. Removed,
and this test is what keeps it removed.

## Why the assertion is EVERY destination, not ANY

A staged config at a path nothing searches is indistinguishable, from the
outside, from one that works — until the workload raises ``FileNotFoundError``
several minutes later, on a box that looked healthy through the whole
bootstrap. Allowing dead destinations alongside a live one is how the dead one
survives long enough for someone to rely on it.

The candidate list here is DERIVED from
``nousergon_lib.config.resolve_experiment_config`` rather than hardcoded: if the
resolver's search order changes, this test fails instead of the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

from nousergon_lib.config import resolve_experiment_config

# Where the bootstrap clones crucible-predictor on the spot box, and the working
# directory every stage's SSM heredoc cds into before invoking python.
_REMOTE_CHECKOUT = Path("/home/ec2-user/predictor")

#: The bootstrap hardcodes `reference` as the experiment slot it exports.
_EXPERIMENT_ID = "reference"

_SPOT_COMMON = Path(__file__).resolve().parents[1] / "infrastructure" / "_spot_common.sh"
_SRC = _SPOT_COMMON.read_text(encoding="utf-8")

# `aws s3 cp "${S3_STAGING}/predictor.yaml" "<dest>"` inside the bootstrap heredoc.
_STAGE_CP = re.compile(
    r'aws\s+s3\s+cp\s+"\$\{S3_STAGING\}/predictor\.yaml"\s+"?(?P<dest>[^"\s]+)"?'
)


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


def _staged_destinations() -> list[str]:
    return [m.group("dest") for m in _STAGE_CP.finditer(_SRC)]


def test_bootstrap_stages_config_at_all():
    assert _staged_destinations(), (
        f"{_SPOT_COMMON.name} no longer stages predictor.yaml in its bootstrap — "
        "if that is deliberate (prebaked image), delete this test with the "
        "reason; otherwise every stage will fail on a fresh box."
    )


def test_bootstrap_stages_config_to_a_resolver_candidate():
    dests = _staged_destinations()
    candidates = _resolver_candidates()
    assert any(d in candidates for d in dests), (
        f"the bootstrap stages predictor.yaml to {dests}, none of which "
        f"config.py searches on a box with the checkout at {_REMOTE_CHECKOUT}. "
        f"Resolver candidates: {candidates}. This is config#6846: the copy "
        "succeeds, the box looks healthy, and the workload dies on "
        "FileNotFoundError several minutes later."
    )


def test_no_staged_destination_is_unreachable():
    """A config staged where nothing looks is dead weight that reads as coverage.

    The instance this pins: the bootstrap used to write a second copy into
    ``<repo>/experiments/<id>/predictor/``, on the assumption that the
    experiment-package candidates are rooted in the repo. They are not — they
    are rooted at the alpha-engine-config checkout.
    """
    candidates = set(_resolver_candidates())
    dead = [d for d in _staged_destinations() if d not in candidates]
    assert not dead, (
        f"the bootstrap stages predictor.yaml to {dead}, which "
        f"resolve_experiment_config never searches. Either the destination is "
        f"wrong or the resolver call in config.py is. Candidates: "
        f"{sorted(candidates)}"
    )
