"""training/commitment.py — the pre-registered out-of-sample commitment window,
as a PREDICATE the promotion loop must clear rather than a declaration in a file
nothing reads.

Why this module exists
----------------------
``alpha-engine-config/private-docs/ALPHA_EXPERIMENT_COMMITMENT.yaml`` has
declared, on ``main`` since 2026-09-10, a 252-session out-of-sample freeze
pinning the serving champion to ``v3.0-meta-2026-08-14-119e069b`` with
``rotation: suspended`` (Brian rulings 2026-09-08 / alpha-engine-config-I9679
and 2026-09-09 / -I10302). Measured 2026-09-12: **nothing in this repo read that
file.** ``git grep -i commitment origin/main -- training/ inference/ model/``
returned zero hits, and on the 2026-09-12 weekly Step Function
``ModelZooSelect`` auto-promoted ``spec-sota-combine-2026-09-11-753cfbed``
(``promoted_kind: arena-pointer``, CPCV IC -0.043) straight over the frozen
version — recorded in
``s3://alpha-engine-research/predictor/model_zoo/promotions/2026-09-11.json``.
The window was voided by the very loop it suspended, on its first weekly
rotation, because the suspension existed only in prose.

That is the whole failure class: a control that is committed, reviewed and
ruled, and absent from the running system. The declaration is now an
ACTUATOR — ``training/model_zoo.py``'s cutover reads this predicate and refuses
to promote while the window stands — and the daily inference pipeline carries
the paired DETECTOR (``inference/stages/commitment_guard.py``), so a window
voided by any path at all (a hand promotion, a config flip, a box that never
pulled) pages instead of passing.

Design posture: REFUSE RATHER THAN GUESS
----------------------------------------
Deliberately the same shape as ``training/model_zoo_gates.py``. Nothing here
defaults a ruling into existence:

* The window is active only on an AFFIRMATIVE reading — a parsed file, a
  ``freeze_date`` at or before ``as_of``, ``rotation: suspended``, and no VOID
  state. Any one of those missing means "no window", stated, never inferred.
* A file that is configured but MISSING or UNPARSEABLE is not "no window" — it
  is an unreadable ruling, and the caller (see
  ``model_zoo._commitment_refusal``) fails CLOSED on it. ``load_commitment``
  therefore distinguishes the two: ``None`` for absent, and a raise for
  present-but-unreadable, so no caller can collapse them into one silent pass.

The predicate does NOT count sessions. ``window_length_sessions`` describes when
the window COMPLETES, which is an adjudication question (the trading-day axis,
holidays, and what a voided-and-restarted window counts from); the promotion
question is only "is the freeze standing today". Reading a session count to
decide whether to promote would make the gate depend on a calendar the
promotion loop has no business owning, and would fail open on any disagreement
about it. Expiry is therefore adjudicated on the console/evaluator surface that
already renders the window, not here.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


# The box checkout the weekly ``ModelZooSelect`` step already ``git pull``s
# before it runs (``infrastructure/spot_train.sh --model-zoo-select`` pulls BOTH
# /home/ec2-user/alpha-engine-predictor and /home/ec2-user/alpha-engine-config),
# so the ruling that reaches the promoter is the one on alpha-engine-config's
# main as of that pull — not a copy baked into this repo's container.
DEFAULT_COMMITMENT_PATH = (
    "/home/ec2-user/alpha-engine-config/private-docs/ALPHA_EXPERIMENT_COMMITMENT.yaml"
)

#: The state string the detector writes when the window has been voided.
VOID_STATE = "VOID"


class CommitmentUnreadable(RuntimeError):
    """The commitment file exists but could not be read as a ruling.

    Separate from "absent" on purpose: a caller must be able to tell "no window
    is declared" from "a window may be declared and I cannot see it", because
    the correct action differs (proceed vs refuse).
    """


def commitment_path() -> str:
    """Resolve the commitment file path.

    ``ALPHA_EXPERIMENT_COMMITMENT_PATH`` (env, via ``config``) overrides;
    otherwise the box checkout path above. Read through ``config`` rather than
    ``os.environ`` directly so the value appears in one place with every other
    resolved setting.
    """
    try:
        import config as cfg

        path = getattr(cfg, "ALPHA_EXPERIMENT_COMMITMENT_PATH", None)
        if path:
            return str(path)
    except Exception as exc:  # noqa: BLE001
        # (a) swallowed: `config` could not be imported (a bare context with no
        # predictor.yaml). (b) the primary deliverable survives because the
        # fallback below resolves the SAME value `config` would have — the env
        # var, else the box-checkout default — so the caller still gets a path
        # and still fails closed if the file behind it is absent. (c) recording
        # surface: this WARNING. Not debug: in production `config` always
        # imports, so reaching this line at all means the resolved path was not
        # the one the rest of the process is using.
        log.warning(
            "commitment: config import unavailable (%s: %s) — resolving the "
            "commitment path from the environment/default instead",
            type(exc).__name__, exc,
        )
    return os.environ.get("ALPHA_EXPERIMENT_COMMITMENT_PATH") or DEFAULT_COMMITMENT_PATH


def load_commitment(path: str | os.PathLike | None = None) -> dict | None:
    """Parse the commitment YAML.

    Returns the parsed mapping, or ``None`` when the file does NOT EXIST — the
    legitimate "no window is declared" state.

    Raises :class:`CommitmentUnreadable` when the file exists but cannot be
    parsed, or parses to something other than a mapping. A present-but-broken
    ruling must never read as an absent one.
    """
    p = Path(path) if path is not None else Path(commitment_path())
    if not p.exists():
        return None
    try:
        import yaml

        with open(p) as fh:
            doc = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 — re-raised as a named error below
        raise CommitmentUnreadable(f"{p}: {type(exc).__name__}: {exc}") from exc
    if doc is None:
        raise CommitmentUnreadable(f"{p}: parsed to null — an empty ruling file")
    if not isinstance(doc, dict):
        raise CommitmentUnreadable(f"{p}: parsed to {type(doc).__name__}, expected a mapping")
    return doc


def _as_date(value) -> _dt.date | None:
    """Coerce a YAML date / ISO string / datetime to a ``date``; None if unusable."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return _dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def freeze_date(commitment: dict | None) -> _dt.date | None:
    """The declared freeze date, or None when unset/unparseable."""
    if not isinstance(commitment, dict):
        return None
    return _as_date(commitment.get("freeze_date"))


def frozen_version_ids(commitment: dict | None) -> list[str]:
    """The registry version ids the window pins.

    Reads ``frozen_version_ids`` and unions the legacy ``frozen_arm_ids`` alias
    (identical content by that file's own comment, retained for older readers).
    The subject is a VERSION, never an arm: a refit inside the frozen arm
    replaces the pinned weights and voids the window exactly as a pointer move
    would (alpha-engine-config-I10302), and an arm id cannot express that.
    """
    if not isinstance(commitment, dict):
        return []
    out: list[str] = []
    for field in ("frozen_version_ids", "frozen_arm_ids"):
        raw = commitment.get(field)
        if isinstance(raw, str):
            raw = [raw]
        for item in raw or []:
            vid = str(item).strip()
            if vid and vid not in out:
                out.append(vid)
    return out


def window_state(commitment: dict | None) -> str | None:
    """The declared window state, upper-cased, or None when the file has none.

    The file carries no state field today (only ``on_void`` rendering
    instructions), so ABSENCE means not-void — stated here rather than left to
    each caller's reading.
    """
    if not isinstance(commitment, dict):
        return None
    raw = commitment.get("state") or commitment.get("window_state")
    if isinstance(raw, dict):
        raw = raw.get("state")
    if raw is None:
        return None
    return str(raw).strip().upper() or None


def window_active(commitment: dict | None, as_of: _dt.date | str | None = None) -> bool:
    """Is the out-of-sample freeze standing on ``as_of``?

    True only on an affirmative reading of all four conditions:

    1. ``commitment`` parsed to a mapping;
    2. ``freeze_date`` is set and ``as_of >= freeze_date``;
    3. ``rotation == "suspended"`` — the file's own statement that the weekly
       promotion loop does not run for the duration;
    4. the window is not VOID (absence of a state field means not-void).

    No session counting — see the module docstring.
    """
    if not isinstance(commitment, dict):
        return False
    fd = freeze_date(commitment)
    if fd is None:
        return False
    as_of_d = _as_date(as_of) or _dt.date.today()
    if as_of_d < fd:
        return False
    rotation = commitment.get("rotation")
    if str(rotation or "").strip().lower() != "suspended":
        return False
    if window_state(commitment) == VOID_STATE:
        return False
    return True


def describe(commitment: dict | None, *, path: str | None = None) -> dict:
    """A compact, JSON-safe record of the ruling, for artifacts and alerts."""
    fd = freeze_date(commitment)
    return {
        "path": path or commitment_path(),
        "freeze_date": fd.isoformat() if fd else None,
        "window_length_sessions": (commitment or {}).get("window_length_sessions"),
        "frozen_scope": (commitment or {}).get("frozen_scope"),
        "frozen_version_ids": frozen_version_ids(commitment),
        "rotation": (commitment or {}).get("rotation"),
        "state": window_state(commitment),
    }
