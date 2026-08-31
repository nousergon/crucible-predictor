"""
analysis/observe_leaderboard.py — realized-edge measurement for the model zoo.

config#671/#673/#702/#1052/L4539. Originally the SLOWER confirm layer for the
Option-B observe tier; under the RELATIVE-BEST promotion rule (#1052) that tier is
retired (a beats-champion challenger PROMOTES, it does not enter an observe soak).
This module now serves TWO realized-edge reads, both pure MEASUREMENT (never
promote / demote / allocate):

1. ``build_champion_realized_monitor`` (the primary, live read) — the
   NOISE-CHASING MONITOR of the PROMOTED champion. Relative-best promotion accepts
   the best-of-N challenger over the incumbent each week without an absolute
   significance gate; the risk is chasing noise. This scores the LIVE champion's
   own ``predictor/predictions/{date}.json`` vs REALIZED 21d sector-neutral alpha
   over a trailing window. A weak/negative realized rank-IC ⇒ ``chasing_noise``
   True — an OBSERVABILITY/alarm signal that we may be over-fitting the selector,
   NOT a gate. ``training/model_zoo.py::run_rotation_and_select`` calls this each
   rotation after the promote and stashes the verdict on the leaderboard.

2. ``build_observe_leaderboard`` (retained) — scores any still-shadowed versions
   (``predictor/predictions_shadow/{version_id}/{date}.json``; the shadow runner
   still shadows registered CHALLENGER-stage versions) against realized 21d alpha
   alongside the champion. A diagnostic comparison surface; with the observe tier
   retired it no longer gates any promotion, and since I9319 it carries no
   promotion-readiness verdict at all — the M-slot pointer is decided by
   ``nousergon_lib.arena`` and recorded in ``arena/model/{date}.json``.
   Operator may still read this for the realized numbers.

Reuses the realized-alpha machinery from ``analysis/variant_cutover_gate.py``
(``compute_realized_alpha_for_pairs``) and the S3 price / sector_map loaders from
``analysis/triple_barrier_cutover_runner.py`` so there is one realized-alpha
implementation, not a parallel shortcut (SOTA mirror, not re-invent).

S3 layout::

    predictor/model_zoo/observe_leaderboard/
      ├── {trading_day}.json   ← per-run leaderboard / champion monitor (dated)
      └── latest.json          ← single-fetch operator UX (mirror)

Both payloads carry ``calendar_date`` + ``trading_day`` per DATE_CONVENTIONS.md.

Usage (programmatic)::

    from analysis.observe_leaderboard import build_champion_realized_monitor
    mon = build_champion_realized_monitor(bucket="alpha-engine-research")

Usage (CLI)::

    python -m analysis.observe_leaderboard --bucket alpha-engine-research --window 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg  # noqa: E402
from analysis.triple_barrier_cutover_runner import (  # noqa: E402
    _load_prices_from_s3,
    _load_sector_map_from_s3,
)
from analysis.variant_cutover_gate import (  # noqa: E402
    compute_realized_alpha_for_pairs,
)

# Canonical alpha field emitted by both live and shadow predictions
# (inference/stages/run_inference.py:835).
ALPHA_FIELD = "predicted_alpha"

# Forward-realized horizon — the canonical 21d label.
DEFAULT_HORIZON = 21

# Trailing prediction-history window. ~6 weeks gives a 4-week soak a buffer for
# the 21d realized-window close.
DEFAULT_WINDOW_DAYS = 42

# S3 prefixes.
_SHADOW_PREFIX = "predictor/predictions_shadow"
_LIVE_PREDICTIONS_PREFIX = "predictor/predictions"
OUTPUT_PREFIX = "predictor/model_zoo/observe_leaderboard"

# ── Evidence bars: what survived §5.0 and what did not (I9319) ─────────────
#
# `champion-challenger-policy.md` §5.0 abolishes minimum-week bars and
# minimum-cohort counts on every slot: they control no error rate at all, and
# they are what deadlocked every promotion. The anytime-valid confidence
# sequence in `nousergon_lib.arena.confseq` replaces them and subsumes them
# naturally — the interval is very wide at week one, so no lead is supported,
# and it narrows as evidence accrues.
#
# REMOVED (I9319): `MIN_SOAK_WEEKS = 4` and the `ready_for_full_promotion`
# verdict it gated. That was exactly a minimum-week bar on the decision path
# for the M slot. The decision now lives in `arena/model/{date}.json`.
#
# The line between the two kinds of floor, which is the whole judgement:
#   * a floor on whether a MEASUREMENT IS WELL-FORMED survives;
#   * a floor on whether EVIDENCE IS SUFFICIENT TO DECIDE does not.
#
# `_MIN_PAIRS_FOR_IC` is the first kind and only the first kind: a Spearman
# correlation over fewer than ten pairs is not a weak statistic, it is not a
# statistic. It is `ArenaConfig.min_paired_dates`'s shape. **It may never be
# read as an evidence bar**, and nothing may use it to withhold a decision the
# confidence sequence is willing to take.
_MIN_PAIRS_FOR_IC = 10
# `MIN_REALIZED_OUTCOMES` is retained for ONE purpose only: deciding whether
# the noise-chasing ALARM can assert at all. An alarm that fires off three
# matured pairs is noise about noise. It is not on any decision path, and
# reinstating it as one would re-create what §5.0 removed.
MIN_REALIZED_OUTCOMES = 20

# ── alpha-engine-config-I9336: measurability vs. merely-thin ───────────────
# Mirrors crucible-research/scoring/leaderboard_scoring.py::measurability_for
# (landing in crucible-research-PR767) — same vocabulary, reimplemented
# locally (the predictor and research Lambdas are separate deployables with
# no shared runtime import boundary). Second adoption of this shape;
# consider lifting both into nousergon-lib (shared-code-policy.md).
MEASURABILITY_MEASURED = "measured"
MEASURABILITY_UNMEASURABLE = "unmeasurable"
# An arm that wrote none of the most recent N cohort dates has stopped
# producing comparable output — mirrors crucible-research's
# COHORT_LAG_UNMEASURABLE_DATES.
MEASURABILITY_LAG_CYCLES = 3

# ── alpha-engine-config-I8219: provenance of a pair row's champion attribution ─
# Every pair row carries WHERE its ``champion_version_id`` came from, so an
# unattributed row is never indistinguishable from a row whose attribution was
# dropped in transit (champion-challenger-policy §7.5, and the defect this
# constant was added for — see ``_prediction_pair_rows``).
ATTR_STAMPED = "stamped"            # the artifact's own champion_version_id
ATTR_PROMOTION_HISTORY = "promotion_history"  # resolved from promotions/{date}.json
ATTR_SHADOW_PREFIX = "shadow_prefix"  # the shadow directory names the producer
ATTR_UNATTRIBUTED = "unattributed"  # no stamp and no history covers this date

# The durable, authoritative record of which registry version became champion on
# which rotation date. ``model_zoo.py::_write_promotion_marker`` writes one object
# per rotation and never rewrites one, so it is a complete cutover history for
# every date from the first marker onward.
_PROMOTIONS_PREFIX = "predictor/model_zoo/promotions"


def _list_shadow_versions(bucket: str, s3_client=None) -> list[str]:
    """The version_ids under ``predictor/predictions_shadow/`` (one dir per
    shadowed version). Best-effort: returns [] on read failure."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    versions: set[str] = set()
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket, Prefix=f"{_SHADOW_PREFIX}/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                pfx = cp.get("Prefix", "")
                # predictor/predictions_shadow/<vid>/  → <vid>
                vid = pfx[len(_SHADOW_PREFIX) + 1:].rstrip("/")
                if vid:
                    versions.add(vid)
    except Exception:  # noqa: BLE001 — empty leaderboard is a valid (no-data) result
        log.warning("observe_leaderboard: failed to list shadow versions", exc_info=True)
        return []
    return sorted(versions)


def _list_registered_challenger_ids(bucket: str, s3_client=None) -> list[str]:
    """version_ids currently registered ``stage=challenger`` — the CENSUS of
    arms that must be measured every cycle (alpha-engine-config-I9336
    deliverable 2). Best-effort: [] on read failure, matching this module's
    existing degrade shape — an unreadable census adds no explicit
    unmeasurable rows rather than blocking the leaderboard build.
    """
    import boto3

    from model.registry import list_versions

    s3 = s3_client or boto3.client("s3")
    try:
        return [
            v["version_id"] for v in list_versions(s3, bucket, stage="challenger")
            if v.get("version_id")
        ]
    except Exception:  # noqa: BLE001 — best-effort census, never fatal
        log.warning("observe_leaderboard: failed to list registered challengers", exc_info=True)
        return []


def _list_shadow_dates_for_version(bucket: str, version_id: str, s3_client=None) -> list[str]:
    """Every date this version has EVER written a shadow prediction for — the
    FULL history (not windowed to ``n_days``), so a genuinely-stale arm (last
    write outside the scoring window) is distinguishable from one that never
    wrote anything at all. Best-effort: [] on read failure."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    dates: list[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket, Prefix=f"{_SHADOW_PREFIX}/{version_id}/",
        ):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".json"):
                    dates.append(name[: -len(".json")])
    except Exception:  # noqa: BLE001 — best-effort; treated as "no dates known"
        log.warning(
            "observe_leaderboard: failed to list shadow dates for %s",
            version_id, exc_info=True,
        )
        return []
    return sorted(dates)


def measurability_for_shadow_arm(
    dates_scored: list[str],
    cohort_dates: list[str],
    *,
    lag: int = MEASURABILITY_LAG_CYCLES,
) -> tuple[str, str | None]:
    """``(measurability, reason)`` for one arm's shadow-write coverage.

    ``cohort_dates`` is the union of shadow-write dates across every OTHER
    currently-registered challenger this cycle — the population this arm is
    judged against. An arm that wrote zero shadows ever, or wrote none of the
    ``lag`` most-recent cohort dates, is UNMEASURABLE and must never render as
    a merely-thin/insufficient-data row (champion-challenger-policy §7.2,
    §3) — that was exactly the I9307-class defect this mirrors: an arm
    starved of comparable output rendering identically to one that merely has
    little.
    """
    cohort = sorted(cohort_dates)
    if not cohort:
        # No cohort at all — every arm is equally silent, a leaderboard-level
        # condition (already surfaced elsewhere), not this arm's fault.
        return MEASURABILITY_MEASURED, None
    scored = sorted(dates_scored)
    if not scored:
        return (
            MEASURABILITY_UNMEASURABLE,
            f"wrote 0 of {len(cohort)} cohort shadow date(s) — produced no "
            "comparable shadow output at all, which is an absence, not thin "
            "evidence",
        )
    recent = cohort[-lag:]
    if not (set(scored) & set(recent)):
        return (
            MEASURABILITY_UNMEASURABLE,
            f"last shadow write {scored[-1]}, but wrote none of the "
            f"{len(recent)} most recent cohort cycle(s) ({', '.join(recent)}) "
            "— has stopped producing comparable shadow output; its remaining "
            "dates are residue, not a thin-but-live record",
        )
    return MEASURABILITY_MEASURED, None


def load_promotion_history(bucket: str, s3_client=None) -> list[dict]:
    """The cutover history, OLDEST FIRST: ``[{"run_date", "champion_version_id"}]``.

    alpha-engine-config-I8219. ``predictor/model_zoo/promotions/{run_date}.json``
    is written once per rotation by ``model_zoo._write_promotion_marker`` and is
    never rewritten, so it is the durable record of which registry version held
    the champion pointer after each rotation. It is what lets a prediction
    artifact written BEFORE the ``champion_version_id`` stamp shipped
    (2026-08-25) still be attributed to the arm that actually produced it.

    Without this, the whole live-prediction history is unattributable and every
    per-arm realized series starts empty — the arm currently serving would have
    no matured outcome of its own until ~30 calendar days after the stamp
    shipped, which is exactly the unmeasurability -I8219 exists to remove.

    Best-effort: an unreadable prefix yields ``[]``, which renders every row
    ``unattributed`` — an honest absence, never a guess.
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    out: list[dict] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{_PROMOTIONS_PREFIX}/"):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key", "")
                if key.endswith(".json"):
                    keys.append(key)
        for key in sorted(keys):
            try:
                marker = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            except Exception:  # noqa: BLE001 — one unreadable marker is not the history
                log.warning("promotion history: could not read %s; skipped", key, exc_info=True)
                continue
            run_date = marker.get("run_date")
            champ = marker.get("champion_version_id_after")
            if run_date and champ:
                out.append({"run_date": str(run_date)[:10], "champion_version_id": champ})
    except Exception:  # noqa: BLE001 — absence is reported, never invented
        log.warning(
            "promotion history: %s/ unreadable — pre-stamp predictions will be "
            "reported unattributed (alpha-engine-config-I8219)",
            _PROMOTIONS_PREFIX, exc_info=True,
        )
        return []
    out.sort(key=lambda r: r["run_date"])
    return out


def champion_serving_on(prediction_date: str, history: list[dict]) -> str | None:
    """The version that held the pointer when ``prediction_date``'s batch ran.

    The rotation writes its marker under the trading day it ran FOR, and the
    preopen inference for that same day has already run by then, so the champion
    serving on date D is the ``champion_version_id_after`` of the newest marker
    whose ``run_date`` is STRICTLY EARLIER than D.

    Validated against live stamped artifacts 2026-08-31: predictions/2026-08-26
    is stamped ``v3.0-meta-2026-08-21-7d3d1cce`` (newest earlier marker
    2026-08-21) and predictions/2026-08-31 is stamped
    ``v3.0-meta-2026-08-14-119e069b`` (newest earlier marker 2026-08-28) — both
    reproduced exactly by this rule.

    ``None`` when no marker precedes the date: the history does not cover it and
    the row stays unattributed.
    """
    day = str(prediction_date)[:10]
    champ = None
    for row in history:
        if row["run_date"] < day:
            champ = row["champion_version_id"]
        else:
            break
    return champ


def _prediction_pair_rows(
    data: dict, fallback_date: str, *,
    version_id: str | None, attribution_source: str,
) -> list[dict]:
    """The pair rows for one predictions artifact — the SINGLE row constructor.

    alpha-engine-config-I8219. Both prediction loaders build rows here rather
    than each assembling its own dict, because they did not: ``_load_shadow_pairs``
    carried ``champion_version_id`` and ``_load_live_pairs`` silently omitted it,
    so every LIVE row reached the consumers keyed ``None``. Measured live
    2026-08-31 on ``observe_leaderboard/latest.json``: ``attribution_status``
    ``unstamped_predictions`` and ``realized_rank_ic_by_version`` a single
    ``version_id: null`` bucket, against artifacts that carry the stamp. The
    same rows feed ``training/arena_model_slot.build_series``, where an
    unclaimed version is dropped — so the M slot's arena was ranking arms on
    shadow output alone, with the serving champion's own live record discarded.

    One constructor is the fix that survives the class: a third loader cannot
    omit the field, because there is no second place to build the row.
    """
    date = data.get("date") or fallback_date
    return [
        {
            "date": date,
            "ticker": p.get("ticker"),
            ALPHA_FIELD: p.get(ALPHA_FIELD),
            "realized_alpha": None,
            "champion_version_id": version_id,
            "champion_version_id_source": attribution_source,
        }
        for p in data.get("predictions", []) or []
    ]


def attribution_coverage(pairs: list[dict]) -> dict:
    """How many rows/dates carry an attributed version, and from which source.

    alpha-engine-config-I8219 — the detection blindness, not the defect. Before
    this, "no prediction in the window carries champion_version_id" was emitted
    by an artifact whose own producer had dropped the field, and nothing in the
    payload could tell that apart from artifacts genuinely predating the stamp.
    A reader can now see which it is.
    """
    by_source: dict[str, int] = {}
    dates: set = set()
    dates_attributed: set = set()
    n_attributed = 0
    for p in pairs:
        src = p.get("champion_version_id_source") or ATTR_UNATTRIBUTED
        by_source[src] = by_source.get(src, 0) + 1
        day = str(p.get("date"))[:10]
        dates.add(day)
        if p.get("champion_version_id") is not None:
            n_attributed += 1
            dates_attributed.add(day)
    return {
        "n_pairs": len(pairs),
        "n_pairs_attributed": n_attributed,
        "n_dates": len(dates),
        "n_dates_attributed": len(dates_attributed),
        "by_source": dict(sorted(by_source.items())),
        "fully_unattributed": bool(pairs) and n_attributed == 0,
    }


def _load_shadow_pairs(bucket: str, version_id: str, n_days: int, s3_client=None) -> list[dict]:
    """Flatten the last ``n_days`` of a version's shadow predictions into
    (date, ticker, predicted_alpha) rows (realized_alpha filled downstream)."""
    import boto3
    import datetime

    s3 = s3_client or boto3.client("s3")
    today = datetime.date.today()
    pairs: list[dict] = []
    for d_offset in range(n_days):
        d = today - datetime.timedelta(days=d_offset)
        key = f"{_SHADOW_PREFIX}/{version_id}/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001 — most days have no file; skip cleanly
            continue
        # alpha-engine-config-I8175 — the arm that produced this day's batch.
        # Carried per pair so the realized rank-IC can be ATTRIBUTED to a version
        # instead of pooled across whatever the champion slot happened to hold.
        # The SHADOW directory names the producer, so the prefix is the
        # provenance here and the file's own `champion_version_id` (the LIVE
        # champion at write time) must NOT be read (policy §7.5).
        pairs.extend(_prediction_pair_rows(
            data, d.isoformat(),
            version_id=version_id, attribution_source=ATTR_SHADOW_PREFIX,
        ))
    return pairs


def _load_live_pairs(bucket: str, n_days: int, s3_client=None,
                     promotion_history: list[dict] | None = None) -> list[dict]:
    """Flatten the last ``n_days`` of LIVE champion predictions, ATTRIBUTED.

    alpha-engine-config-I8219. Every row carries ``champion_version_id`` and the
    ``champion_version_id_source`` it came from:

    * ``stamped`` — the artifact's own ``champion_version_id`` (live since
      2026-08-25). Preferred whenever present.
    * ``promotion_history`` — resolved from ``predictor/model_zoo/promotions/``
      for artifacts written before the stamp shipped. Without this, three months
      of live champion predictions are unattributable and every per-arm realized
      series is empty until ~30 calendar days after the stamp — which would leave
      -I8219's unmeasurability in place while looking fixed.
    * ``unattributed`` — no stamp and no marker precedes the date. Reported, never
      guessed, and never folded into another arm's series.

    Where a row is stamped AND the history covers it, the two are compared and a
    disagreement is logged loudly; the STAMP always wins, because it is the
    producer's own record of itself.
    """
    import boto3
    import datetime

    s3 = s3_client or boto3.client("s3")
    if promotion_history is None:
        promotion_history = load_promotion_history(bucket, s3_client=s3)
    today = datetime.date.today()
    pairs: list[dict] = []
    for d_offset in range(n_days):
        d = today - datetime.timedelta(days=d_offset)
        key = f"{_LIVE_PREDICTIONS_PREFIX}/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except Exception:  # noqa: BLE001
            continue
        date = str(data.get("date") or d.isoformat())[:10]
        stamped = data.get("champion_version_id")
        from_history = champion_serving_on(date, promotion_history)
        if stamped:
            version_id, source = stamped, ATTR_STAMPED
            if from_history and from_history != stamped:
                log.warning(
                    "attribution: predictions/%s is stamped %s but the promotion "
                    "history resolves %s — using the STAMP (the producer's own "
                    "record). A persistent disagreement means the marker write "
                    "and the inference stamp disagree about the cutover boundary "
                    "(alpha-engine-config-I8219).",
                    date, stamped, from_history,
                )
        elif from_history:
            version_id, source = from_history, ATTR_PROMOTION_HISTORY
        else:
            version_id, source = None, ATTR_UNATTRIBUTED
        pairs.extend(_prediction_pair_rows(
            data, d.isoformat(), version_id=version_id, attribution_source=source,
        ))
    return pairs


def _realized_rank_ic(pairs: list[dict]) -> tuple[float | None, int, int]:
    """Spearman rank-IC of ``predicted_alpha`` vs ``realized_alpha`` over pairs
    whose forward window has CLOSED (realized_alpha is finite). Rank-IC matches
    the canonical-alpha framework's OOS Spearman read (SOTA over raw Pearson on
    small matured samples). Returns ``(ic_or_None, n_matured, n_weeks_spanned)``.
    """
    import numpy as np
    from scipy.stats import spearmanr

    pred = np.array(
        [p.get(ALPHA_FIELD) if p.get(ALPHA_FIELD) is not None else np.nan for p in pairs],
        dtype=float,
    )
    real = np.array(
        [p.get("realized_alpha") if p.get("realized_alpha") is not None else np.nan for p in pairs],
        dtype=float,
    )
    mask = np.isfinite(pred) & np.isfinite(real)
    n = int(mask.sum())
    weeks = _distinct_weeks([p.get("date") for p, m in zip(pairs, mask) if m])
    if n < _MIN_PAIRS_FOR_IC:
        return None, n, weeks
    sp = spearmanr(pred[mask], real[mask])
    ic = float(sp.correlation) if np.isfinite(sp.correlation) else None
    return ic, n, weeks


def _distinct_weeks(dates) -> int:
    """Count distinct ISO (year, week) buckets among the prediction dates — the
    coverage breadth, reported for a reader. It gates nothing: minimum-week
    bars are abolished on this slot (champion-challenger-policy §5.0)."""
    import datetime

    weeks: set = set()
    for d in dates:
        if d is None:
            continue
        try:
            dt = datetime.date.fromisoformat(str(d)[:10])
        except Exception:  # noqa: BLE001
            continue
        iso = dt.isocalendar()
        weeks.add((iso[0], iso[1]))
    return len(weeks)


def _write_payload(payload: dict, bucket: str, trading_day: str | None,
                   *, write_latest: bool, s3_client=None) -> str:
    """Write a realized-edge payload to ``observe_leaderboard/{trading_day}.json``
    (+ ``latest.json`` mirror). Single write chokepoint shared by the champion
    monitor and the (retained) shadow leaderboard — keeps exactly two PUT sites in
    this file (pinned in tests/test_artifact_registry_coverage.py)."""
    import boto3

    s3 = s3_client or boto3.client("s3")
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    key = f"{OUTPUT_PREFIX}/{trading_day or 'latest'}.json"
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    log.info("observe_leaderboard written to s3://%s/%s", bucket, key)
    if write_latest:
        latest_key = f"{OUTPUT_PREFIX}/latest.json"
        s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
        log.info("observe_leaderboard mirrored to s3://%s/%s", bucket, latest_key)
    return key


def _attributed_realized_rank_ic(
    pairs: list[dict], serving_version_id: str | None
) -> dict:
    """The realized 21d rank-IC of the SERVING champion's OWN matured predictions.

    alpha-engine-config-I8175. The pooled read this module has always emitted is
    an unattributed SLOT aggregate, and under weekly rotation it structurally
    cannot describe the arm currently serving:

      * the window is the last ``DEFAULT_WINDOW_DAYS`` (42) calendar days of live
        predictions, with no version filter;
      * only pairs whose 21d forward window has CLOSED contribute to the IC;
      * so the matured subset is the OLDEST ~3 weeks of a 6-week window, while the
        champion rotates weekly — the serving version's own predictions are all
        still unmatured and contribute NOTHING.

    Measured 2026-08-22 against the live history: the ``-0.2626`` that raised
    alpha-engine-config-I8175 is attributable to ``spec-residual-mom-2026-07-17``
    (era mean realized IC -0.31) and ``spec-residual-mom-2026-07-10`` (-0.06), NOT
    to the serving ``v3.0-meta-2026-08-14-119e069b``, which has never had a single
    matured outcome. Gating a demote on the aggregate would remove a champion for
    its predecessors' results (`champion-challenger-policy` §7.5).

    Returns a dict carrying an explicit ``attribution_status`` — one of
    ``attributed`` / ``no_matured_outcomes`` / ``version_unknown`` /
    ``unstamped_predictions``. ``realized_rank_ic`` is None for every status but
    the first: an unmeasured statistic is reported as unmeasured, never as zero
    and never as the aggregate wearing the serving version's name.
    """
    if not serving_version_id:
        return {
            "version_id": None,
            "realized_rank_ic": None,
            "n_matured_outcomes": 0,
            "n_weeks_coverage": 0,
            "attribution_status": "version_unknown",
            "reason": (
                "the serving champion's registry version_id could not be resolved, "
                "so no prediction can be attributed to it"
            ),
        }

    stamped = [p for p in pairs if p.get("champion_version_id") is not None]
    if not stamped:
        return {
            "version_id": serving_version_id,
            "realized_rank_ic": None,
            "n_matured_outcomes": 0,
            "n_weeks_coverage": 0,
            "attribution_status": "unstamped_predictions",
            "reason": (
                "no prediction in the window could be attributed to a version — "
                "neither the artifact's own champion_version_id stamp nor the "
                "promotions/ cutover history covers any date in it. Read "
                "`attribution_coverage` before concluding the artifacts predate "
                "the stamp: until alpha-engine-config-I8219 this status was also "
                "produced by the LOADER dropping the field. The aggregate below "
                "is a SLOT read and must not be attributed to any one version."
            ),
        }

    own = [p for p in stamped if p.get("champion_version_id") == serving_version_id]
    ic, n, weeks = _realized_rank_ic(own) if own else (None, 0, 0)
    if ic is None:
        return {
            "version_id": serving_version_id,
            "realized_rank_ic": None,
            "n_matured_outcomes": n,
            "n_weeks_coverage": weeks,
            "attribution_status": "no_matured_outcomes",
            "reason": (
                f"the serving champion has {n} matured outcome(s) of its own in the "
                f"window (need >= {_MIN_PAIRS_FOR_IC} for a rank-IC). Under weekly "
                "rotation against a 21d horizon this is the EXPECTED state for a "
                "recently-promoted champion — it is not a verdict, and no gate may "
                "read it as one."
            ),
        }
    return {
        "version_id": serving_version_id,
        "realized_rank_ic": ic,
        "n_matured_outcomes": n,
        "n_weeks_coverage": weeks,
        "attribution_status": "attributed",
        "reason": (
            f"realized 21d rank-IC {ic:+.4f} over {n} matured outcomes / {weeks} "
            f"weeks produced by {serving_version_id} itself"
        ),
    }


def champion_line_of(version_id: str | None) -> str | None:
    """The champion LINE a registry version belongs to — its model_version prefix.

    ``v3.0-meta-2026-08-14-119e069b`` -> ``v3.0-meta``;
    ``spec-residual-mom-2026-07-17-f478ece3`` -> ``spec-residual-mom``.

    alpha-engine-config-I8175. The line matters because of a cadence arithmetic
    that makes per-VERSION attribution unusable as a gate input on its own: the
    champion rotates WEEKLY, while a 21-trading-day forward window takes ~30
    calendar days to close. A version promoted at T therefore has its first
    matured outcome around T+30, by which point four or five further rotations
    have happened and it is long out of service. The version currently serving
    will read ``no_matured_outcomes`` essentially always — the system rotates
    faster than it can measure.

    Measured 2026-08-22: every ``v3.0-meta`` champion from 2026-07-24 onward had
    ZERO matured outcomes, while the line itself had served continuously since
    2026-07-24. The line is what has actually been trading capital, and it is the
    coarser unit an exit gate can realistically read.

    Which of the two an exit gate consumes is a reserved decision, not a default —
    see ``cfg.MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE``.
    """
    if not version_id:
        return None
    parts = str(version_id).split("-")
    # Version ids are "<model_version>-<YYYY>-<MM>-<DD>-<sha>"; strip the dated
    # suffix by finding the first 4-digit year-looking component.
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit():
            return "-".join(parts[:i]) or None
    return str(version_id)


def _attributed_line_realized_rank_ic(
    pairs: list[dict], serving_version_id: str | None
) -> dict:
    """The realized 21d rank-IC of the serving champion's LINE — every version
    sharing its ``model_version`` prefix. See ``champion_line_of`` for why this
    exists alongside the per-version read (alpha-engine-config-I8175)."""
    line = champion_line_of(serving_version_id)
    if line is None:
        return {
            "line": None,
            "realized_rank_ic": None,
            "n_matured_outcomes": 0,
            "n_weeks_coverage": 0,
            "n_versions": 0,
            "attribution_status": "version_unknown",
            "reason": "the serving champion's version_id could not be resolved",
        }
    own = [p for p in pairs if champion_line_of(p.get("champion_version_id")) == line]
    if not own:
        return {
            "line": line,
            "realized_rank_ic": None,
            "n_matured_outcomes": 0,
            "n_weeks_coverage": 0,
            "n_versions": 0,
            "attribution_status": "unstamped_predictions",
            "reason": (
                "no prediction in the window is attributed to a version on this "
                "line, by stamp or by promotions/ history — see "
                "`attribution_coverage` (alpha-engine-config-I8219)"
            ),
        }
    versions = {p.get("champion_version_id") for p in own}
    ic, n, weeks = _realized_rank_ic(own)
    if ic is None:
        return {
            "line": line,
            "realized_rank_ic": None,
            "n_matured_outcomes": n,
            "n_weeks_coverage": weeks,
            "n_versions": len(versions),
            "attribution_status": "no_matured_outcomes",
            "reason": (
                f"the {line} line has {n} matured outcome(s) in the window (need >= "
                f"{_MIN_PAIRS_FOR_IC} for a rank-IC)"
            ),
        }
    return {
        "line": line,
        "realized_rank_ic": ic,
        "n_matured_outcomes": n,
        "n_weeks_coverage": weeks,
        "n_versions": len(versions),
        "attribution_status": "attributed",
        "reason": (
            f"realized 21d rank-IC {ic:+.4f} over {n} matured outcomes / {weeks} "
            f"weeks produced by {len(versions)} version(s) of the {line} line"
        ),
    }


def _realized_rank_ic_by_version(pairs: list[dict]) -> list[dict]:
    """Per-version realized 21d rank-IC over the window — the diagnostic that makes
    a mis-attributed aggregate visible instead of arguable
    (alpha-engine-config-I8175)."""
    buckets: dict = {}
    for p in pairs:
        buckets.setdefault(p.get("champion_version_id"), []).append(p)
    out = []
    for vid, rows in buckets.items():
        ic, n, weeks = _realized_rank_ic(rows)
        out.append({
            "version_id": vid,
            "realized_rank_ic": ic,
            "n_matured_outcomes": n,
            "n_weeks_coverage": weeks,
        })
    return sorted(out, key=lambda r: (r["version_id"] is None, str(r["version_id"])))


def per_date_rank_ic_by_version(pairs: list[dict]) -> dict:
    """``{version_id: {date: rank_ic_or_None}}`` — the arena's score series.

    alpha-engine-config-I9322. The same realized market-relative rank-IC
    ``_realized_rank_ic_by_version`` already computes, at the granularity the
    arena engine needs: **per date, not pooled**. Pooling across dates is what
    the arena's longest-common-window pairing exists to avoid, so this stops
    one step earlier and hands over the per-date series.

    A ``None`` value means the day's cross-section was too thin to form a
    correlation at all (``_MIN_PAIRS_FOR_IC``) — a well-formedness fact, which
    the caller records as a MISS. It never means "this arm scored zero", and it
    is never an evidence bar.
    """
    buckets: dict = {}
    for p in pairs:
        key = (p.get("champion_version_id"), str(p.get("date"))[:10])
        buckets.setdefault(key, []).append(p)
    out: dict = {}
    for (vid, date), rows in buckets.items():
        ic, _n, _weeks = _realized_rank_ic(rows)
        out.setdefault(vid, {})[date] = ic
    return out


def _resolve_serving_champion_version_id(bucket: str, s3_client=None) -> str | None:
    """The registry version_id of the champion serving right now. None (honest
    absence) when the registry is unreadable."""
    try:
        import boto3

        from model.registry import list_versions

        champs = list_versions(s3_client or boto3.client("s3"), bucket, stage="champion")
    except Exception:  # noqa: BLE001 — monitor is observability; absence is reported
        log.warning(
            "champion_monitor: could not resolve the serving champion version_id — "
            "the attributed read will report version_unknown "
            "(alpha-engine-config-I8175)",
            exc_info=True,
        )
        return None
    return champs[0].get("version_id") if champs else None


def build_champion_realized_monitor(
    bucket: str | None = None,
    n_days: int = DEFAULT_WINDOW_DAYS,
    horizon_days: int = DEFAULT_HORIZON,
    *,
    date_str: str | None = None,
    write_to_s3: bool = True,
    write_latest: bool = True,
    s3_client=None,
    live_pairs: list[dict] | None = None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
    serving_version_id: str | None = None,
) -> dict:
    """NOISE-CHASING MONITOR of the PROMOTED champion (config#671/#673/#1052).

    Relative-best promotion accepts the best-of-N challenger over the incumbent
    every week with no absolute-significance gate; the risk is chasing noise. This
    scores the LIVE champion's own ``predictor/predictions/{date}.json`` against
    REALIZED 21d sector-neutral alpha over a trailing window and flags
    ``chasing_noise`` when the realized rank-IC is non-positive (or unavailable
    once enough outcomes have matured). Pure OBSERVABILITY — it NEVER promotes,
    demotes, or allocates; the verdict is an alarm/diagnostic signal only.

    Test-injectable: ``live_pairs`` / ``prices_by_ticker`` / ``sector_map`` bypass
    S3 when provided.
    """
    bucket = bucket or cfg.S3_BUCKET
    horizon_days = horizon_days or int(getattr(cfg, "FORWARD_DAYS", 21))

    calendar_date = None
    trading_day = date_str
    try:
        from krepis.dates import now_dual
        dual = now_dual()
        calendar_date = dual.calendar_date
        trading_day = trading_day or (
            dual.trading_day.isoformat() if hasattr(dual.trading_day, "isoformat")
            else str(dual.trading_day)
        )
    except Exception:  # noqa: BLE001 — date resolution must not block the monitor
        log.warning("champion_monitor: now_dual failed; date fields best-effort", exc_info=True)

    if live_pairs is None:
        live_pairs = _load_live_pairs(bucket, n_days, s3_client=s3_client)

    if prices_by_ticker is None or sector_map is None:
        tickers = {p["ticker"] for p in live_pairs if p.get("ticker")}
        sector_map = sector_map or _load_sector_map_from_s3(bucket, s3_client=s3_client)
        bench_symbols = set(sector_map.values()) | {"SPY"}
        prices_by_ticker = prices_by_ticker or _load_prices_from_s3(
            bucket, tickers | bench_symbols, s3_client=s3_client,
        )

    compute_realized_alpha_for_pairs(
        live_pairs, horizon_days=horizon_days,
        prices_by_ticker=prices_by_ticker, sector_map=sector_map,
    )
    champ_ic, champ_n, champ_weeks = _realized_rank_ic(live_pairs)

    # alpha-engine-config-I8175 — the ATTRIBUTED read: the serving champion's own
    # matured predictions only. This is the ONLY quantity the auto-demote gate
    # (L4539) may act on; `champion` below stays the unattributed SLOT aggregate
    # it has always been, kept for continuity of the existing series and for the
    # noise-watch alert, but it is now labelled as such.
    if serving_version_id is None:
        serving_version_id = _resolve_serving_champion_version_id(
            bucket, s3_client=s3_client
        )
    attributed = _attributed_realized_rank_ic(live_pairs, serving_version_id)
    line_attributed = _attributed_line_realized_rank_ic(live_pairs, serving_version_id)
    by_version = _realized_rank_ic_by_version(live_pairs)

    # chasing_noise verdict: only assertable once enough outcomes have matured.
    # Non-positive realized rank-IC on >= MIN_REALIZED_OUTCOMES matured pairs ⇒ the
    # promoted champion shows no realized predictive edge → likely chasing noise.
    if champ_ic is None or champ_n < MIN_REALIZED_OUTCOMES:
        chasing_noise = None
        verdict_reason = (
            f"insufficient matured outcomes ({champ_n}; need >= {MIN_REALIZED_OUTCOMES}) "
            f"to assert a realized-edge verdict"
        )
    elif champ_ic <= 0.0:
        chasing_noise = True
        verdict_reason = (
            f"NOISE WATCH: champion realized rank-IC {champ_ic:.4f} <= 0 over "
            f"{champ_n} matured outcomes / {champ_weeks} weeks — relative-best "
            f"promotion may be chasing noise (observability, NOT a gate)"
        )
    else:
        chasing_noise = False
        verdict_reason = (
            f"healthy: champion realized rank-IC {champ_ic:.4f} > 0 over {champ_n} "
            f"matured outcomes / {champ_weeks} weeks"
        )

    payload = {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "date": trading_day,
        "kind": "champion_realized_monitor",
        "window_days": n_days,
        "horizon_days": horizon_days,
        "champion": {
            "realized_rank_ic": champ_ic,
            "n_matured_outcomes": champ_n,
            "n_weeks_coverage": champ_weeks,
            # alpha-engine-config-I8175 — say what this number is. It pools every
            # champion that served in the window, so it describes the SLOT, not the
            # arm named by `serving_champion_attributed` below.
            "scope": "champion_slot_aggregate_unattributed",
        },
        # The per-VERSION read: the serving champion's own matured predictions.
        # Under weekly rotation against a 21d horizon this is `no_matured_outcomes`
        # essentially always — the system rotates faster than it can measure.
        "serving_champion_attributed": attributed,
        # The per-LINE read: every version sharing the serving champion's
        # model_version prefix. The coarser unit an exit gate can realistically
        # read. Which of the two the gate consumes is a reserved decision
        # (cfg.MODEL_ZOO_DEMOTE_ATTRIBUTION_SCOPE) — see alpha-engine-config-I8175.
        "serving_line_attributed": line_attributed,
        # Every version with matured outcomes in the window, so a mis-attributed
        # aggregate is visible rather than arguable.
        "realized_rank_ic_by_version": by_version,
        # alpha-engine-config-I8219 — how much of the window could be attributed
        # at all, and from which source. This is the guard, not the fix: an
        # `unstamped_predictions` verdict is now checkable against the coverage
        # that produced it, so "the artifacts predate the stamp" and "the loader
        # dropped the field" can never again render identically
        # (champion-challenger-policy §7.2).
        "attribution_coverage": attribution_coverage(live_pairs),
        "chasing_noise": chasing_noise,
        "verdict_reason": verdict_reason,
        "note": (
            "Relative-best promotion (config#671/#673/#1052): the best-of-N "
            "challenger beating the champion by margin auto-promotes weekly; DSR is "
            "observability, not a gate. This monitor flags whether that selection is "
            "chasing noise via the PROMOTED champion's realized 21d rank-IC. It never "
            "promotes/demotes/allocates."
        ),
    }

    if write_to_s3:
        payload["s3_key"] = _write_payload(
            payload, bucket, trading_day, write_latest=write_latest, s3_client=s3_client,
        )

    log.info(
        "champion_realized_monitor: realized rank-IC=%s over %d matured / %d weeks; "
        "chasing_noise=%s",
        champ_ic, champ_n, champ_weeks, chasing_noise,
    )
    return payload


def build_observe_leaderboard(
    bucket: str | None = None,
    n_days: int = DEFAULT_WINDOW_DAYS,
    horizon_days: int = DEFAULT_HORIZON,
    *,
    write_to_s3: bool = True,
    write_latest: bool = True,
    s3_client=None,
    shadow_pairs_by_version: dict | None = None,
    live_pairs: list[dict] | None = None,
    prices_by_ticker: dict | None = None,
    sector_map: dict | None = None,
    date_str: str | None = None,
    registered_challenger_ids: list[str] | None = None,
    shadow_dates_by_version: dict[str, list[str]] | None = None,
) -> dict:
    """Score every shadow-run version against realized 21d alpha + emit the
    leaderboard.

    Pure MEASUREMENT — it never promotes or allocates, and since I9319 it no
    longer carries a promotion-readiness verdict at all. The minimum-week and
    minimum-cohort bars that verdict rested on are abolished
    (champion-challenger-policy §5.0); the M slot's pointer is decided by
    ``nousergon_lib.arena`` on realized market-relative rank-IC and recorded in
    ``arena/model/{date}.json``. ``beats_champion_realized`` remains as a point
    estimate for a reader, and is not a verdict.

    Test-injectable: ``shadow_pairs_by_version`` / ``live_pairs`` /
    ``prices_by_ticker`` / ``sector_map`` / ``registered_challenger_ids`` /
    ``shadow_dates_by_version`` bypass S3 when provided.

    alpha-engine-config-I9336: every currently-registered ``stage=challenger``
    arm gets an explicit row, even one that has written NO shadow prediction
    at all (previously entirely ABSENT from the leaderboard rather than
    reported). Each arm's ``measurability`` is ``measured`` or
    ``unmeasurable`` (see ``measurability_for_shadow_arm``) — an unmeasurable
    arm overrides the usual insufficient-data verdict and triggers one
    ops_alert per build naming every starved arm.
    """
    bucket = bucket or cfg.S3_BUCKET
    horizon_days = horizon_days or int(getattr(cfg, "FORWARD_DAYS", 21))

    # Resolve dual dates (best-effort; tests pass date_str directly).
    calendar_date = None
    trading_day = date_str
    try:
        from krepis.dates import now_dual
        dual = now_dual()
        calendar_date = dual.calendar_date
        trading_day = trading_day or (
            dual.trading_day.isoformat() if hasattr(dual.trading_day, "isoformat")
            else str(dual.trading_day)
        )
    except Exception:  # noqa: BLE001 — date resolution must not block the leaderboard
        log.warning("observe_leaderboard: now_dual failed; date fields best-effort", exc_info=True)

    # 1) Gather shadow + live prediction rows.
    if shadow_pairs_by_version is None:
        versions = _list_shadow_versions(bucket, s3_client=s3_client)
        shadow_pairs_by_version = {
            vid: _load_shadow_pairs(bucket, vid, n_days, s3_client=s3_client)
            for vid in versions
        }
    if live_pairs is None:
        live_pairs = _load_live_pairs(bucket, n_days, s3_client=s3_client)

    # 2) Load prices + sector_map once for the whole universe (shared across all
    #    versions; realized alpha is version-agnostic given (date, ticker)).
    all_pairs_for_universe = list(live_pairs)
    for _vp in shadow_pairs_by_version.values():
        all_pairs_for_universe.extend(_vp)
    if prices_by_ticker is None or sector_map is None:
        tickers = {p["ticker"] for p in all_pairs_for_universe if p.get("ticker")}
        sector_map = sector_map or _load_sector_map_from_s3(bucket, s3_client=s3_client)
        bench_symbols = set(sector_map.values()) | {"SPY"}
        prices_by_ticker = prices_by_ticker or _load_prices_from_s3(
            bucket, tickers | bench_symbols, s3_client=s3_client,
        )

    # 3) Fill realized alpha for the champion + each version's rows.
    compute_realized_alpha_for_pairs(
        live_pairs, horizon_days=horizon_days,
        prices_by_ticker=prices_by_ticker, sector_map=sector_map,
    )
    champ_ic, champ_n, champ_weeks = _realized_rank_ic(live_pairs)

    # alpha-engine-config-I9336: the CENSUS of arms that must be measured this
    # cycle — every currently-registered stage=challenger version, whether or
    # not it has ever written a shadow prediction. Without this a version
    # that never wrote a shadow simply never appears in
    # ``shadow_pairs_by_version`` (keyed off an S3 directory listing) and is
    # silently absent from the leaderboard rather than reported.
    if registered_challenger_ids is None:
        registered_challenger_ids = _list_registered_challenger_ids(bucket, s3_client=s3_client)
    registered_challenger_ids = list(dict.fromkeys(registered_challenger_ids))  # de-dup, order-stable

    all_vids = list(dict.fromkeys(list(shadow_pairs_by_version.keys()) + registered_challenger_ids))
    if shadow_dates_by_version is None:
        shadow_dates_by_version = {
            vid: _list_shadow_dates_for_version(bucket, vid, s3_client=s3_client)
            for vid in all_vids
        }
    # The cohort is the population a registered arm is judged against: every
    # OTHER currently-registered challenger's shadow-write dates. Falls back
    # to every known version's dates if the registered census is empty
    # (read failure) — degrade gracefully rather than suppress the check.
    cohort_source_vids = registered_challenger_ids or all_vids
    cohort_dates = sorted({
        d for vid in cohort_source_vids for d in shadow_dates_by_version.get(vid, [])
    })

    entries: list[dict] = []
    starved: list[str] = []
    for vid in all_vids:
        vpairs = shadow_pairs_by_version.get(vid, [])
        compute_realized_alpha_for_pairs(
            vpairs, horizon_days=horizon_days,
            prices_by_ticker=prices_by_ticker, sector_map=sector_map,
        )
        v_ic, v_n, v_weeks = _realized_rank_ic(vpairs)
        beats = (
            v_ic is not None and champ_ic is not None and v_ic >= champ_ic
        )
        measurability, measurability_reason = measurability_for_shadow_arm(
            shadow_dates_by_version.get(vid, []), cohort_dates,
        )
        if v_ic is None:
            verdict_reason = (
                f"unmeasurable: {v_n} matured pairs, below the {_MIN_PAIRS_FOR_IC} "
                f"a Spearman correlation needs to exist at all. Not a weak "
                f"result — no result (champion-challenger-policy §7.2)."
            )
        elif not beats:
            verdict_reason = (
                f"realized rank-IC {v_ic:.4f} < champion {champ_ic:.4f} over "
                f"{v_weeks} weeks / {v_n} matured outcomes (point estimate)"
            )
        else:
            verdict_reason = (
                f"realized rank-IC {v_ic:.4f} >= champion {champ_ic:.4f} over "
                f"{v_weeks} weeks / {v_n} matured outcomes (point estimate). "
                f"Whether that lead is SUPPORTED, and whether the pointer moves, "
                f"is decided by the anytime-valid sequence in "
                f"arena/model/{{date}}.json — never here (I9319/I9322)."
            )
        if measurability == MEASURABILITY_UNMEASURABLE:
            # Overrides the insufficient/soak-too-young framing above: this is
            # not "not enough evidence yet", it is "produced no comparable
            # output" — a defect that resolves by fixing the shadow write
            # path, never by waiting (champion-challenger-policy §7.2).
            verdict_reason = f"UNMEASURABLE: {measurability_reason}"
            starved.append(vid)
        entries.append({
            "version_id": vid,
            "realized_rank_ic": v_ic,
            "n_matured_outcomes": v_n,
            "n_weeks_coverage": v_weeks,
            "beats_champion_realized": beats,
            "verdict_reason": verdict_reason,
            "measurability": measurability,
            "measurability_reason": measurability_reason,
        })

    if starved:
        try:
            from ops_alerts import publish_ops_alert
            publish_ops_alert(
                message=(
                    f"observe_leaderboard: {len(starved)} registered "
                    f"challenger(s) UNMEASURABLE this cycle "
                    f"({', '.join(starved)}) — produced no comparable shadow "
                    f"output within the last {MEASURABILITY_LAG_CYCLES} cohort "
                    f"cycle(s) (alpha-engine-config-I9336). Never renders as a "
                    f"merely-thin row."
                ),
                # DELIBERATELY "warning", not "error"/"critical" — kept as a
                # LITERAL keyword argument (not a variable) because
                # nousergon-data's alert_class_pr_guard.py resolves a call
                # site's registry severity by a static regex over the
                # quoted-string form of this exact keyword argument; a
                # non-literal argument resolves to `dynamic` and would force
                # the alert_classes row below to declare `severities:
                # [dynamic]` instead of the true, single, deliberate value.
                # (Do not write that regex's own pattern in a comment near
                # this call — the guard scans raw source text including
                # comments, and an example match here is indistinguishable
                # from a second call site to it.)
                #
                # A starved shadow arm is a MEASUREMENT-COVERAGE signal
                # (fixed by repairing the shadow write path, or by waiting for
                # `_select_challengers_for_cycle`'s rotation to reach it) —
                # never a trading-halt condition, so it must never land as an
                # immediate page in the one operator chat. That conflation
                # (severity gating the buzz while every severity landed in
                # the same destination) is the exact 2026-08-28 alert-
                # destination-arc lesson (fleet memory
                # project_alert_destination_arc_260829): severity does not
                # by itself pick a destination tier, so this row must pair a
                # non-paging severity with the batched routing response, not
                # merely "not critical".
                #
                # Registered as `class: predictor_shadow_leaderboard_
                # unmeasurable_arm` in nousergon-data's
                # infrastructure/overseer/playbooks.yaml::alert_classes with
                # `severities: [warning]`, `intake: bus`, `response:
                # drain-queue` — the batched alert-drain queue, NOT
                # `response: operator` (declared human-only/paging). Companion
                # PR required per that file's CI guard
                # (.github/workflows/alert-class-pr-guard.yml); see
                # alpha-engine-config-I9336.
                severity="warning",
                source="alpha-engine-predictor/analysis/observe_leaderboard.py::build_observe_leaderboard",
                dedup_key=f"predictor_shadow_unmeasurable_{trading_day}",
            )
        except Exception:  # noqa: BLE001 — alert is best-effort observability
            log.warning("observe_leaderboard: unmeasurable-arm alert itself failed", exc_info=True)

    # Rank by realized IC desc (None last). There is deliberately no
    # promotion-readiness key to sort on any more: this artifact MEASURES, the
    # arena cycle DECIDES, and a diagnostic that ranks by its own readiness
    # verdict is one edit away from being read as the decision (I9319).
    entries.sort(
        key=lambda e: -(e["realized_rank_ic"] if e["realized_rank_ic"] is not None else -1e9)
    )

    payload = {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "date": trading_day,
        "window_days": n_days,
        "horizon_days": horizon_days,
        # No `soak_criteria` block: the minimum-week and minimum-cohort bars
        # it declared are abolished for this slot (champion-challenger-policy
        # §5.0, alpha-engine-config-I9319). The evidence bar is the
        # anytime-valid confidence sequence, and it lives in the arena cycle.
        "decision_authority": {
            "artifact": "arena/model/{date}.json",
            "note": (
                "This leaderboard MEASURES realized edge. It decides nothing. "
                "The M slot's pointer is decided by nousergon_lib.arena on "
                "realized market-relative rank-IC (I9322)."
            ),
        },
        "champion": {
            "realized_rank_ic": champ_ic,
            "n_matured_outcomes": champ_n,
            "n_weeks_coverage": champ_weeks,
        },
        "entries": entries,
    }

    if write_to_s3:
        payload["s3_key"] = _write_payload(
            payload, bucket, trading_day, write_latest=write_latest, s3_client=s3_client,
        )

    log.info(
        "observe_leaderboard: %d version(s) scored (champion realized "
        "rank-IC=%s over %d matured / %d weeks). Measurement only — the "
        "pointer decision is arena/model/{date}.json.",
        len(entries), champ_ic, champ_n, champ_weeks,
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bucket", default=None,
                        help=f"S3 bucket. Default: cfg.S3_BUCKET ({cfg.S3_BUCKET}).")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"Trailing prediction-history window in days. Default: {DEFAULT_WINDOW_DAYS}.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Forward-realized window in trading days. Default: {DEFAULT_HORIZON}.")
    parser.add_argument("--no-write", action="store_true",
                        help="Compute + print the leaderboard but do not write to S3.")
    args = parser.parse_args()

    result = build_observe_leaderboard(
        bucket=args.bucket, n_days=args.window, horizon_days=args.horizon,
        write_to_s3=not args.no_write,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
