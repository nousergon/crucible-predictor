"""model/retired_arms.py — arms retired from the shadow rotation, and why.

alpha-engine-config-I10180.

The decision this file records
------------------------------
The five ``spec-sota-combine-2026-07-*`` arms are RETIRED from shadowing, not
kept as labelled negative controls. Their bundles are NOT deleted — they are
evidence, and ``model.registry`` keeps them listable — but they no longer occupy
the rotation and their historical shadow rows are marked at the leaderboard
reader so a retired arm's scores can never silently mix with a live one's.

Why retired and not a control: a negative control has to be READ by something to
be worth its slot, and nothing reads these. Meanwhile ``SHADOW_VERSIONS_MAX_N``
is 5 and there were five of them, so they occupied the entire rotation — five
slots per day not spent on an arm that could win — while the realized-edge
leaderboard scored their constant-alpha output as if it were a model's.

Measured 2026-09-08 (full evidence: alpha-engine-config-I10178), on the served
artifacts under ``predictor/predictions_shadow/{version_id}/2026-09-08.json``:

    spec-sota-combine-2026-07-17-bb8dc165   6 distinct alpha / 29   sd 0.001959
    spec-sota-combine-2026-07-24-8578f8ae   6 / 29                  sd 0.000931
    spec-sota-combine-2026-07-29-8c530ef0   6 / 29                  sd 0.000937
    spec-sota-combine-2026-07-30-19204f35   6 / 29                  sd 0.000237
    spec-sota-combine-2026-07-31-0c5140d5   6 / 29                  sd 0.000942

    (live champion, same batch: 29 distinct, sd 0.005819)

Why the retirement is declared HERE and not derived from a manifest number
--------------------------------------------------------------------------
``xsec_variance_share`` (crucible-predictor-PR611) is the general mechanism, and
``_select_challengers_for_cycle`` does drop any arm whose manifest carries a
below-floor value. It cannot reach these five, and the reason is worth stating
because it is the same reason for every pre-2026-09-08 vintage:

* their manifests predate the metric, so they carry nothing;
* it is not computable for them after the fact either. The metric needs the
  panel the arm was FITTED on, and only one fitted panel is persisted
  (``predictor/diagnostics/oos_rows/v3.0-meta/2026-09-04.parquet``). The
  top-level ``oos_rows/{date}.parquet`` files are the ~150k-row full-universe
  panels with the research columns zeroed, not a fitted matrix.
* scored on the 2026-09-04 panel instead — which is NOT their own — those five
  return 0.918, 0.836, 0.722, 0.638 and 0.405, all comfortably above the 0.10
  floor. That number is a cross-vintage read and it is not evidence about these
  arms; publishing it as if it were would be worse than having none.

So the retirement rests on the served-artifact evidence above, which is direct,
and is recorded here rather than left implicit — which is exactly the branch
alpha-engine-config-I10180 offers for pre-2026-09-08 vintages. An absent
``xsec_variance_share`` is never read as healthy: it is uncomputable and named.

Why a code constant rather than an S3 lineage patch
----------------------------------------------------
``model.registry`` has an ``archived`` stage and ``_patch_stage`` could set it.
That is an S3 write, so it would be an operator step performed after a merge —
and a PR must be deployable by the merge button alone (a tracked issue recording
a command is not a mechanism that fires). A declared constant deploys with the
code, is reviewable in the diff, and is the durable statement of WHY.
"""

from __future__ import annotations

#: ``version_id -> reason``. An arm listed here is excluded from the shadow
#: rotation and marked ``retired`` on the observe leaderboard. Bundles are kept.
RETIRED_SHADOW_VERSIONS: dict[str, str] = {
    vid: (
        "cross-sectionally degenerate: emitted 6 distinct predicted_alpha "
        "across 29 tickers on 2026-09-08 (24 identical), alpha stdev "
        f"{sd}, against the live champion's 29 distinct / 0.005819 on the "
        "same batch. Retired from the shadow rotation on the served-artifact "
        "evidence; xsec_variance_share is uncomputable for this vintage "
        "because its own fitted panel is not persisted "
        "(alpha-engine-config-I10180, root cause -I9269 / -I9255)."
    )
    for vid, sd in (
        ("spec-sota-combine-2026-07-17-bb8dc165", "0.001959"),
        ("spec-sota-combine-2026-07-24-8578f8ae", "0.000931"),
        ("spec-sota-combine-2026-07-29-8c530ef0", "0.000937"),
        ("spec-sota-combine-2026-07-30-19204f35", "0.000237"),
        ("spec-sota-combine-2026-07-31-0c5140d5", "0.000942"),
    )
}

#: The date the decision above was taken, for anyone reading a shadow artifact
#: written before it.
RETIRED_ON = "2026-09-08"


def is_retired(version_id: str | None) -> bool:
    return bool(version_id) and version_id in RETIRED_SHADOW_VERSIONS


def retirement_reason(version_id: str | None) -> str | None:
    return RETIRED_SHADOW_VERSIONS.get(version_id or "")
