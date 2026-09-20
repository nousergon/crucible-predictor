#!/usr/bin/env python3
"""Backfill ``arena/model/{date}.verdict`` for cycles written before it existed.

alpha-engine-config-I11101. ``emit_cycle`` writes the verdict projection
alongside every cycle from crucible-predictor-PR631 onward, but cycles written
before that have no projection — and the weekly SF now READS it, so a re-run
over one of those dates degrades on a cycle that was in fact decided.

The projection is derived from each cycle's own ``decision.status``. Nothing is
computed, inferred or defaulted here: a cycle that cannot be read, or whose
status is outside ``arena_cycle.schema.json``'s enum, is SKIPPED and named. A
missing projection makes the SF degrade honestly; a wrong one would make it
route a cycle on a verdict nobody reached, so skipping is always the safer
failure.

Idempotent. ``--write`` is required; without it this prints the plan and exits.

    python3 scripts/backfill_arena_verdicts.py                 # plan only
    python3 scripts/backfill_arena_verdicts.py --write
"""
from __future__ import annotations

import argparse
import json
import sys

import boto3

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from training.arena_model_slot import (  # noqa: E402
    CYCLE_PREFIX,
    _emit_verdict_projection,
    _verdict_vocabulary,
)

BUCKET = "alpha-engine-research"


def _backfill_latest(s3, bucket: str, vocabulary) -> None:
    """Set ``latest.verdict`` from ``latest.json``, which is the pointer the
    slot itself maintains — never from whatever this run's last dated cycle
    was."""
    key = f"{CYCLE_PREFIX}/latest.json"
    try:
        doc = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP   latest.verdict: {key} unreadable: {exc}")
        return
    status = (doc.get("decision") or {}).get("status")
    if status not in vocabulary:
        print(f"  SKIP   latest.verdict: decision.status={status!r} outside vocabulary")
        return
    as_of = doc.get("as_of")
    _emit_verdict_projection(s3, bucket, doc, as_of=as_of, mirror_latest=True)
    print(f"  WRITE  latest.verdict -> {status}  (from latest.json, as_of={as_of})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--write", action="store_true",
                    help="actually PUT; without it, plan only")
    args = ap.parse_args()

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    cycles, existing = [], set()
    for page in paginator.paginate(Bucket=args.bucket, Prefix=f"{CYCLE_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".verdict"):
                existing.add(key)
            elif key.endswith(".json") and not key.endswith("/register.json"):
                # latest.json is a POINTER, not a dated cycle. Walking it here
                # would derive as_of="latest" from the filename and write the
                # same key _backfill_latest writes — right by coincidence of
                # naming, wrong the moment either side changes, and a
                # double-write either way. It is handled once, below, from the
                # document's own as_of.
                if not key.endswith("/latest.json"):
                    cycles.append(key)

    vocabulary = _verdict_vocabulary()
    planned, skipped = [], []
    for key in sorted(cycles):
        as_of = key.rsplit("/", 1)[-1][: -len(".json")]
        if f"{CYCLE_PREFIX}/{as_of}.verdict" in existing:
            continue
        try:
            doc = json.loads(s3.get_object(Bucket=args.bucket, Key=key)["Body"].read())
        except Exception as exc:  # noqa: BLE001
            skipped.append((as_of, f"unreadable: {exc}"))
            continue
        status = (doc.get("decision") or {}).get("status")
        if status not in vocabulary:
            skipped.append((as_of, f"decision.status={status!r} outside {sorted(vocabulary)}"))
            continue
        planned.append((as_of, doc, status))

    for as_of, _doc, status in planned:
        print(f"  {'WRITE' if args.write else 'would write'}  {as_of} -> {status}")
    for as_of, why in skipped:
        print(f"  SKIP   {as_of}: {why}")
    if not planned and not skipped:
        print("  nothing to do — every cycle already has its projection")

    if not args.write and planned:
        # Named in the plan so the operator sees the full set of writes, not
        # only the dated ones.
        print("  would write  latest.verdict (from latest.json)")

    if args.write:
        for as_of, doc, _status in planned:
            # mirror_latest=False: latest.verdict is set ONCE below, from
            # latest.json, not by whichever dated cycle happens to be last in
            # this loop.
            _emit_verdict_projection(s3, args.bucket, doc, as_of=as_of,
                                     mirror_latest=False)
        _backfill_latest(s3, args.bucket, vocabulary)
    elif planned:
        print("\n(plan only — re-run with --write)")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
