#!/usr/bin/env python3
"""Backfill ``xsec_sd`` over champion history, on a HELD-CONSTANT panel.

alpha-engine-config-I10185, deliverable 4.

What it does
------------
Scores every registered coefficient vector matching ``--prefix`` on ONE fitted
meta-training design matrix, with
``training/xsec_variance_share.cross_sectional_variance_share`` — the same
function the fit-time detector and the promotion veto's reference use. Because
the panel is identical for every row, the resulting series measures the MODELS
and not the vintages they were each fitted on.

Why not each model's own panel
------------------------------
Only one fitted panel is persisted:
``predictor/diagnostics/oos_rows/v3.0-meta/2026-09-04.parquet`` (and its
``latest`` alias; the per-spec ``spec-sota-combine/`` and ``spec-residual-mom/``
prefixes hold the same 2026-09-04 date, and the ``v3.0-meta`` and
``spec-sota-combine`` copies are byte-identical by ETag). The top-level
``oos_rows/{date}.parquet`` files are the ~150k-row full-universe panels with
the research columns zeroed — NOT the fitted matrix, and scoring a coefficient
vector on one would produce a number that looks like an ``xsec_sd`` and is not.

So a per-vintage backfill is not available at any price, and a cross-vintage one
would be dishonest. The held-constant panel is the honest artifact, and it is
labelled as such in the output (``panel_key``, ``basis``).

``xsec_variance_share`` is emitted alongside but is NOT the point of this
backfill: every measured champion sits far above its 0.10 floor (0.70 .. 0.98),
so the series carries no separation. ``xsec_sd`` separates the two known-bad
promotions from all eighteen others.

Usage
-----
    python scripts/backfill_xsec_sd.py --prefix v3.0-meta- --write

Read-only without ``--write``. With ``--write`` it puts ONE small JSON to
``predictor/diagnostics/xsec_sd_backfill/{panel_date}.json`` — a diagnostics
artifact. It never touches a registry manifest: those are promoted, immutable
evidence, and a backfilled number written into one would be indistinguishable
from a number the trainer produced at fit time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_PANEL_KEY = "predictor/diagnostics/oos_rows/v3.0-meta/latest.parquet"
REGISTRY_PREFIX = "predictor/registry/"
OUT_PREFIX = "predictor/diagnostics/xsec_sd_backfill"


def _list_versions(s3, bucket: str, prefix: str) -> list[str]:
    out: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=REGISTRY_PREFIX, Delimiter="/",
    ):
        for cp in page.get("CommonPrefixes", []):
            vid = cp["Prefix"][len(REGISTRY_PREFIX):].rstrip("/")
            if vid.startswith(prefix):
                out.append(vid)
    return sorted(out)


def _coefficients(manifest: dict) -> dict:
    mm = (manifest.get("models") or {}).get("meta_model") or {}
    return dict(mm.get("coefficients") or manifest.get("meta_coefficients") or {})


def main(argv=None) -> int:
    import boto3
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.xsec_variance_share import cross_sectional_variance_share

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--panel-key", default=DEFAULT_PANEL_KEY)
    ap.add_argument("--panel-date", default=None,
                    help="rotation date this panel belongs to; inferred from a "
                         "dated --panel-key, else the panel's last row")
    ap.add_argument("--prefix", default="v3.0-meta-",
                    help="registry version_id prefix to score")
    ap.add_argument("--write", action="store_true",
                    help="put the result to the diagnostics prefix")
    args = ap.parse_args(argv)

    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        s3.download_fileobj(args.bucket, args.panel_key, tmp)
        tmp.flush()
        df = pd.read_parquet(tmp.name)
    dates = df["date"].astype(str).tolist()
    # The ROTATION date this panel belongs to — taken from the key when the key
    # is dated, else supplied. The default key is a `latest` alias, and an
    # artifact called `latest.json` records nothing about which vintage it
    # measured. `panel_last_date` is a different and also-recorded quantity: the
    # newest row in the panel, which trails the rotation date by the forward
    # horizon plus embargo (measured: rotation 2026-09-04, last row 2026-08-06).
    _key_date = args.panel_key.rsplit("/", 1)[-1].removesuffix(".parquet")
    panel_date = args.panel_date or (
        _key_date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", _key_date)
        else max(dates)[:10]
    )

    rows: list[dict] = []
    for vid in _list_versions(s3, args.bucket, args.prefix):
        try:
            raw = s3.get_object(
                Bucket=args.bucket,
                Key=f"{REGISTRY_PREFIX}{vid}/manifest.json",
            )["Body"].read()
            coefs = _coefficients(json.loads(raw))
        except Exception as exc:  # noqa: BLE001 — reported per row, never silent
            rows.append({"version_id": vid, "status": "manifest_unreadable",
                         "reason": str(exc)})
            continue
        names = [k for k in coefs if k != "intercept"]
        if not names:
            rows.append({"version_id": vid, "status": "no_coefficients"})
            continue
        # Features a model was fitted on that this panel does not carry are
        # supplied as 0.0 and NAMED. On the measured history every such feature
        # (`guidance_direction`, `risk_factor_count_delta_raw`,
        # `management_tone_zscore`) carries coefficient exactly 0.0 and
        # `panel_std` 0.0 in its own manifest, so it moves neither variance —
        # but an unnamed zero-fill is the alpha-engine-config-I5949 trap and
        # must never be silent.
        absent = [n for n in names if n not in df.columns]
        import numpy as np
        X = np.column_stack([
            df[n].to_numpy(dtype=float) if n in df.columns
            else np.zeros(len(df), dtype=float)
            for n in names
        ])
        summary = cross_sectional_variance_share(
            X, dates, names, {n: float(coefs[n]) for n in names},
            intercept=float(coefs.get("intercept", 0.0)),
        )
        rows.append({
            "version_id": vid,
            "status": summary.get("status"),
            "verdict": summary.get("verdict"),
            "xsec_sd": summary.get("xsec_sd"),
            "xsec_variance_share": summary.get("xsec_variance_share"),
            "total_sd": summary.get("total_sd"),
            "features_absent_from_panel": absent,
        })

    payload = {
        "generated_by": "crucible-predictor/scripts/backfill_xsec_sd.py",
        "tracked": "alpha-engine-config-I10185",
        "panel_key": args.panel_key,
        "panel_date": panel_date,
        "panel_last_date": max(dates)[:10],
        "n_rows": int(len(df)),
        "n_dates": len(set(dates)),
        "prefix": args.prefix,
        "basis": "coefficients_scored_on_a_single_held_constant_panel",
        "caveat": (
            "Every row is scored on the SAME panel, so the series compares "
            "models rather than vintages. It is NOT each model's own fit-time "
            "number — only the 2026-09-04 fitted panel is persisted."
        ),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.write:
        key = f"{OUT_PREFIX}/{panel_date}.json"
        s3.put_object(
            Bucket=args.bucket, Key=key,
            Body=json.dumps(payload, indent=2).encode(),
            ContentType="application/json",
        )
        print(f"wrote s3://{args.bucket}/{key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
