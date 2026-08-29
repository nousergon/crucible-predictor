"""training/data_completeness.py — input completeness, measured in ROWS.

Why this module exists (alpha-engine-config-I9287 / -I9289 / -I9290):

``store.arctic_reader.download_from_arctic`` reported ``coverage_ratio =
n_written / n_expected`` — a count of SYMBOLS, never of rows per symbol.
On 2026-08-15 the ArcticDB ``macro/VIX3M`` series held **16 rows** against
~2514 for SPY/VIX/TNX/IRX. The symbol was present, so ``coverage_ratio``
read ``1.0`` in every affected manifest, ``data_coverage_degraded`` read
``False``, and the training run was recorded as fully-covered. Downstream,
``model/regime_predictor.build_features`` ffilled the 16-row series into
``vix_term_slope`` and a blanket ``dropna()`` truncated the whole regime
frame from 799 dates to 16 — after which
``training/meta_trainer.py`` zero-filled all six ``macro_*`` meta features
plus ``regime_intensity_z`` for every OOS date. Two full rotations
(2026-08-22, 2026-08-29) trained and scored every model-zoo arm on a
constant-zero macro block while every artifact asserted complete coverage.

A symbol-presence check cannot see any of that. This module measures the
three things that actually decide whether an input is usable:

  **rows**       — how much history the symbol holds, against the reference
                   series (SPY) that the training panel is indexed on.
  **last_date**  — whether the symbol is still being written. ``macro/HYOAS``
                   is rewritten weekly but its last row is 2026-05-07: a
                   FROZEN producer, invisible to any freshness check that
                   looks at object mtime rather than at the data.
  **date span**  — whether the symbol's own dates cover the window the
                   model is trained and scored over. A series that starts
                   after the OOS window opens cannot inform it.

Design rules, all load-bearing:

* **Nothing here substitutes a value.** The failure this module exists to
  catch IS a silent substitution (a constant ``0.0`` written where a macro
  reading belonged). Every finding is either a hard failure that raises
  ``DataCompletenessError`` or a NAMED degradation recorded on the manifest
  under ``data_completeness.degradations`` — never a quiet default.
* **Severity is a property of the input, declared in the register below**,
  not decided at the call site. ``required`` inputs raise; ``optional``
  inputs record a named degradation and name the features they starve.
* **The register is the single list of what training consumes.** A new
  macro series gets a row here in the same PR that consumes it, or the
  completeness gate is blind to it — which is precisely how VIX3M ran dark.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timezone

log = logging.getLogger(__name__)

__all__ = [
    "DataCompletenessError",
    "InputSpec",
    "INPUT_REGISTER",
    "REFERENCE_SYMBOL",
    "measure_symbol_coverage",
    "evaluate_completeness",
    "assert_trainable",
    "summarize_universe_rows",
]


class DataCompletenessError(RuntimeError):
    """A REQUIRED training input is absent, too short, or frozen.

    Raised before training starts. Deliberately not caught anywhere on the
    training path: a run that cannot see its inputs must fail loud rather
    than emit a model whose manifest claims coverage it did not have.
    """


# The reference series every other input is measured against. SPY is the
# benchmark leg of the market-relative label and ``train_handler`` already
# refuses to train without it, so it is the one series guaranteed present
# and the honest denominator for "how much history should this have?".
REFERENCE_SYMBOL = "SPY"

# Headline-status precedence, most-actionable first. `stale` outranks
# `short_history` deliberately: a short series is frequently a declared
# upstream limit, a frozen one is always a live regression.
_STATUS_PRECEDENCE = (
    "absent", "stale", "insufficient_span", "short_history", "unknown_dates",
)


@dataclass(frozen=True)
class InputSpec:
    """One declared training input and the bar it must clear.

    Attributes
    ----------
    symbol : str
        ArcticDB symbol / parquet stem, e.g. ``"VIX3M"``.
    severity : {"required", "optional"}
        ``required`` — a failure raises ``DataCompletenessError`` and no
        model is trained. ``optional`` — a failure is recorded as a named
        degradation on the manifest, naming ``features`` as starved.
    consumers : tuple[str, ...]
        Human-readable call sites, so a finding names WHERE it bites.
    features : tuple[str, ...]
        The feature columns this input feeds. A degradation names these so
        a reader does not have to trace the graph to learn what went dark.
    min_rows_ratio : float
        Floor on ``rows / reference_rows``. Below it the series is
        ``short_history``. Defaults to 0.90 — a macro series read off the
        same calendar as SPY should be within a few percent of it.
    max_staleness_days : int
        Calendar days between the input's ``last_date`` and the reference's
        ``last_date`` before the series is ``stale``. Weekly-published
        FRED series get a wider window than daily market data.
    """

    symbol: str
    severity: str
    consumers: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    min_rows_ratio: float = 0.90
    max_staleness_days: int = 7


# ── The register ──────────────────────────────────────────────────────────
# Every series `training/meta_trainer.py::run_meta_training` loads by name
# via ``_load_close``. Adding a ``_load_close`` call without adding a row
# here leaves the new input outside the gate — the VIX3M failure mode.
#
# `required` = the model is not trainable without it. VIX3M is required
# BECAUSE of 2026-08-15: it was treated as an optional nicety, its collapse
# silently deleted the entire regime frame, and two rotations scored every
# arm on constant zeros.
INPUT_REGISTER: tuple[InputSpec, ...] = (
    InputSpec(
        symbol="SPY", severity="required",
        consumers=("regime_predictor.build_features", "compute_labels benchmark"),
        features=("spy_20d_return", "spy_20d_vol", "market_breadth"),
    ),
    InputSpec(
        symbol="VIX", severity="required",
        consumers=("regime_predictor.build_features",),
        features=("vix_level", "vix_term_slope", "vix_vix3m_ratio"),
    ),
    InputSpec(
        symbol="VIX3M", severity="required",
        consumers=("regime_predictor.build_features",),
        features=("vix_term_slope", "vix_vix3m_ratio"),
    ),
    InputSpec(
        symbol="TNX", severity="required",
        consumers=("regime_predictor.build_features",),
        features=("yield_curve_slope", "yield_curve_10y_2y"),
    ),
    InputSpec(
        symbol="IRX", severity="required",
        consumers=("regime_predictor.build_features",),
        features=("yield_curve_slope",),
    ),
    # FRED-sourced credit/curve macros. Optional: `build_features` has a
    # declared neutral fallback for each, so their absence does not delete
    # the regime frame the way VIX3M's collapse did. Their absence is still
    # a NAMED degradation — a run that trains without a credit block is not
    # the same experiment as one that trains with it, and the leaderboard
    # must say so rather than let the two compare as equals.
    InputSpec(
        symbol="TWO", severity="optional",
        consumers=("regime_predictor.build_features",),
        features=("yield_curve_10y_2y",),
        max_staleness_days=10,
    ),
    InputSpec(
        symbol="HYOAS", severity="optional",
        consumers=("regime_predictor.build_features",),
        features=("hy_oas_level", "hy_oas_change_21d"),
        # ICE BofA HY OAS is license-gated to 2023+ on FRED, so a full-length
        # row bar would report `short_history` on EVERY run and drown the
        # finding that actually matters here — the series froze on
        # 2026-05-07 while its collector kept rewriting it weekly (-I9287).
        min_rows_ratio=0.25,
        # FRED publishes HYOAS daily but with a lag; 10 days is generous
        # and still catches the live 2026-05-07 freeze by ~3.7 months.
        max_staleness_days=10,
    ),
    InputSpec(
        symbol="BAA10Y", severity="optional",
        consumers=("regime_predictor.build_features",),
        features=("baa10y_level", "baa10y_change_21d"),
        max_staleness_days=10,
    ),
    InputSpec(
        symbol="GLD", severity="optional",
        consumers=("meta_trainer._load_close",), features=(),
        max_staleness_days=10,
    ),
    InputSpec(
        symbol="USO", severity="optional",
        consumers=("meta_trainer._load_close",), features=(),
        max_staleness_days=10,
    ),
)


def _as_date(value) -> "_date | None":
    """Coerce a pandas Timestamp / datetime / ISO string to ``date``."""
    if value is None:
        return None
    if isinstance(value, _date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def _iso(value) -> "str | None":
    d = _as_date(value)
    return d.isoformat() if d is not None else None


def measure_symbol_coverage(df) -> dict:
    """Return ``{rows, first_date, last_date}`` for one loaded DataFrame.

    A DataFrame with a non-date index (or an empty one) yields ``None``
    dates rather than a guess — an unparseable index is reported as such,
    never silently replaced with the run date.
    """
    rows = int(len(df)) if df is not None else 0
    first = last = None
    if rows:
        try:
            idx = df.index
            first, last = _iso(idx[0]), _iso(idx[-1])
        except Exception:  # noqa: BLE001 — index shape is the FINDING here.
            # Recorded as unknown dates on a non-zero row count, which the
            # evaluator reports as `unknown_dates`. Swallowing nothing: the
            # row count survives and the date fields say they are unknown.
            log.warning(
                "data_completeness: could not read first/last date off a "
                "%d-row frame — reporting unknown_dates, not a substituted "
                "date.", rows, exc_info=True,
            )
    return {"rows": rows, "first_date": first, "last_date": last}


@dataclass
class _Finding:
    symbol: str
    status: str
    severity: str
    reason: str
    features: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "input": self.symbol,
            "status": self.status,
            "severity": self.severity,
            "reason": self.reason,
            "features_affected": list(self.features),
        }


def evaluate_completeness(
    per_symbol: dict,
    *,
    register: "tuple[InputSpec, ...] | None" = None,
    oos_window: "tuple[str, str] | None" = None,
) -> dict:
    """Grade every registered input against the reference series.

    Parameters
    ----------
    per_symbol : dict
        ``{symbol: {"rows": int, "first_date": str|None, "last_date": str|None}}``
        as produced by ``store.arctic_reader.download_from_arctic``.
    register : tuple[InputSpec, ...], optional
        Defaults to ``INPUT_REGISTER``.
    oos_window : (str, str), optional
        ``(first_date, last_date)`` of the panel the model will be scored
        over. When given, an input whose own span does not cover it is
        reported ``insufficient_span`` — the check the row count alone
        cannot make (a series can be long AND start too late).

    Returns
    -------
    dict
        ``status`` is ``"failed"`` if any required input failed, else
        ``"degraded"`` if any optional one did, else ``"ok"``. The caller
        decides what to do with it; ``assert_trainable`` is the fail-loud
        half.
    """
    register = register if register is not None else INPUT_REGISTER
    per_symbol = per_symbol or {}

    ref = per_symbol.get(REFERENCE_SYMBOL) or {}
    ref_rows = int(ref.get("rows") or 0)
    ref_last = _as_date(ref.get("last_date"))

    inputs: dict[str, dict] = {}
    findings: list[_Finding] = []

    for spec in register:
        obs = per_symbol.get(spec.symbol)
        rows = int((obs or {}).get("rows") or 0)
        first_d = _as_date((obs or {}).get("first_date"))
        last_d = _as_date((obs or {}).get("last_date"))
        rows_ratio = (rows / ref_rows) if ref_rows else None
        stale_days = (
            (ref_last - last_d).days if (ref_last and last_d) else None
        )

        # Every applicable condition is evaluated, then a DECLARED
        # precedence picks the headline status. Ordering matters and is not
        # arbitrary: a producer that FROZE is more actionable and more
        # deceptive than one that is merely short, because the short one is
        # often a known upstream availability limit (HYOAS is FRED
        # license-gated to 2023+) while the frozen one is a live regression.
        # The full list survives on ``issues`` so nothing is hidden by the
        # headline.
        issues: list[tuple[str, str]] = []
        if obs is None or rows == 0:
            issues.append(("absent", (
                f"{spec.symbol} is absent from the loaded cache (rows=0). "
                f"Features starved: {', '.join(spec.features) or 'none declared'}."
            )))
        else:
            if last_d is None or first_d is None:
                issues.append(("unknown_dates", (
                    f"{spec.symbol} holds {rows} rows but its index dates "
                    "could not be read, so span and staleness are "
                    "unverifiable."
                )))
            if stale_days is not None and stale_days > spec.max_staleness_days:
                issues.append(("stale", (
                    f"{spec.symbol} last writes {last_d.isoformat()} against "
                    f"{REFERENCE_SYMBOL}'s {ref_last.isoformat()} — "
                    f"{stale_days} days behind (max "
                    f"{spec.max_staleness_days}). A frozen producer ffills "
                    "its last value forward; the feature reads as smooth, "
                    "not missing."
                )))
            if oos_window and first_d and last_d:
                win_start = _as_date(oos_window[0])
                win_end = _as_date(oos_window[1])
                if win_start and win_end and (first_d > win_start or last_d < win_end):
                    issues.append(("insufficient_span", (
                        f"{spec.symbol} spans {first_d.isoformat()}.."
                        f"{last_d.isoformat()} but the scored window is "
                        f"{win_start.isoformat()}..{win_end.isoformat()} — "
                        "the series does not cover the dates it is scored "
                        "over."
                    )))
            if rows_ratio is not None and rows_ratio < spec.min_rows_ratio:
                issues.append(("short_history", (
                    f"{spec.symbol} holds {rows} rows against "
                    f"{REFERENCE_SYMBOL}'s {ref_rows} (ratio "
                    f"{rows_ratio:.4f} < floor {spec.min_rows_ratio}). A "
                    "symbol-presence check reads this as full coverage; it "
                    "is not."
                )))

        by_status = dict(issues)
        status, reason = "complete", ""
        for candidate in _STATUS_PRECEDENCE:
            if candidate in by_status:
                status, reason = candidate, by_status[candidate]
                break

        inputs[spec.symbol] = {
            "status": status,
            "severity": spec.severity,
            "rows": rows,
            "rows_ratio": round(rows_ratio, 6) if rows_ratio is not None else None,
            "first_date": (obs or {}).get("first_date"),
            "last_date": (obs or {}).get("last_date"),
            "stale_days": stale_days,
            "consumers": list(spec.consumers),
            "features": list(spec.features),
            "reason": reason or None,
            # Every condition that tripped, not only the headline one.
            "issues": [{"status": st, "reason": rs} for st, rs in issues],
        }
        if status != "complete":
            findings.append(
                _Finding(spec.symbol, status, spec.severity, reason, spec.features)
            )

    failures = [f for f in findings if f.severity == "required"]
    degradations = [f for f in findings if f.severity != "required"]
    status = "failed" if failures else ("degraded" if degradations else "ok")

    return {
        "status": status,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "symbol": REFERENCE_SYMBOL,
            "rows": ref_rows,
            "first_date": ref.get("first_date"),
            "last_date": ref.get("last_date"),
        },
        "oos_window": list(oos_window) if oos_window else None,
        "inputs": inputs,
        "failures": [f.as_dict() for f in failures],
        "degradations": [f.as_dict() for f in degradations],
        # The honest headline number, replacing the symbol-count ratio as
        # the thing a reader should look at. `1.0` here means every
        # registered input cleared its ROW and DATE bar, which is the claim
        # `coverage_ratio` was mistakenly read as making.
        "input_completeness_ratio": (
            round((len(register) - len(findings)) / len(register), 6)
            if register else 1.0
        ),
    }


def assert_trainable(block: dict) -> None:
    """Raise ``DataCompletenessError`` when a REQUIRED input failed.

    Fail-loud half of the gate. Degradations do not raise — they are
    recorded by the caller onto the manifest, where the model zoo can see
    that two arms did not train on the same inputs.
    """
    failures = (block or {}).get("failures") or []
    if not failures:
        for d in (block or {}).get("degradations") or []:
            log.warning(
                "data_completeness: DEGRADED input %s (%s) — %s",
                d.get("input"), d.get("status"), d.get("reason"),
            )
        return
    detail = "; ".join(f"{f['input']} [{f['status']}] {f['reason']}" for f in failures)
    raise DataCompletenessError(
        "Refusing to train: "
        f"{len(failures)} REQUIRED input(s) failed the row/date completeness "
        f"gate — {detail} "
        "(alpha-engine-config-I9290: a symbol-count coverage_ratio of 1.0 does "
        "not mean the rows are there; this gate measures rows per symbol and "
        "the date span, and never substitutes a constant for a missing input.)"
    )


def summarize_universe_rows(per_symbol: dict, *, registered: "set[str] | None" = None,
                            worst_n: int = 20) -> dict:
    """Row-distribution summary over the non-macro universe symbols.

    Keeps the manifest small: the full ~900-symbol map is not persisted,
    only the distribution and the worst offenders — enough to see a
    starved universe without carrying a 70 KB dict into every artifact.
    """
    registered = registered or {s.symbol for s in INPUT_REGISTER}
    rows = {
        sym: int((v or {}).get("rows") or 0)
        for sym, v in (per_symbol or {}).items()
        if sym not in registered
    }
    if not rows:
        return {"n_symbols": 0, "min_rows": None, "median_rows": None,
                "p05_rows": None, "worst": []}
    ordered = sorted(rows.values())
    n = len(ordered)
    return {
        "n_symbols": n,
        "min_rows": ordered[0],
        "median_rows": ordered[n // 2],
        "p05_rows": ordered[max(0, int(0.05 * n) - 1)],
        "worst": [
            {"symbol": s, "rows": r,
             "last_date": (per_symbol.get(s) or {}).get("last_date")}
            for s, r in sorted(rows.items(), key=lambda kv: kv[1])[:worst_n]
        ],
    }
