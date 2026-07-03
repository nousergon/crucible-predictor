"""
analysis/low_n_selection_estimator.py — low-n selection-bias promotion estimator.

config#1524 (§0.6b) PROTOTYPE of the family decided in config#1523: an
**anytime-valid e-process** (empirical-Bernstein testing-by-betting wealth) per
challenger, combined across the challenger family by **e-BH** (Wang & Ramdas
2022) for FDR control under *arbitrary* dependence, with **empirical-Bayes
shrinkage used ONLY as a bet-sizing prior** — never as the promotion gate.

WHY NOT the standard LdP battery here (config#1522 thesis)
---------------------------------------------------------
Deflated-Sharpe / PBO (``training/deflated_sharpe.py``,
``nousergon_lib.quant.stats.pbo``) assume a long backtest return stream; at our
regime — a handful of realized forward cohorts, many concurrent LLM-agent
challengers, promoting on live outcomes — deflation zeros out at ≈ 1 independent
block (documented in ``training/model_zoo.py:489-559``) and the fresh-best-wins
fallback (relative-best, margin=0) overfits the garden of forking paths at
n_eff single digits. The right machinery is fixed-n-free and native to optional
stopping: testing-by-betting e-values.

WHAT AN E-VALUE BUYS US
-----------------------
For challenger ``j`` we test H0_j: "challenger has NO edge over the champion",
i.e. ``E[D_t] <= 0`` where ``D_t`` is the paired per-cohort outcome difference
(challenger − champion realized alpha/PnL for the same cohort). The wealth
process ``W_t = prod_{s<=t} (1 + lambda_s * D_s)`` with a **predictable** bet
``lambda_s in [0, lambda_max]`` is a non-negative supermartingale under H0
(``E[1 + lambda_s D_s | F_{s-1}] = 1 + lambda_s E[D_s|.] <= 1``), so by Ville's
inequality ``P(sup_t W_t >= 1/alpha) <= alpha`` — a valid test at ANY stopping
time and at ANY cohort depth, with no fixed-n assumption. ``W_t`` (and its
running max) is an **e-value**.

FDR UNDER DEPENDENCE (e-BH)
---------------------------
Challengers share market regime / overlapping windows, so their tests are
arbitrarily dependent. p-value BH (``nousergon_lib.quant.stats.multiple_testing
.benjamini_hochberg``) is not valid under arbitrary dependence, but its e-value
analog — **e-BH** — is: sort e-values descending, find the largest ``k`` with
``e_(k) >= m / (alpha * k)``, reject those ``k``. FDR is controlled at ``alpha``
under *any* dependence structure. This is the multiplicity layer the family scan
(#1523) selected precisely for the correlated-challenger regime.

EMPIRICAL-BAYES = BET SIZING ONLY
---------------------------------
The bet ``lambda_s`` is sized from a running empirical-Bernstein estimate of the
edge/variance seen so far, **shrunk toward a conservative prior mean** (default
0.0 = the null). Shrinkage stabilises the bet at low-n and damps noise-chasing;
critically it only changes the *size* of a predictable, non-negative bet, so the
e-value guarantee is preserved (any ``lambda_s in [0, lambda_max]`` is valid).
EB never decides promotion — the e-BH rejection set does.

STATUS: PROTOTYPE for the #1524 kill-gate. Pure ``numpy``; reuses no promotion
logic from ``model_zoo`` (that wiring is #1525, gated on this passing). If the
kill-gate PASSES on real accrued cohorts (~2026-07-20), #1525 lifts the
validated core to ``nousergon-lib`` on second-consumer adoption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class ChallengerCohorts:
    """Paired per-cohort outcome differences for one challenger vs the champion.

    ``diffs[t]`` = (challenger realized outcome − champion realized outcome) on
    cohort ``t`` (e.g. sector-neutral realized 21d alpha, or PnL). Same cohort
    ordering across challengers so the shared-regime dependence is honest.
    """

    challenger_id: str
    diffs: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "diffs", np.asarray(self.diffs, dtype=float))

    @property
    def n_cohorts(self) -> int:
        return int(self.diffs.size)


@dataclass(frozen=True)
class EProcessResult:
    challenger_id: str
    e_value: float  # terminal wealth W_T — a valid e-value by optional stopping
    running_max_e: float  # sup_t W_t — also a valid e-value (anytime)
    n_cohorts: int
    wealth_path: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class PromotionVerdict:
    """Which challenger (if any) promotes, plus the audit trail (fail-loud)."""

    method: str
    alpha: float
    promoted_id: str | None
    rejected_ids: tuple[str, ...]
    e_values: dict[str, float]
    blocked: bool  # True ⇒ no defensible promotion; the fail-loud outcome


def e_process(
    diffs: np.ndarray,
    *,
    bound: float,
    lambda_max: float | None = None,
    eb_prior_mean: float = 0.0,
    eb_shrink: float = 0.3,
) -> EProcessResult:
    """Empirical-Bernstein testing-by-betting e-process for H0: ``E[D] <= 0``.

    Args:
        diffs: per-cohort paired outcome differences ``D_1..D_T``.
        bound: symmetric winsorisation bound ``b``; ``D_t`` is clipped to
            ``[-b, b]`` so wealth stays non-negative for ``lambda in [0, 1/b]``.
        lambda_max: max predictable bet. Defaults to ``0.5 / bound`` (half-Kelly
            safety margin) which keeps ``1 + lambda*D >= 0.5 > 0``.
        eb_prior_mean: empirical-Bayes shrinkage target for the edge estimate
            used to *size* the bet (default 0.0 = shrink toward the null). Bet
            sizing only — never a gate.
        eb_shrink: shrinkage weight ``w in [0,1]`` toward ``eb_prior_mean``
            (default 0.3 — light shrink; enough to damp low-n noise-chasing
            without smothering a genuine edge's wealth growth).

    Returns:
        ``EProcessResult`` with terminal + running-max wealth (both e-values).

    The bet at step ``t`` is predictable (uses cohorts ``< t`` only):
    ``lambda_t = clip( m_eb / (v + eps), 0, lambda_max )`` where ``m_eb`` is the
    EB-shrunk running mean of ``D`` and ``v`` the running (empirical-Bernstein)
    variance. Predictability + non-negativity ⇒ ``W_t`` is a supermartingale
    under H0 ⇒ a valid e-value at any stopping time.
    """
    d = np.clip(np.asarray(diffs, dtype=float), -bound, bound)
    n = d.size
    if lambda_max is None:
        # 0.9/bound keeps 1 + lambda*D >= 0.1 > 0 (wealth stays positive) while
        # betting hard enough to accumulate power as cohorts accrue.
        lambda_max = 0.9 / bound
    lambda_max = float(min(lambda_max, 1.0 / bound))

    wealth = 1.0
    path = np.empty(n, dtype=float)
    run_max = 1.0
    # Running (predictable) mean/var over cohorts strictly before t.
    seen = np.empty(0, dtype=float)
    for t in range(n):
        if seen.size == 0:
            m_hat, v_hat = 0.0, 1.0  # no info yet ⇒ neutral, no bet
        else:
            m_hat = float(seen.mean())
            # empirical-Bernstein-style variance floor keeps early bets tame
            v_hat = float(seen.var(ddof=0)) + 0.25
        m_eb = (1.0 - eb_shrink) * m_hat + eb_shrink * eb_prior_mean
        lam = m_eb / (v_hat + _EPS)
        lam = float(np.clip(lam, 0.0, lambda_max))
        wealth *= 1.0 + lam * d[t]
        wealth = max(wealth, 0.0)
        path[t] = wealth
        run_max = max(run_max, wealth)
        seen = np.append(seen, d[t])

    return EProcessResult(
        challenger_id="",
        e_value=float(wealth),
        running_max_e=float(run_max),
        n_cohorts=int(n),
        wealth_path=path,
    )


def e_bh(e_values: list[float], alpha: float = 0.10) -> list[bool]:
    """e-BH (Wang & Ramdas 2022): FDR ``<= alpha`` under ARBITRARY dependence.

    Reject the ``k*`` largest e-values where ``k* = max{ k : e_(k) >= m/(alpha*k) }``
    (``e_(k)`` the k-th largest e-value, ``m`` the number of hypotheses).

    Unlike ``multiple_testing.benjamini_hochberg`` (p-values; only valid under
    independence / PRDS), e-BH needs no dependence assumption — the right tool
    for the correlated-challenger family (config#1523 decision).
    """
    m = len(e_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: e_values[i], reverse=True)
    k_star = 0
    for rank, idx in enumerate(order, start=1):
        if e_values[idx] >= m / (alpha * rank):
            k_star = rank
    reject = [False] * m
    for rank in range(k_star):
        reject[order[rank]] = True
    return reject


def estimator_promote(
    cohorts: list[ChallengerCohorts],
    *,
    alpha: float = 0.10,
    bound: float = 3.0,
    lambda_max: float | None = None,
    use_family_prior: bool = True,
) -> PromotionVerdict:
    """The low-n estimator's promotion verdict (fail-loud).

    1. Optionally set the EB bet-sizing prior to a *conservative* family signal
       (min(0, grand mean) — never a positive prior that would inflate bets).
    2. Compute each challenger's e-process e-value.
    3. e-BH across the family for FDR-controlled rejection under dependence.
    4. Promote the highest-e-value challenger among the e-BH rejections; if none
       are rejected, BLOCK (no statistically-defended promotion) — never the
       fresh-best-wins silent promote.
    """
    if not cohorts:
        return PromotionVerdict("estimator", alpha, None, (), {}, blocked=True)

    prior_mean = 0.0
    if use_family_prior:
        all_means = [float(c.diffs.mean()) for c in cohorts if c.n_cohorts]
        if all_means:
            # Shrink toward the family grand mean, but clamp at 0 so the prior is
            # never optimistic — keeps the e-value guarantee conservative.
            prior_mean = min(0.0, float(np.mean(all_means)))

    e_values: dict[str, float] = {}
    for c in cohorts:
        if c.n_cohorts == 0:
            e_values[c.challenger_id] = 1.0
            continue
        res = e_process(
            c.diffs,
            bound=bound,
            lambda_max=lambda_max,
            eb_prior_mean=prior_mean,
        )
        # Running max is the anytime-valid e-value (honours optional stopping).
        e_values[c.challenger_id] = res.running_max_e

    ids = [c.challenger_id for c in cohorts]
    reject = e_bh([e_values[i] for i in ids], alpha=alpha)
    rejected = tuple(ids[i] for i in range(len(ids)) if reject[i])

    if not rejected:
        return PromotionVerdict(
            "estimator", alpha, None, (), e_values, blocked=True
        )
    promoted = max(rejected, key=lambda i: e_values[i])
    return PromotionVerdict(
        "estimator", alpha, promoted, rejected, e_values, blocked=False
    )


def fresh_best_wins_promote(
    cohorts: list[ChallengerCohorts],
) -> PromotionVerdict:
    """The CURRENT punt (config#1522): relative-best, margin=0.

    Promote ``argmax_j mean(D_j)`` — no significance gate, never blocks. This is
    the baseline the #1524 kill-gate must beat; modelled here so the comparison
    is apples-to-apples on identical cohorts.
    """
    scored = [(c.challenger_id, float(c.diffs.mean()) if c.n_cohorts else -np.inf)
              for c in cohorts]
    scored = [s for s in scored if np.isfinite(s[1])]
    if not scored:
        return PromotionVerdict("fresh_best_wins", 0.0, None, (), {}, blocked=True)
    promoted = max(scored, key=lambda s: s[1])[0]
    return PromotionVerdict(
        "fresh_best_wins",
        0.0,
        promoted,
        (promoted,),
        {cid: sc for cid, sc in scored},
        blocked=False,
    )


def cohorts_from_realized_records(
    records: dict[str, list[dict]],
    *,
    challenger_key: str = "realized_alpha",
    champion_key: str = "champion_realized_alpha",
) -> list[ChallengerCohorts]:
    """Adapter: build ``ChallengerCohorts`` from realized-outcome records.

    ``records[challenger_id]`` = per-cohort dicts each carrying the challenger's
    and champion's realized outcome for that cohort (as produced by
    ``analysis/observe_leaderboard.build_observe_leaderboard`` / the scanner &
    producer OBSERVE shadows). Pure + testable; the S3/ArcticDB read that
    populates ``records`` is wired by the kill-gate script so the real-cohort run
    (~2026-07-20) is a data-swap, not a code change.
    """
    out: list[ChallengerCohorts] = []
    for cid, rows in records.items():
        diffs = np.array(
            [float(r[challenger_key]) - float(r[champion_key]) for r in rows],
            dtype=float,
        )
        out.append(ChallengerCohorts(challenger_id=cid, diffs=diffs))
    return out
