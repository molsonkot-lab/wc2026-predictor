"""
Dixon-Coles bivariate Poisson model for score prediction.
Converts Elo ratings → expected goals → score distribution.
"""
import numpy as np
from scipy.stats import poisson as scipy_poisson
from typing import Tuple
import math
from config import (
    AVG_GOALS_PER_TEAM, DC_RHO, ODDS_BLEND_WEIGHT, ELO_GOAL_EXPONENT,
)

MAX_GOALS = 9   # max goals in distribution (>9 is negligible)


def elo_to_lambdas(elo_home: float, elo_away: float,
                   home_adv_elo: float = 0.0,
                   goal_env: float = 1.0,
                   base_goals: float = None) -> Tuple[float, float]:
    """
    Convert Elo ratings to Poisson lambdas (expected goals per team).
    Calibrated: 200-pt Elo gap ≈ 63% win probability.

    goal_env scales the overall scoring baseline for match conditions
    (heat / altitude); 1.0 = neutral. It multiplies both teams equally,
    so it shifts the total goals without biasing either side.
    base_goals overrides AVG_GOALS_PER_TEAM (fed by the nightly auto-tune,
    which re-estimates the real per-team goal average from finished matches).
    """
    dr = (elo_home - elo_away + home_adv_elo) / 400
    ratio = 10 ** (dr * ELO_GOAL_EXPONENT)
    base = (AVG_GOALS_PER_TEAM if base_goals is None else base_goals) * goal_env
    lam = base * math.sqrt(ratio)
    mu = base / math.sqrt(ratio)
    # Clamp to reasonable range
    lam = max(0.3, min(4.5, lam))
    mu = max(0.3, min(4.5, mu))
    return lam, mu


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles correction for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float = DC_RHO) -> np.ndarray:
    """
    P[home_goals, away_goals] using Dixon-Coles bivariate Poisson.
    Shape: (MAX_GOALS+1, MAX_GOALS+1)
    """
    n = MAX_GOALS + 1
    M = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            tau = _dc_tau(x, y, lam, mu, rho)
            M[x, y] = tau * scipy_poisson.pmf(x, lam) * scipy_poisson.pmf(y, mu)
    M /= M.sum()  # renormalise after tau correction
    return M


def match_probabilities(lam: float, mu: float,
                        rho: float = DC_RHO) -> Tuple[float, float, float]:
    """Return (p_home_win, p_draw, p_away_win)."""
    M = score_matrix(lam, mu, rho)
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    return p_home, p_draw, p_away


def most_likely_score(lam: float, mu: float) -> Tuple[int, int]:
    """Return the most probable exact scoreline (the matrix mode / argmax).

    NOTE: For typical football lambdas (~1.0–1.8) the Poisson mode is floor(λ),
    so this collapses to 1:0 / 1:1 / 0:0 in the vast majority of matches even
    when one side is a clear favourite. Prefer `expected_score` (continuous,
    responds to strength) or `top_scorelines` (shows the spread) for display.
    """
    M = score_matrix(lam, mu)
    idx = int(np.argmax(M))
    return divmod(idx, MAX_GOALS + 1)


def expected_score(lam: float, mu: float) -> Tuple[int, int]:
    """Rounded expected goals — a point estimate that actually tracks team
    strength (unlike the argmax mode). e.g. xG 1.6–1.0 → 2:1, not 1:0."""
    return int(round(lam)), int(round(mu))


def top_scorelines(lam: float, mu: float, rho: float = DC_RHO,
                   n: int = 3) -> list:
    """Return the n most likely exact scorelines as [((h, a), prob), ...].

    Surfacing this (instead of a single modal score) makes the probability
    spread visible — 1:0 being 'most likely' at only ~12% is the whole point.
    """
    M = score_matrix(lam, mu, rho)
    flat = [(M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
    flat.sort(reverse=True)
    return [((i, j), float(p)) for p, i, j in flat[:n]]


def fit_lambdas_to_probs(p_home: float, p_draw: float, p_away: float,
                         total: float, rho: float = DC_RHO) -> Tuple[float, float]:
    """
    Invert the Dixon-Coles model: find (lam, mu) whose W/D/L probabilities best
    match a target probability vector, keeping the total-goals level fixed.

    Why: match probabilities get blended with bookmaker odds, but the raw Elo
    lambdas used for the scoreline display don't — so the score could contradict
    the (market-informed) W/D/L numbers. Fitting lambdas back to the blended
    probs makes the predicted score absorb the market signal too.
    """
    total = max(0.6, min(9.0, total))
    best, best_err = (total / 2, total / 2), float("inf")
    for i in range(33):
        w = 0.12 + (0.88 - 0.12) * i / 32
        lam = max(0.3, min(4.5, total * w))
        mu = max(0.3, min(4.5, total - lam))
        ph, pd, pa = match_probabilities(lam, mu, rho)
        err = (ph - p_home) ** 2 + (pd - p_draw) ** 2 + (pa - p_away) ** 2
        if err < best_err:
            best_err, best = err, (lam, mu)
    return best


def representative_score(lam: float, mu: float,
                         probs: Tuple[float, float, float] = None,
                         rho: float = DC_RHO) -> Tuple[Tuple[int, int], float]:
    """
    Most likely exact scoreline CONSISTENT with the predicted outcome.

    Rounded xG collapses to 2:1 / 1:1 / 3:1 for nearly every match; the raw
    matrix argmax collapses to 1:0 / 1:1. This picks argmax within the region
    of the highest-probability outcome (home win / draw / away win), so the
    scoreline never contradicts the headline W/D/L call and still varies with
    the goal distribution. Returns ((home, away), p_of_that_scoreline).
    """
    M = score_matrix(lam, mu, rho)
    if probs is None:
        probs = match_probabilities(lam, mu, rho)
    outcome = int(np.argmax(probs))          # 0=home win, 1=draw, 2=away win
    n = M.shape[0]
    mask = np.zeros_like(M, dtype=bool)
    idx = np.arange(n)
    if outcome == 0:
        mask[np.tril_indices(n, -1)] = True   # home_goals > away_goals
    elif outcome == 1:
        mask[idx, idx] = True
    else:
        mask[np.triu_indices(n, 1)] = True
    Mm = np.where(mask, M, -1.0)
    flat = int(np.argmax(Mm))
    h, a = divmod(flat, n)
    return (h, a), float(M[h, a])


def blend_with_odds(p_model: Tuple[float, float, float],
                    p_odds: Tuple[float, float, float],
                    weight: float = ODDS_BLEND_WEIGHT) -> Tuple[float, float, float]:
    """Weighted blend of model and market probabilities."""
    blended = [(1 - weight) * pm + weight * po for pm, po in zip(p_model, p_odds)]
    total = sum(blended)
    return tuple(b / total for b in blended)


def sample_score_fast(lam: float, mu: float) -> Tuple[int, int]:
    """Fast independent Poisson sample (no dependence). Kept for compatibility."""
    return int(np.random.poisson(lam)), int(np.random.poisson(mu))


from functools import lru_cache


@lru_cache(maxsize=4096)
def _dc_flat_cdf(lam_r: float, mu_r: float, rho: float) -> Tuple[np.ndarray, int]:
    """
    Cached flattened CDF of the Dixon-Coles score matrix for sampling.
    lam_r/mu_r are rounded lambdas (grid) so the cache stays small but accurate.
    Returns (cumulative_probs_1d, n) where n = MAX_GOALS+1.
    """
    M = score_matrix(lam_r, mu_r, rho)
    return np.cumsum(M.ravel()), M.shape[0]


def sample_score_dc(lam: float, mu: float, rho: float = DC_RHO) -> Tuple[int, int]:
    """
    Sample an exact scoreline from the Dixon-Coles bivariate distribution.

    Unlike independent Poisson, this captures the empirically-observed
    dependence between home and away goals (the low-score correction of
    Dixon & Coles 1997; consistent with the copula / dependence literature,
    e.g. Petretta 2025, PARX-Copula). Cached on a 0.05-goal grid for speed so
    a full tournament Monte Carlo stays fast.
    """
    lam_r = round(min(4.5, max(0.3, lam)) * 20) / 20.0
    mu_r  = round(min(4.5, max(0.3, mu))  * 20) / 20.0
    cdf, n = _dc_flat_cdf(lam_r, mu_r, rho)
    idx = int(np.searchsorted(cdf, np.random.random() * cdf[-1]))
    return divmod(idx, n)
