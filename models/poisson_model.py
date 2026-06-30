"""
Dixon-Coles bivariate Poisson model for score prediction.
Converts Elo ratings → expected goals → score distribution.
"""
import numpy as np
from scipy.stats import poisson as scipy_poisson
from typing import Tuple
import math
from config import AVG_GOALS_PER_TEAM, DC_RHO, ODDS_BLEND_WEIGHT

MAX_GOALS = 9   # max goals in distribution (>9 is negligible)


def elo_to_lambdas(elo_home: float, elo_away: float,
                   home_adv_elo: float = 0.0) -> Tuple[float, float]:
    """
    Convert Elo ratings to Poisson lambdas (expected goals per team).
    Calibrated: 200-pt Elo gap ≈ 63% win probability; avg 1.25 goals/team.
    """
    dr = (elo_home - elo_away + home_adv_elo) / 400
    ratio = 10 ** (dr * 0.55)   # softer than win-prob, empirically calibrated
    lam = AVG_GOALS_PER_TEAM * math.sqrt(ratio)
    mu = AVG_GOALS_PER_TEAM / math.sqrt(ratio)
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
    """Return the most probable exact scoreline."""
    M = score_matrix(lam, mu)
    idx = int(np.argmax(M))
    return divmod(idx, MAX_GOALS + 1)


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
