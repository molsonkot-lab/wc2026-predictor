"""
Market calibration.

The single strongest public signal for who wins a tournament is the bookmaker
outright (winner) market. Pure seed-Elo + Poisson ignores it entirely, which is
the main accuracy gap. This module nudges each team's Elo rating so that the
*simulated* champion probabilities converge toward the market's outright
probabilities — anchoring to the market while keeping all match-level structure
(Dixon-Coles scores, H2H, host advantage, knockout paths) intact and internally
consistent.

`market_weight` is the anchoring strength:
    0.0 → ignore market, keep pure model
    1.0 → fully match the market's champion distribution
    ~0.7 (default) → mostly market, model fills the gaps (long-shots, paths)

Only teams present in the market are adjusted. Teams without a quoted price keep
their model Elo; their probabilities shift naturally as the favourites are
re-rated, which is the desired behaviour.
"""
import math
from typing import Dict, Tuple


def _resolve_market_to_ids(out_map: Dict[str, float],
                           team_map: Dict[int, dict]) -> Dict[int, float]:
    """Map {team_name_lower: prob} → {team_id: prob}, tolerant of name drift."""
    name_to_id = {}
    for tid, t in team_map.items():
        nm = (t.get("name") or "").lower().strip()
        if nm:
            name_to_id[nm] = tid

    resolved: Dict[int, float] = {}
    for name_lower, p in out_map.items():
        if p <= 0:
            continue
        tid = name_to_id.get(name_lower)
        if tid is None:
            # fall back to substring match (e.g. "korea republic" vs "south korea")
            for nm, i in name_to_id.items():
                if name_lower in nm or nm in name_lower:
                    tid = i
                    break
        if tid is not None:
            resolved[tid] = resolved.get(tid, 0.0) + p
    return resolved


def calibrate_elo_to_market(
    sim,
    out_map: Dict[str, float],
    team_map: Dict[int, dict],
    *,
    market_weight: float = 0.7,
    iterations: int = 10,
    n_sims: int = 4000,
    learning_rate: float = 0.6,
    max_step: float = 150.0,
    fixed_results: Dict[int, Tuple[int, int]] = None,
    base_overrides: Dict[int, float] = None,
) -> Tuple[Dict[int, float], dict]:
    """
    Returns (elo_overrides, info).

    elo_overrides : {team_id: calibrated_elo} ready to pass to run_simulations.
    info          : diagnostics (resolved team count, per-iteration max error).

    `base_overrides` (e.g. player-injury adjusted Elo) is used as the starting
    point so calibration stacks on top of, rather than discards, those effects.
    """
    market = _resolve_market_to_ids(out_map, team_map)
    info = {"resolved": len(market), "errors": []}
    if not market:
        info["status"] = "no_market"
        return dict(base_overrides or {}), info

    # Renormalise market over the teams we could resolve (it may not sum to 1).
    msum = sum(market.values())
    market = {tid: p / msum for tid, p in market.items()}

    # Start from current (optionally injury-adjusted) ratings for every team.
    overrides: Dict[int, float] = {}
    for tid in sim.teams:
        if base_overrides and tid in base_overrides:
            overrides[tid] = base_overrides[tid]
        else:
            overrides[tid] = sim.elo.get_rating(tid)

    for it in range(iterations):
        probs = sim.run_simulations(n_sims, fixed_results=fixed_results,
                                    elo_overrides=overrides)
        # Decay the step size over iterations to settle instead of oscillating.
        lr = learning_rate / (1 + 0.2 * it)
        max_err = 0.0
        for tid, p_mkt in market.items():
            p_mod = max(probs.get(tid, {}).get("win", 0.0), 1e-5)
            # Blend model & market in log-space → the target this iteration aims at.
            log_target = ((1 - market_weight) * math.log(p_mod)
                          + market_weight * math.log(p_mkt))
            # Elo logit slope is ln(10)/400 per point; invert to get the step that
            # would move log-prob by (log_target - log p_mod), damped by lr.
            delta = lr * (log_target - math.log(p_mod)) * (400 / math.log(10))
            delta = max(-max_step, min(max_step, delta))
            overrides[tid] += delta
            max_err = max(max_err, abs(p_mkt - p_mod))
        info["errors"].append(round(max_err, 4))

    info["status"] = "ok"
    return overrides, info
