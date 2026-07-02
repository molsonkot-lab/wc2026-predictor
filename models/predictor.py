"""
Match prediction engine with natural-language reasoning.
Combines Elo + Dixon-Coles + odds + player factors.
"""
from typing import Dict, Tuple, Optional
from models.elo import EloSystem
from models.poisson_model import (
    elo_to_lambdas, match_probabilities, blend_with_odds,
    fit_lambdas_to_probs, representative_score,
)
from data.players import get_key_players, compute_player_adjusted_elo
import numpy as np


def predict_match(
    home_id: int,
    away_id: int,
    home_name: str,
    away_name: str,
    home_tla: str,
    away_tla: str,
    elo: EloSystem,
    player_statuses: Dict[str, str],   # {player_name: "FIT"|"DOUBTFUL"|"OUT"}
    odds_probs: Optional[Tuple[float, float, float]] = None,  # from bookmaker
    home_adv_elo: float = 0.0,
    odds_weight: Optional[float] = None,   # market blend weight (None → config default)
    goal_env: float = 1.0,                 # heat/altitude scoring multiplier (1.0 = neutral)
    avg_goals: Optional[float] = None,     # tuned per-team goal baseline (None → config default)
) -> Dict:
    """
    Returns a prediction dict with probabilities, expected score,
    most-likely score, and a human-readable reasoning string.
    """
    from config import ODDS_BLEND_WEIGHT
    w = ODDS_BLEND_WEIGHT if odds_weight is None else odds_weight

    base_elo_h = elo.get_rating(home_id)
    base_elo_a = elo.get_rating(away_id)

    adj_elo_h = compute_player_adjusted_elo(base_elo_h, home_tla, player_statuses)
    adj_elo_a = compute_player_adjusted_elo(base_elo_a, away_tla, player_statuses)

    lam, mu = elo_to_lambdas(adj_elo_h, adj_elo_a, home_adv_elo,
                             goal_env=goal_env, base_goals=avg_goals)
    p_h, p_d, p_a = match_probabilities(lam, mu)

    # Blend with market odds if available, then refit lambdas to the blended
    # probabilities so the scoreline reflects the market signal too (keeping
    # the environment-adjusted total-goals level).
    if odds_probs and w > 0:
        p_h, p_d, p_a = blend_with_odds((p_h, p_d, p_a), odds_probs, weight=w)
        lam, mu = fit_lambdas_to_probs(p_h, p_d, p_a, total=lam + mu)

    (ml_home, ml_away), _ = representative_score(lam, mu, (p_h, p_d, p_a))

    reasoning = _build_reasoning(
        home_name=home_name, away_name=away_name,
        home_tla=home_tla, away_tla=away_tla,
        base_elo_h=base_elo_h, base_elo_a=base_elo_a,
        adj_elo_h=adj_elo_h, adj_elo_a=adj_elo_a,
        lam=lam, mu=mu,
        p_h=p_h, p_d=p_d, p_a=p_a,
        odds_probs=odds_probs,
        player_statuses=player_statuses,
    )

    return {
        "p_home": p_h,
        "p_draw": p_d,
        "p_away": p_a,
        "xg_home": lam,
        "xg_away": mu,
        "predicted_home": ml_home,
        "predicted_away": ml_away,
        "adj_elo_home": adj_elo_h,
        "adj_elo_away": adj_elo_a,
        "reasoning": reasoning,
    }


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def _build_reasoning(
    home_name, away_name, home_tla, away_tla,
    base_elo_h, base_elo_a, adj_elo_h, adj_elo_a,
    lam, mu, p_h, p_d, p_a, odds_probs, player_statuses
) -> str:
    lines = []

    # ── Probabilities ──────────────────────────────────────────
    lines.append("**预测概率**")
    lines.append(f"- {home_name} 胜：**{p_h*100:.1f}%** {_bar(p_h)}")
    lines.append(f"- 平局：{p_d*100:.1f}% {_bar(p_d)}")
    lines.append(f"- {away_name} 胜：{p_a*100:.1f}% {_bar(p_a)}")
    lines.append(f"- 预期比分：**{lam:.2f} – {mu:.2f}**（xG）")
    lines.append("")

    # ── Elo analysis ───────────────────────────────────────────
    elo_diff = adj_elo_h - adj_elo_a
    lines.append("**队伍实力（Elo）**")
    lines.append(f"- {home_name} 基础 Elo：{base_elo_h:.0f} → 球员调整后：{adj_elo_h:.0f}")
    lines.append(f"- {away_name} 基础 Elo：{base_elo_a:.0f} → 球员调整后：{adj_elo_a:.0f}")
    if abs(elo_diff) < 30:
        lines.append("- 📊 两队实力**非常接近**，任何结果皆有可能")
    elif elo_diff > 0:
        adv = "主队"
        lines.append(f"- 📊 {home_name} 拥有 **+{elo_diff:.0f} Elo 优势**，约对应 {p_h*100:.0f}% 胜率")
    else:
        lines.append(f"- 📊 {away_name} 拥有 **+{abs(elo_diff):.0f} Elo 优势**，约对应 {p_a*100:.0f}% 胜率")
    lines.append("")

    # ── Market odds ────────────────────────────────────────────
    if odds_probs:
        lines.append("**赔率市场信号**")
        o_h, o_d, o_a = odds_probs
        lines.append(f"- 博彩市场：{home_name} {o_h*100:.1f}% | 平 {o_d*100:.1f}% | {away_name} {o_a*100:.1f}%")
        model_vs_market = p_h - o_h
        if abs(model_vs_market) > 0.05:
            direction = "高于" if model_vs_market > 0 else "低于"
            lines.append(f"- ⚖️ 模型对 {home_name} 的评估**{direction}**市场 {abs(model_vs_market)*100:.1f}%")
        else:
            lines.append("- ✅ 模型与市场基本一致，预测可信度较高")
        lines.append("")

    # ── Key player status ──────────────────────────────────────
    home_players = get_key_players(home_tla)
    away_players = get_key_players(away_tla)

    def _player_lines(players, tla, team_name):
        result = []
        for p in players:
            status = player_statuses.get(p["name"], p["status"])
            icon = {"FIT": "✅", "DOUBTFUL": "⚠️", "OUT": "❌"}.get(status, "❓")
            note = ""
            if status == "OUT":
                note = f"（-{p['impact']} Elo）"
            elif status == "DOUBTFUL":
                note = f"（-{p['impact']*0.5:.0f} Elo 折算）"
            result.append(f"  - {icon} **{p['name']}** [{p['position']}] {note}— {p['notes']}")
        return result

    if home_players or away_players:
        lines.append("**核心球员状态**")
        if home_players:
            lines.append(f"*{home_name}:*")
            lines.extend(_player_lines(home_players, home_tla, home_name))
        if away_players:
            lines.append(f"*{away_name}:*")
            lines.extend(_player_lines(away_players, away_tla, away_name))

        # Impact summary
        h_adj = base_elo_h - adj_elo_h
        a_adj = base_elo_a - adj_elo_a
        if h_adj > 20:
            lines.append(f"- ⚠️ 伤病/停赛使 {home_name} 实力下降约 **{h_adj:.0f} Elo 点**")
        if a_adj > 20:
            lines.append(f"- ⚠️ 伤病/停赛使 {away_name} 实力下降约 **{a_adj:.0f} Elo 点**")
        lines.append("")

    # ── Prediction summary ─────────────────────────────────────
    lines.append("**综合结论**")
    if p_h > 0.55:
        verdict = f"{home_name} **获胜可能性较大**（{p_h*100:.0f}%）"
    elif p_a > 0.55:
        verdict = f"{away_name} **获胜可能性较大**（{p_a*100:.0f}%）"
    elif p_d > 0.30:
        verdict = f"**两队实力接近，平局概率较高**（{p_d*100:.0f}%）"
    else:
        verdict = f"**结果难以预测**，{home_name} 略微占优"
    lines.append(f"- {verdict}")

    return "\n".join(lines)
