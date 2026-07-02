"""
真实历史数据回测（量化式参数拟合 + 样本外验证）。

数据：github.com/martj42/international_results（1872 年至今全部国际 A 级赛，
真实比分，含历届世界杯）。本脚本：

  1. 按时间顺序重放全部历史比赛，滚动维护 Elo（和 app 同一套思想）；
  2. 训练集 = 1998–2014 五届世界杯真实比分：网格搜索 + 最大似然拟合
       ELO_GOAL_EXPONENT（Elo差→进球差的转换强度）
       DC_RHO（Dixon-Coles 低分修正）
       AVG_GOALS_PER_TEAM（现代世界杯进球基线）
  3. 验证集 = 2018 + 2022 两届世界杯（拟合时完全没见过）：
     对比各比分点估计策略的真实命中率（矩阵众数 / 四舍五入xG /
     xG锚定MAP 各档 σ），选真实数据上最优的 σ。

用法：python3 scripts/backtest.py
输出：拟合报告 + 把结果写入 tuned_params.json 的 "backtest" 字段。
2026 年比赛被显式排除——它们留给每晚 auto_tune 的前瞻式闭环，
这里保持严格的样本外。
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.poisson_model import (elo_to_lambdas, score_matrix,
                                  match_probabilities, representative_score)
import models.poisson_model as pm

CSV_PATH = ROOT / "data" / "intl_results.csv"
CSV_URL = ("https://raw.githubusercontent.com/martj42/"
           "international_results/master/results.csv")
PARAMS_PATH = ROOT / "tuned_params.json"

TRAIN_YEARS = {1998, 2002, 2006, 2010, 2014}   # 拟合
VALID_YEARS = {2018, 2022}                     # 样本外验证
ELO_HOME_ADV = 65
MAX_G = 9


def load_matches():
    if not CSV_PATH.exists():
        import urllib.request
        print(f"downloading {CSV_URL} ...")
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            rows.append({
                "date": r["date"],
                "home": r["home_team"], "away": r["away_team"],
                "hg": min(int(r["home_score"]), MAX_G),
                "ag": min(int(r["away_score"]), MAX_G),
                "tour": r["tournament"],
                "neutral": r["neutral"].strip().upper() == "TRUE",
            })
    rows.sort(key=lambda x: x["date"])
    return rows


def _k_factor(tour: str) -> float:
    if tour == "FIFA World Cup":
        return 50
    if "qualification" in tour.lower() or "Copa" in tour or "Euro" in tour \
            or "Cup" in tour or "Championship" in tour:
        return 35
    return 20   # friendlies etc.


def replay_elo(rows):
    """滚动重放全部历史，记下每场世界杯赛前 Elo（真实的赛前信息，无未来函数）。"""
    elo = defaultdict(lambda: 1500.0)
    wc = []
    for m in rows:
        year = int(m["date"][:4])
        if m["tour"] == "FIFA World Cup" and year >= 1998 and year != 2026:
            wc.append({**m, "year": year,
                       "elo_h": elo[m["home"]], "elo_a": elo[m["away"]]})
        # Elo 更新（eloratings.net 风格：净胜球放大 K）
        dr = elo[m["home"]] - elo[m["away"]] + (0 if m["neutral"] else ELO_HOME_ADV)
        exp_h = 1 / (1 + 10 ** (-dr / 400))
        res_h = 1.0 if m["hg"] > m["ag"] else (0.5 if m["hg"] == m["ag"] else 0.0)
        diff = abs(m["hg"] - m["ag"])
        g = 1.0 if diff < 2 else (1.5 if diff == 2 else (11 + diff) / 8)
        delta = _k_factor(m["tour"]) * g * (res_h - exp_h)
        elo[m["home"]] += delta
        elo[m["away"]] -= delta
    return wc


def _lambdas(m, exponent, base_goals):
    """按赛前 Elo 算 xG，主场优势只给非中立场（世界杯多数中立）。"""
    old = pm.__dict__["ELO_GOAL_EXPONENT"]
    pm.ELO_GOAL_EXPONENT = exponent
    try:
        adv = 0 if m["neutral"] else ELO_HOME_ADV
        return elo_to_lambdas(m["elo_h"], m["elo_a"], adv, base_goals=base_goals)
    finally:
        pm.ELO_GOAL_EXPONENT = old


def fit(train):
    """网格 + 最大似然：真实世界杯比分的负对数似然最小化。"""
    base_goals = sum(m["hg"] + m["ag"] for m in train) / (2 * len(train))
    best = None
    for exponent in np.arange(0.30, 1.01, 0.05):
        for rho in np.arange(0.00, 0.17, 0.02):
            nll = 0.0
            for m in train:
                lam, mu = _lambdas(m, exponent, base_goals)
                M = score_matrix(lam, mu, rho)
                nll += -math.log(max(M[m["hg"], m["ag"]], 1e-12))
            nll /= len(train)
            if best is None or nll < best[0]:
                best = (nll, round(float(exponent), 2), round(float(rho), 2))
    return {"avg_goals_per_team": round(base_goals, 3),
            "elo_goal_exponent": best[1], "dc_rho": best[2],
            "train_nll": round(best[0], 4), "n_train": len(train)}


def evaluate(valid, fitted):
    """样本外：各比分点估计策略在 2018/2022 真实比分上的命中率。"""
    exps, rho, bg = fitted["elo_goal_exponent"], fitted["dc_rho"], fitted["avg_goals_per_team"]
    sigmas = [0.6, 0.8, 1.0, 1.2, 1.5]
    hits = {f"map_sigma_{s}": 0 for s in sigmas}
    hits.update({"matrix_mode": 0, "rounded_xg": 0, "outcome": 0})
    old_sigma = pm.SCORE_TETHER_SIGMA
    for m in valid:
        lam, mu = _lambdas(m, exps, bg)
        probs = match_probabilities(lam, mu, rho)
        actual = (m["hg"], m["ag"])
        # 胜平负命中（参照系）
        pred_o = int(np.argmax(probs))
        act_o = 0 if m["hg"] > m["ag"] else (1 if m["hg"] == m["ag"] else 2)
        hits["outcome"] += (pred_o == act_o)
        # 策略1：矩阵众数
        M = score_matrix(lam, mu, rho)
        hits["matrix_mode"] += (divmod(int(np.argmax(M)), M.shape[0]) == actual)
        # 策略2：四舍五入 xG
        hits["rounded_xg"] += ((int(round(lam)), int(round(mu))) == actual)
        # 策略3：xG锚定 MAP（各档 σ）
        for s in sigmas:
            pm.SCORE_TETHER_SIGMA = s
            (h, a), _ = representative_score(lam, mu, probs, rho)
            hits[f"map_sigma_{s}"] += ((h, a) == actual)
    pm.SCORE_TETHER_SIGMA = old_sigma
    n = len(valid)
    return {k: round(v / n, 4) for k, v in hits.items()}, n


def main():
    rows = load_matches()
    wc = replay_elo(rows)
    train = [m for m in wc if m["year"] in TRAIN_YEARS]
    valid = [m for m in wc if m["year"] in VALID_YEARS]
    print(f"历史比赛 {len(rows)} 场；世界杯训练集 {len(train)} 场(1998-2014)，"
          f"验证集 {len(valid)} 场(2018+2022)")

    fitted = fit(train)
    print("\n── 最大似然拟合（真实世界杯比分）──")
    for k, v in fitted.items():
        print(f"  {k}: {v}")

    scores, n = evaluate(valid, fitted)
    print(f"\n── 样本外验证（{n} 场，精确比分命中率）──")
    print(f"  胜平负命中率(参照): {scores.pop('outcome')*100:.1f}%")
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s}: {v*100:.1f}%")

    best_sigma = max((k for k in scores if k.startswith("map_sigma")),
                     key=lambda k: scores[k])
    fitted["score_sigma"] = float(best_sigma.rsplit("_", 1)[1])
    fitted["valid_hit_rates"] = scores

    params = {}
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text())
    params["backtest"] = fitted
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n已写入 {PARAMS_PATH.name} 的 backtest 字段。")


if __name__ == "__main__":
    main()
