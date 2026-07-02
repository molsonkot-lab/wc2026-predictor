"""
攻防分离能力参数（Maher 1982 / Ley et al. 2019 时间衰减泊松排名法）。

这是 Groll et al.（2018 世界杯预测随机森林论文）验证过的最强单一特征：
用真实历史比分拟合每队的「进攻力 att」和「防守力 def」，
    E[主队进球] = base × att_home × def_away × host_factor
    E[客队进球] = base × att_away × def_home
att>1 = 攻强，def>1 = 防漏。与单一 Elo 标量的区别：同样实力差下，
攻强守弱的队（巴西型）和攻弱守强的队（摩洛哥型）比分分布完全不同——
这正是精确比分预测需要、而 Elo 给不了的信息。

拟合：时间衰减加权泊松最大似然（交替乘法更新 + 向 1.0 收缩防小样本过拟合）。
半衰期 2 年（Ley et al. 拟合的最优量级），窗口 8 年，友谊赛权重减半。
数据源：github.com/martj42/international_results 真实比分。
"""
import csv
import math
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Optional

DATA_CSV = Path(__file__).parent.parent / "data" / "intl_results.csv"
CSV_URL = ("https://raw.githubusercontent.com/martj42/"
           "international_results/master/results.csv")

HALF_LIFE_DAYS = 730       # 时间衰减半衰期（2年）
WINDOW_DAYS = 2920         # 只看最近 8 年
FRIENDLY_WEIGHT = 0.5      # 友谊赛信息量打对折
SHRINK_K = 8.0             # 收缩先验强度（约等于 8 场"均值队"虚拟比赛）
N_ITER = 30

# football-data.org API 队名 → 历史数据集队名（不一致的才需要列）
API_TO_DATASET = {
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Ireland": "Republic of Ireland",
    "Curaçao": "Curacao",
}


def _dataset_name(api_name: str) -> str:
    return API_TO_DATASET.get(api_name, api_name)


def load_rows(csv_path: Path = DATA_CSV):
    """读历史比分（比分为 NA 的未来场次自动跳过）。缺文件时自动下载。"""
    if not csv_path.exists():
        import urllib.request
        urllib.request.urlretrieve(CSV_URL, csv_path)
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            rows.append((r["date"], r["home_team"], r["away_team"],
                         int(r["home_score"]), int(r["away_score"]),
                         r["tournament"], r["neutral"].strip().upper() == "TRUE"))
    return rows


def fit_abilities(rows, asof: str) -> Dict:
    """
    用 asof（'YYYY-MM-DD'）之前的比赛拟合攻防参数（严格无未来函数）。
    返回 {"teams": {name: [att, def]}, "base": μ, "host": 主场进球乘子, "asof": …}
    """
    y, m, d = (int(x) for x in asof.split("-"))
    asof_d = date(y, m, d)

    sample = []   # (w, home, away, hg, ag, neutral)
    for dt, h, a, hg, ag, tour, neutral in rows:
        try:
            dd = (asof_d - date(*[int(x) for x in dt.split("-")])).days
        except ValueError:
            continue
        if dd <= 0 or dd > WINDOW_DAYS:
            continue
        w = 0.5 ** (dd / HALF_LIFE_DAYS)
        if tour == "Friendly":
            w *= FRIENDLY_WEIGHT
        sample.append((w, h, a, min(hg, 9), min(ag, 9), neutral))

    tw = sum(s[0] for s in sample)
    base = sum(s[0] * (s[3] + s[4]) for s in sample) / (2 * tw)
    # 主场进球乘子（中立场不计）
    hw = [(s[0], s[3], s[4]) for s in sample if not s[5]]
    host = (sum(w * hg for w, hg, _ in hw) /
            max(1e-9, sum(w * ag for w, _, ag in hw))) ** 0.5

    att, dfn = {}, {}
    teams = {t for s in sample for t in (s[1], s[2])}
    for t in teams:
        att[t] = dfn[t] = 1.0

    for _ in range(N_ITER):
        num_a = {t: SHRINK_K * base for t in teams}   # 收缩先验：K 场均值队
        den_a = {t: SHRINK_K * base for t in teams}
        num_d = {t: SHRINK_K * base for t in teams}
        den_d = {t: SHRINK_K * base for t in teams}
        for w, h, a, hg, ag, neutral in sample:
            fh = 1.0 if neutral else host
            fa = 1.0 if neutral else 1.0 / host
            num_a[h] += w * hg;  den_a[h] += w * base * dfn[a] * fh
            num_a[a] += w * ag;  den_a[a] += w * base * dfn[h] * fa
            num_d[h] += w * ag;  den_d[h] += w * base * att[a] * fa
            num_d[a] += w * hg;  den_d[a] += w * base * att[h] * fh
        for t in teams:
            att[t] = num_a[t] / den_a[t]
            dfn[t] = num_d[t] / den_d[t]
        # 归一化保证可辨识：均值锚在 1.0
        ma = sum(att.values()) / len(att)
        md = sum(dfn.values()) / len(dfn)
        for t in teams:
            att[t] /= ma
            dfn[t] /= md

    return {"asof": asof, "base": round(base, 4), "host": round(host, 4),
            "teams": {t: [round(att[t], 4), round(dfn[t], 4)] for t in teams}}


def ad_lambdas(abilities: Dict, home_api: str, away_api: str,
               base_goals: float = None) -> Optional[Tuple[float, float]]:
    """
    按攻防参数给出 (λ_home, μ_away)。世界杯按中立场处理（东道主优势由
    Elo 通道的 WC_HOST_ADVANTAGE 承担，这里不重复计）。
    任一队不在参数表里返回 None（调用方退回纯 Elo）。
    """
    th = abilities.get("teams", {}).get(_dataset_name(home_api))
    ta = abilities.get("teams", {}).get(_dataset_name(away_api))
    if not th or not ta:
        return None
    base = base_goals if base_goals is not None else abilities.get("base", 1.24)
    lam = base * th[0] * ta[1]
    mu = base * ta[0] * th[1]
    return max(0.3, min(4.5, lam)), max(0.3, min(4.5, mu))
