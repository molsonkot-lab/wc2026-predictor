"""
Prediction log + reconciliation (复盘).

Every time the model predicts a scheduled match we snapshot the prediction
(win/draw/loss probabilities + model scoreline + mystic scoreline + xG). After
the match finishes we compare the snapshot against the real result:

    - outcome hit : did the highest-probability result (W/D/L) actually happen?
    - score hit   : did EITHER the model scoreline OR the mystic scoreline
                    match the exact result? (双轨比分，任一命中即算命中)
    - Brier score : squared error of the W/D/L probability vector (lower better)
    - log-loss    : −log(prob assigned to the actual outcome) (lower better)
    - goal diagnostics : per-match total-goal error, so systematic over/under
                    prediction of goals is visible and tunable.

The mystic scoreline is always snapshotted at the canonical strength 0.5 and
without venue feng-shui, so the review metric is stable regardless of what the
slider/city happened to be set to when the page was viewed. Legacy rows logged
before the mystic columns existed get a deterministic best-effort recompute.
"""
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "cache.db"

# 复盘用的玄学口径：固定强度，与侧边栏滑块无关，保证指标可比。
MYSTIC_REVIEW_STRENGTH = 0.5

_EXTRA_COLS = [
    ("myst_home", "INTEGER"),   # 玄学比分（强度0.5口径）
    ("myst_away", "INTEGER"),
    ("xg_home", "REAL"),        # 预测时的连续 xG，供进球水平诊断/调参
    ("xg_away", "REAL"),
]


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                match_id   INTEGER PRIMARY KEY,
                home_name  TEXT,
                away_name  TEXT,
                p_home     REAL,
                p_draw     REAL,
                p_away     REAL,
                pred_home  INTEGER,
                pred_away  INTEGER,
                utc_date   TEXT,
                created_at TEXT
            );
        """)
        # 老库迁移：补上玄学比分/xG 列（已存在则跳过）
        have = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
        for col, typ in _EXTRA_COLS:
            if col not in have:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")


def log_prediction(match_id: int, home_name: str, away_name: str,
                   p_home: float, p_draw: float, p_away: float,
                   pred_home: int, pred_away: int, utc_date: str = "",
                   myst_home: int = None, myst_away: int = None,
                   xg_home: float = None, xg_away: float = None):
    """
    Snapshot a prediction. Only the FIRST prediction per match is kept (the one
    made while the match was still upcoming), so reconciliation measures a
    genuine forecast, not hindsight. Call this for SCHEDULED matches.
    """
    _init()
    with _db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO predictions
            (match_id, home_name, away_name, p_home, p_draw, p_away,
             pred_home, pred_away, utc_date, created_at,
             myst_home, myst_away, xg_home, xg_away)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [match_id, home_name, away_name, p_home, p_draw, p_away,
              pred_home, pred_away, utc_date,
              datetime.now(timezone.utc).isoformat(),
              myst_home, myst_away, xg_home, xg_away])


def _outcome(hg: int, ag: int) -> str:
    return "H" if hg > ag else ("A" if ag > hg else "D")


def _legacy_mystic_score(r) -> tuple:
    """
    老记录没存玄学比分时的确定性补算：用存档队名/日期重推玄学 bias，
    按复盘口径（强度0.5、不含场地）作用到存档 xG（缺 xG 则用整数比分近似）。
    """
    from models.metaphysics import metaphysics_reading, tilt_xg
    reading = metaphysics_reading(r["home_name"], r["away_name"], "",
                                  (r["utc_date"] or "")[:10])
    lam = r["xg_home"] if r["xg_home"] is not None else float(r["pred_home"])
    mu = r["xg_away"] if r["xg_away"] is not None else float(r["pred_away"])
    tl, tm = tilt_xg(lam, mu, reading["bias"], MYSTIC_REVIEW_STRENGTH)
    return int(round(tl)), int(round(tm))


def _row_from_repo_log(rec: dict, market_weight: float,
                       avg_goals: float = None) -> Optional[dict]:
    """
    从 data/prediction_log.json（auto_tune 每晚快照、随 git 持久化）恢复一条
    预测记录。线上 Streamlit Cloud 的 SQLite 每次重新部署都会被清空，没有这层
    兜底，历史命中（如 科特迪瓦1-2挪威、英格兰2-1刚果）就会从复盘里消失。

    新版快照直接带 pred_score/myst_score；老快照只有赛前概率，则做**忠实还原**：
    当年 app 展示的比分 = 纯 Elo xG 四舍五入（进球基线还是 1.40 的年代）。
    反推纯 model_p（它就是当时 Elo lambdas 过 DC 模型的输出，反演即恢复原
    lambdas）再按老规则取整——复盘的是"当时真实给出的预测"，不是新算法的
    事后重算。概率轨（Brier/胜负）仍用含市场的混合概率，与当时展示一致。
    """
    from models.poisson_model import fit_lambdas_to_probs, blend_with_odds
    from models.metaphysics import metaphysics_reading, tilt_xg
    mp, kp = rec.get("model_p"), rec.get("market_p")
    if not mp:
        return None
    p = blend_with_odds(tuple(mp), tuple(kp), market_weight) if kp else tuple(mp)
    if rec.get("pred_score"):
        ph_, pa_ = rec["pred_score"]
        mh, ma = rec.get("myst_score", rec["pred_score"])
        lam, mu = rec.get("xg") or (float(ph_), float(pa_))
    else:
        LEGACY_TOTAL = 2.8   # 老配置 AVG_GOALS_PER_TEAM=1.40 × 2
        lam, mu = fit_lambdas_to_probs(*mp, total=LEGACY_TOTAL)
        ph_, pa_ = int(round(lam)), int(round(mu))
        reading = metaphysics_reading(rec.get("home", ""), rec.get("away", ""),
                                      "", rec.get("utc", "")[:10])
        tl, tm = tilt_xg(lam, mu, reading["bias"], MYSTIC_REVIEW_STRENGTH)
        mh, ma = int(round(tl)), int(round(tm))
    return {
        "home_name": rec.get("home", "?"), "away_name": rec.get("away", "?"),
        "p_home": p[0], "p_draw": p[1], "p_away": p[2],
        "pred_home": int(ph_), "pred_away": int(pa_),
        "myst_home": int(mh), "myst_away": int(ma),
        "xg_home": lam, "xg_away": mu,
        "utc_date": rec.get("utc", ""),
    }


def reconcile(finished: Dict[int, tuple], match_meta: Dict[int, dict] = None,
              repo_log: Dict[str, dict] = None, market_weight: float = 0.7,
              avg_goals: float = None) -> List[dict]:
    """
    finished: {match_id: (home_goals, away_goals)} for FINISHED matches.
    repo_log: data/prediction_log.json 的内容——SQLite 里缺的比赛从这里恢复
              （云端文件系统随部署重置，git 里的快照才是持久档案）。
    Returns a per-match review list, newest first, each with:
        names, predicted probs/scores (model + mystic), actual score,
        outcome_hit, score_hit (either-track), brier, logloss, goal errors.
    """
    _init()
    match_meta = match_meta or {}
    repo_log = repo_log or {}
    with _db() as conn:
        logged = {r["match_id"]: r for r in conn.execute("SELECT * FROM predictions")}

    reviews = []
    for mid, (hg, ag) in finished.items():
        r, source = logged.get(mid), "live"
        if r is None and str(mid) in repo_log:
            r, source = _row_from_repo_log(repo_log[str(mid)], market_weight,
                                           avg_goals), "repo"
        if r is None:
            continue
        actual = _outcome(hg, ag)
        probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        pred_outcome = max(probs, key=probs.get)
        p_actual = max(probs[actual], 1e-9)

        # Brier over the 3-class vector
        target = {"H": 0.0, "D": 0.0, "A": 0.0}
        target[actual] = 1.0
        brier = sum((probs[k] - target[k]) ** 2 for k in probs)

        mh, ma = r["myst_home"], r["myst_away"]
        if mh is None or ma is None:
            mh, ma = _legacy_mystic_score(r)

        score_hit_base = (r["pred_home"] == hg and r["pred_away"] == ag)
        score_hit_myst = (mh == hg and ma == ag)

        reviews.append({
            "match_id": mid,
            "source": source,
            "home_name": r["home_name"],
            "away_name": r["away_name"],
            "p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
            "pred_home": r["pred_home"], "pred_away": r["pred_away"],
            "myst_home": mh, "myst_away": ma,
            "xg_home": r["xg_home"], "xg_away": r["xg_away"],
            "actual_home": hg, "actual_away": ag,
            "outcome_hit": pred_outcome == actual,
            # 双轨比分：模型或玄学任一命中即算命中
            "score_hit": score_hit_base or score_hit_myst,
            "score_hit_base": score_hit_base,
            "score_hit_myst": score_hit_myst,
            # 进球诊断：正=预测进球偏多
            "goal_err": (r["pred_home"] + r["pred_away"]) - (hg + ag),
            "brier": brier,
            "logloss": -math.log(p_actual),
            "utc_date": r["utc_date"],
        })

    reviews.sort(key=lambda x: x["utc_date"] or "", reverse=True)
    return reviews


def summary(reviews: List[dict]) -> Optional[dict]:
    """Aggregate accuracy metrics across reconciled matches."""
    if not reviews:
        return None
    n = len(reviews)
    return {
        "n": n,
        "outcome_acc": sum(r["outcome_hit"] for r in reviews) / n,
        "score_acc": sum(r["score_hit"] for r in reviews) / n,
        "score_acc_base": sum(r["score_hit_base"] for r in reviews) / n,
        "score_acc_myst": sum(r["score_hit_myst"] for r in reviews) / n,
        "avg_brier": sum(r["brier"] for r in reviews) / n,
        "avg_logloss": sum(r["logloss"] for r in reviews) / n,
        # 进球水平诊断：偏差(带符号，正=整体预测偏多) 与平均绝对误差
        "goal_bias": sum(r["goal_err"] for r in reviews) / n,
        "goal_mae": sum(abs(r["goal_err"]) for r in reviews) / n,
    }
