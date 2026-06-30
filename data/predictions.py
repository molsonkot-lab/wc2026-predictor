"""
Prediction log + reconciliation (复盘).

Every time the model predicts a scheduled match we snapshot the prediction
(win/draw/loss probabilities + most-likely score). After the match finishes we
compare the snapshot against the real result to measure accuracy:

    - outcome hit : did the highest-probability result (W/D/L) actually happen?
    - score hit   : did the predicted exact scoreline happen?
    - Brier score : squared error of the W/D/L probability vector (lower better)
    - log-loss    : −log(prob assigned to the actual outcome) (lower better)

This is the real-data QC loop that lets the model be tuned over time instead of
guessed at.
"""
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent / "cache.db"


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


def log_prediction(match_id: int, home_name: str, away_name: str,
                   p_home: float, p_draw: float, p_away: float,
                   pred_home: int, pred_away: int, utc_date: str = ""):
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
             pred_home, pred_away, utc_date, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, [match_id, home_name, away_name, p_home, p_draw, p_away,
              pred_home, pred_away, utc_date,
              datetime.now(timezone.utc).isoformat()])


def _outcome(hg: int, ag: int) -> str:
    return "H" if hg > ag else ("A" if ag > hg else "D")


def reconcile(finished: Dict[int, tuple], match_meta: Dict[int, dict] = None) -> List[dict]:
    """
    finished: {match_id: (home_goals, away_goals)} for FINISHED matches.
    Returns a per-match review list, newest first, each with:
        names, predicted probs/score, actual score, outcome_hit, score_hit,
        brier, logloss.
    """
    _init()
    match_meta = match_meta or {}
    rows = []
    with _db() as conn:
        logged = {r["match_id"]: r for r in conn.execute("SELECT * FROM predictions")}

    reviews = []
    for mid, (hg, ag) in finished.items():
        r = logged.get(mid)
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

        reviews.append({
            "match_id": mid,
            "home_name": r["home_name"],
            "away_name": r["away_name"],
            "p_home": r["p_home"], "p_draw": r["p_draw"], "p_away": r["p_away"],
            "pred_home": r["pred_home"], "pred_away": r["pred_away"],
            "actual_home": hg, "actual_away": ag,
            "outcome_hit": pred_outcome == actual,
            "score_hit": (r["pred_home"] == hg and r["pred_away"] == ag),
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
        "avg_brier": sum(r["brier"] for r in reviews) / n,
        "avg_logloss": sum(r["logloss"] for r in reviews) / n,
    }
