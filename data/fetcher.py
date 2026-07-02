import requests
import sqlite3
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from config import FOOTBALL_DATA_TOKEN, ODDS_API_KEY, WC_CODE, WC_SEASON

DB_PATH = Path(__file__).parent / "cache.db"
FOOTBALL_API = "https://api.football-data.org/v4"
ODDS_API = "https://api.the-odds-api.com/v4"


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                updated_at TEXT
            );
        """)


def _get_cached(key: str, ttl_minutes: int = 15):
    _init_db()
    with _db() as conn:
        row = conn.execute("SELECT data, updated_at FROM cache WHERE key=?", [key]).fetchone()
        if row:
            updated = datetime.fromisoformat(row["updated_at"])
            age = (datetime.now(timezone.utc) - updated.replace(tzinfo=timezone.utc)).total_seconds() / 60
            if age < ttl_minutes:
                return json.loads(row["data"])
    return None


def _set_cache(key: str, data):
    _init_db()
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, data, updated_at) VALUES (?, ?, ?)",
            [key, json.dumps(data), datetime.now(timezone.utc).isoformat()]
        )


def fetch_wc_matches(force_refresh: bool = False):
    key = f"wc_matches_{WC_SEASON}"
    if not force_refresh:
        cached = _get_cached(key, ttl_minutes=10)
        if cached is not None:
            return cached

    resp = requests.get(
        f"{FOOTBALL_API}/competitions/{WC_CODE}/matches",
        headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
        params={"season": WC_SEASON},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()["matches"]
    _set_cache(key, data)
    return data


def fetch_wc_teams(force_refresh: bool = False):
    key = f"wc_teams_{WC_SEASON}"
    if not force_refresh:
        cached = _get_cached(key, ttl_minutes=1440)  # 24h
        if cached is not None:
            return cached

    resp = requests.get(
        f"{FOOTBALL_API}/competitions/{WC_CODE}/teams",
        headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
        params={"season": WC_SEASON},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()["teams"]
    _set_cache(key, data)
    return data


def fetch_team_squad(team_id: int, force_refresh: bool = False):
    """Fetch full squad for a team, rate-limited to avoid 429"""
    key = f"squad_{team_id}"
    if not force_refresh:
        cached = _get_cached(key, ttl_minutes=720)
        if cached is not None:
            return cached

    time.sleep(7)  # respect 10 req/min free tier limit
    try:
        resp = requests.get(
            f"{FOOTBALL_API}/teams/{team_id}",
            headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("squad", [])
        _set_cache(key, data)
        return data
    except Exception:
        return []


def fetch_match_odds(force_refresh: bool = False):
    # h2h + totals 同一次请求拿全（The Odds API 按 markets×regions 计费，
    # 这里 2 credits/次、缓存1小时——大小球盘口是市场对总进球的直接定价，
    # 用于把预测比分的总进球水平锚到市场）。
    key = "wc_odds_h2h_totals"
    if not force_refresh:
        cached = _get_cached(key, ttl_minutes=60)
        if cached is not None:
            return cached

    try:
        resp = requests.get(
            f"{ODDS_API}/sports/soccer_fifa_world_cup/odds/",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "eu",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
                "bookmakers": "pinnacle,betfair,bet365",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        _set_cache(key, data)
        return data
    except Exception:
        return []


def fetch_outright_odds(force_refresh: bool = False):
    key = "wc_odds_winner"
    if not force_refresh:
        cached = _get_cached(key, ttl_minutes=120)
        if cached is not None:
            return cached

    try:
        resp = requests.get(
            f"{ODDS_API}/sports/soccer_fifa_world_cup_winner/odds/",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "eu",
                "markets": "outrights",
                "oddsFormat": "decimal",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        _set_cache(key, data)
        return data
    except Exception:
        return []


def build_odds_map(odds_data: list) -> dict:
    """
    Returns {frozenset({name_a, name_b}): (p_home, p_draw, p_away)}
    Uses Pinnacle > Betfair > first available bookmaker.
    Removes bookmaker vig via basic normalization.
    """
    odds_map = {}
    for game in odds_data:
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        bk = (
            next((b for b in bookmakers if b["title"] == "Pinnacle"), None)
            or next((b for b in bookmakers if b["title"] == "Betfair"), None)
            or bookmakers[0]
        )
        for market in bk.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            if len(outcomes) < 2:
                continue

            home_name = game["home_team"]
            away_name = game["away_team"]
            draw_key = "Draw"

            home_odds = outcomes.get(home_name, 2.0)
            away_odds = outcomes.get(away_name, 2.0)
            draw_odds = outcomes.get(draw_key, 3.5)

            raw = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
            total = sum(raw)
            p_home, p_draw, p_away = [r / total for r in raw]

            key = frozenset([home_name, away_name])
            odds_map[key] = (p_home, p_draw, p_away)

    return odds_map


def build_totals_map(odds_data: list) -> dict:
    """
    Returns {frozenset({name_a, name_b}): (line, p_over)} from the totals
    (over/under) market, de-vigged. Prefers Pinnacle, line closest to 2.5.
    """
    tmap = {}
    for game in odds_data:
        bookmakers = game.get("bookmakers", [])
        bk = (
            next((b for b in bookmakers if b["title"] == "Pinnacle"), None)
            or next((b for b in bookmakers if b["title"] == "Betfair"), None)
            or (bookmakers[0] if bookmakers else None)
        )
        if not bk:
            continue
        best = None   # (|line-2.5|, line, p_over)
        for market in bk.get("markets", []):
            if market["key"] != "totals":
                continue
            by_line = {}
            for o in market.get("outcomes", []):
                pt = o.get("point")
                if pt is None or o.get("price", 0) <= 1:
                    continue
                by_line.setdefault(pt, {})[o["name"]] = o["price"]
            for line, prices in by_line.items():
                if "Over" not in prices or "Under" not in prices:
                    continue
                po = (1 / prices["Over"]) / (1 / prices["Over"] + 1 / prices["Under"])
                cand = (abs(line - 2.5), line, po)
                if best is None or cand < best:
                    best = cand
        if best:
            tmap[frozenset([game["home_team"], game["away_team"]])] = (best[1], best[2])
    return tmap


def build_outright_map(outright_data: list) -> dict:
    """Returns {team_name_lower: probability} from outright winner odds"""
    result = {}
    for event in outright_data:
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                outcomes = market.get("outcomes", [])
                raw = [1 / o["price"] for o in outcomes if o["price"] > 0]
                total = sum(raw)
                if total == 0:
                    continue
                for o in outcomes:
                    p = (1 / o["price"]) / total
                    result[o["name"].lower()] = p
            break  # first bookmaker only
        break
    return result


def get_fixed_results(matches: list) -> dict:
    """
    Returns {match_id: (home_goals, away_goals)} —— **常规90分钟比分**。

    1X2 盘口、Dixon-Coles 模型、Elo、复盘统计的参照系都是常规时间。
    football-data 的 fullTime 在加时会计入加时球、点球大战时**连点球数都
    计入**（如 德国vs巴拉圭 常规 1-1、点球 3-4，fullTime 给 4-5）——
    直接用会同时污染 Elo（净胜球放大）、复盘命中率和每晚市场调参。
    加时/点球细节由 get_result_details() 单独提供。
    """
    fixed = {}
    for m in matches:
        if m["status"] != "FINISHED":
            continue
        sc = m["score"]
        src = sc["fullTime"]
        if sc.get("duration", "REGULAR") != "REGULAR":
            src = sc.get("regularTime") or src
        if src["home"] is not None and src["away"] is not None:
            fixed[m["id"]] = (int(src["home"]), int(src["away"]))
    return fixed


def get_result_details(matches: list) -> dict:
    """
    加时/点球另册记录：{match_id: {"duration", "et", "pens", "winner"}}
    仅含非常规时间结束的已完赛场次。et = 120分钟比分，pens = 点球比分。
    """
    det = {}
    for m in matches:
        if m["status"] != "FINISHED":
            continue
        sc = m["score"]
        dur = sc.get("duration", "REGULAR")
        if dur == "REGULAR":
            continue
        ft = sc.get("fullTime", {})
        pens = sc.get("penalties")
        et = None
        if dur == "EXTRA_TIME":
            et = (ft.get("home"), ft.get("away"))
        elif dur == "PENALTY_SHOOTOUT":
            rt = sc.get("regularTime", {})
            ex = sc.get("extraTime", {})
            if rt.get("home") is not None and ex.get("home") is not None:
                et = (rt["home"] + ex["home"], rt["away"] + ex["away"])
        det[m["id"]] = {
            "duration": dur,
            "et": et,
            "pens": (pens["home"], pens["away"]) if pens else None,
            "winner": sc.get("winner"),
        }
    return det


def get_ko_winners(matches: list) -> dict:
    """
    已完赛淘汰赛的真实晋级方：{frozenset({home_id, away_id}): winner_id}。
    供模拟器锁定——已经踢完的淘汰赛不允许在蒙特卡洛里被"重新踢一遍"
    （法国真实晋级了，模拟里就不能再输掉 R32），点球晋级方也以此为准。
    """
    winners = {}
    for m in matches:
        if m["status"] != "FINISHED" or m.get("stage") == "GROUP_STAGE":
            continue
        hid, aid = m["homeTeam"].get("id"), m["awayTeam"].get("id")
        w = m["score"].get("winner")
        if not hid or not aid or w not in ("HOME_TEAM", "AWAY_TEAM"):
            continue
        winners[frozenset((hid, aid))] = hid if w == "HOME_TEAM" else aid
    return winners
