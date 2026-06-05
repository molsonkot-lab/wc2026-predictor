"""
Head-to-head historical records for WC 2026 match pairs.
Data sourced from football-data.org /v4/matches/{id}/head2head.
"""
import time
import requests
from typing import Dict, Optional, Tuple

from data.fetcher import _get_cached, _set_cache, FOOTBALL_API, FOOTBALL_DATA_TOKEN

H2H_TTL   = 30 * 24 * 60   # 30-day cache
H2H_MIN   = 3               # minimum H2H matches to trust the signal
H2H_SCALE = 100             # Elo pts per unit deviation (±0.5 → ±50 before cap)
H2H_MAX   = 40              # hard cap ±40 Elo pts (~±5-6% win probability)


def fetch_h2h(match_id: int) -> Optional[Dict]:
    """
    Fetch H2H aggregates for a scheduled match.
    Returns {"home_wins", "draws", "away_wins"} or None on error / rate-limit.
    Does NOT sleep — callers that batch-fetch should manage rate limiting.
    """
    key = f"h2h_{match_id}"
    cached = _get_cached(key, ttl_minutes=H2H_TTL)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{FOOTBALL_API}/matches/{match_id}/head2head",
            headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
            params={"limit": 10},
            timeout=15,
        )
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        data = resp.json()
        # football-data.org v4 uses either "head2head" or "aggregates" key
        block  = data.get("head2head") or data.get("aggregates") or {}
        home_s = block.get("homeTeam", {})
        away_s = block.get("awayTeam", {})
        result = {
            "home_wins": int(home_s.get("wins", 0)),
            "draws":     int(home_s.get("draws", 0)),
            "away_wins": int(away_s.get("wins", 0)),
        }
        _set_cache(key, result)
        return result
    except Exception:
        return None


def elo_adjustment(home_wins: int, draws: int, away_wins: int) -> float:
    """
    Elo adjustment for the home team based on H2H record.
    Positive  → home team historically dominates this opponent.
    Negative  → away team historically dominates.
    Capped at ±H2H_MAX.
    """
    total = home_wins + draws + away_wins
    if total < H2H_MIN:
        return 0.0
    home_score = home_wins + 0.5 * draws
    deviation  = (home_score / total) - 0.5   # −0.5 … +0.5
    return max(-H2H_MAX, min(H2H_MAX, deviation * H2H_SCALE))


def build_h2h_map_from_cache(matches: list) -> Dict[Tuple[int, int], float]:
    """
    Instant, non-blocking. Returns {(team_a_id, team_b_id): elo_adj}
    for all group stage pairs that are already in SQLite cache.
    The reverse pair (b, a) is stored with negated adjustment.
    """
    result: Dict[Tuple[int, int], float] = {}
    for m in matches:
        if m.get("stage") != "GROUP_STAGE":
            continue
        home_id = m["homeTeam"].get("id")
        away_id = m["awayTeam"].get("id")
        if not home_id or not away_id:
            continue
        cached = _get_cached(f"h2h_{m['id']}", ttl_minutes=H2H_TTL)
        if cached is None:
            continue
        adj = elo_adjustment(cached["home_wins"], cached["draws"], cached["away_wins"])
        result[(home_id, away_id)] =  adj
        result[(away_id, home_id)] = -adj
    return result


def count_cached(matches: list) -> Tuple[int, int]:
    """Returns (cached_count, total_group_matches)."""
    group = [
        m for m in matches
        if m.get("stage") == "GROUP_STAGE"
        and m["homeTeam"].get("id")
        and m["awayTeam"].get("id")
    ]
    cached = sum(
        1 for m in group
        if _get_cached(f"h2h_{m['id']}", ttl_minutes=H2H_TTL) is not None
    )
    return cached, len(group)


def fetch_all_h2h(matches: list, progress_cb=None) -> int:
    """
    Fetch H2H for all uncached group stage matches with rate-limit throttling.
    progress_cb(done, total) called after each fetch.
    Returns the number of newly fetched records.
    """
    group = [
        m for m in matches
        if m.get("stage") == "GROUP_STAGE"
        and m["homeTeam"].get("id")
        and m["awayTeam"].get("id")
    ]
    uncached = [
        m for m in group
        if _get_cached(f"h2h_{m['id']}", ttl_minutes=H2H_TTL) is None
    ]
    fetched = 0
    for i, m in enumerate(uncached):
        result = fetch_h2h(m["id"])
        if result is not None:
            fetched += 1
        if progress_cb:
            progress_cb(i + 1, len(uncached))
        time.sleep(7)   # respect 10 req/min free-tier limit
    return fetched
