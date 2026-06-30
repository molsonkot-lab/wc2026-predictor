"""
Forward-looking self-correction (runs in GitHub Actions, nightly).

Two jobs each run:
  1. SNAPSHOT every still-upcoming match: pure-model W/D/L probs + current
     bookmaker odds → append to data/prediction_log.json (committed to repo).
  2. TUNE market_weight: over logged matches that have since FINISHED and had a
     bookmaker snapshot, find the model↔market blend weight minimising log-loss.
     Written to tuned_params.json, which the live app reads as its default.

Why forward-looking: pre-match bookmaker odds for already-finished matches can't
be recovered after kickoff, so the loop can only learn from matches that finish
*after* they were snapshotted. With < MIN_SAMPLES finished samples it keeps the
default weight (guards against overfitting on a tiny sample).
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.fetcher import (fetch_wc_matches, fetch_wc_teams, fetch_match_odds,
                          build_odds_map, get_fixed_results)
from data.players import compute_player_adjusted_elo, KEY_PLAYERS
from models.elo import EloSystem
from models.poisson_model import elo_to_lambdas, match_probabilities
from config import HOST_TEAMS, WC_HOST_ADVANTAGE

LOG_PATH    = ROOT / "data" / "prediction_log.json"
PARAMS_PATH = ROOT / "tuned_params.json"
MIN_SAMPLES = 5
IDX = {"H": 0, "D": 1, "A": 2}


def _load(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _outcome(hg, ag):
    return "H" if hg > ag else ("A" if ag > hg else "D")


def main():
    matches = fetch_wc_matches(force_refresh=True)
    teams   = fetch_wc_teams()
    odds_map = build_odds_map(fetch_match_odds(force_refresh=True))
    fixed    = get_fixed_results(matches)

    # Current Elo (updated with all finished results) + injury adjustments.
    elo = EloSystem()
    elo.initialize(teams)
    for m in sorted(matches, key=lambda x: x["utcDate"]):
        if m["status"] == "FINISHED" and m["id"] in fixed:
            hg, ag = fixed[m["id"]]
            elo.update(m["homeTeam"]["id"], m["awayTeam"]["id"], hg, ag,
                       m.get("stage", "GROUP_STAGE"))
    statuses = {p["name"]: p["status"] for v in KEY_PLAYERS.values() for p in v}

    log = _load(LOG_PATH, {})

    # ── 1. Snapshot upcoming, not-yet-logged matches ──────────────────
    new_snaps = 0
    for m in matches:
        mid = str(m["id"])
        ht, at = m["homeTeam"], m["awayTeam"]
        if not ht.get("id") or not at.get("id"):
            continue
        if m["status"] == "FINISHED" or mid in log:
            continue
        h_name, a_name = ht.get("name", ""), at.get("name", "")
        adj_h = compute_player_adjusted_elo(elo.get_rating(ht["id"]), ht.get("tla", ""), statuses)
        adj_a = compute_player_adjusted_elo(elo.get_rating(at["id"]), at.get("tla", ""), statuses)
        host = WC_HOST_ADVANTAGE if ht.get("tla", "") in HOST_TEAMS else 0.0
        lam, mu = elo_to_lambdas(adj_h, adj_a, host)
        mp = match_probabilities(lam, mu)
        market = odds_map.get(frozenset([h_name, a_name]))
        log[mid] = {
            "home": h_name, "away": a_name, "utc": m["utcDate"],
            "model_p": [round(x, 5) for x in mp],
            "market_p": [round(x, 5) for x in market] if market else None,
        }
        new_snaps += 1

    # ── 2. Build tuning set & optimise market_weight ──────────────────
    samples = []
    for mid, rec in log.items():
        imid = int(mid)
        if imid in fixed and rec.get("market_p"):
            hg, ag = fixed[imid]
            samples.append((rec["model_p"], rec["market_p"], _outcome(hg, ag)))

    params = _load(PARAMS_PATH, {"market_weight": 0.7})
    if len(samples) >= MIN_SAMPLES:
        best_w, best_ll = params.get("market_weight", 0.7), float("inf")
        for i in range(21):
            w = i / 20.0
            ll = 0.0
            for mp, kp, o in samples:
                p = (1 - w) * mp[IDX[o]] + w * kp[IDX[o]]
                ll += -math.log(max(p, 1e-9))
            ll /= len(samples)
            if ll < best_ll:
                best_ll, best_w = ll, w
        params = {
            "market_weight": best_w,
            "n_samples": len(samples),
            "logloss": round(best_ll, 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        params["n_samples"] = len(samples)
        params["note"] = f"samples<{MIN_SAMPLES}, keeping default to avoid overfit"
        params["updated_at"] = datetime.now(timezone.utc).isoformat()

    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"snapshots added: {new_snaps} | tuning samples: {len(samples)} "
          f"| market_weight -> {params['market_weight']}")


if __name__ == "__main__":
    main()
