def _get_secret(key: str, fallback: str) -> str:
    """Read from Streamlit secrets (cloud) or fall back to hardcoded value (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, fallback)
    except Exception:
        return fallback

FOOTBALL_DATA_TOKEN = _get_secret("FOOTBALL_DATA_TOKEN", "8c16981ea5514c61b9edb6380a0c625c")
ODDS_API_KEY        = _get_secret("ODDS_API_KEY",        "7d09f26985bd580cc6486fd3bac20857")

WC_CODE = "WC"
WC_SEASON = 2026
N_SIMULATIONS = 30_000

# Elo K-factors per tournament stage
ELO_K = {
    "GROUP_STAGE": 30,
    "ROUND_OF_32": 45,
    "ROUND_OF_16": 45,
    "QUARTER_FINALS": 60,
    "SEMI_FINALS": 60,
    "THIRD_PLACE": 45,
    "FINAL": 60,
}

# Home advantage in Elo points (USA/MEX/CAN are hosts)
HOST_TEAMS = {"USA", "MEX", "CAN"}
HOME_ADVANTAGE_ELO = 65    # full home advantage
WC_HOST_ADVANTAGE = 45     # reduced at World Cup (neutral stadiums mostly)

# International football goal averages
AVG_GOALS_PER_TEAM = 1.25  # goals per team per match
DC_RHO = 0.08              # Dixon-Coles low-score correction

# How much to trust bookmaker odds vs Elo (0=pure Elo, 1=pure odds)
ODDS_BLEND_WEIGHT = 0.35

# Player impact on Elo: how many points team loses per status
PLAYER_IMPACT = {
    "OUT":      1.0,   # full impact
    "DOUBTFUL": 0.5,   # half impact
    "FIT":      0.0,
}
