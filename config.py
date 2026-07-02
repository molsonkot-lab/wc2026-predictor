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

# International football goal averages.
# 以下三个值由 scripts/backtest.py 在 1998–2014 五届世界杯 320 场真实比分上
# 最大似然拟合、2018+2022 两届 128 场样本外验证（详见 tuned_params.json 的
# backtest 字段）。别再手拍——要改先跑回测。
AVG_GOALS_PER_TEAM = 1.24  # 真实世界杯场均每队进球（此前手拍 1.40 系统性偏高；
                           # 每晚 auto_tune 再按 2026 实际战绩微调）
DC_RHO = 0.0               # Dixon-Coles 低分修正在世界杯数据上拟合为 0
# How strongly an Elo gap translates into a goal-difference (xG) gap.
# 回测最优值恰为 0.70（网格 0.30–1.00 逐 0.05 搜索）。
ELO_GOAL_EXPONENT = 0.70

# How much to trust bookmaker odds vs Elo (0=pure Elo, 1=pure odds)
ODDS_BLEND_WEIGHT = 0.35

# Player impact on Elo: how many points team loses per status
PLAYER_IMPACT = {
    "OUT":      1.0,   # full impact
    "DOUBTFUL": 0.5,   # half impact
    "FIT":      0.0,
}
