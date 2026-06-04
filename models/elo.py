import math
from typing import Dict, Tuple
from config import ELO_K, HOST_TEAMS, WC_HOST_ADVANTAGE

# Seed Elo ratings for all 48 WC 2026 teams (by TLA)
# Based on World Football Elo Ratings pre-tournament estimates
SEED_ELO: Dict[str, float] = {
    # Elite
    "FRA": 2085, "ESP": 2060, "ENG": 2040, "BRA": 2030,
    "ARG": 2020, "POR": 2000, "NED": 1975, "GER": 1970,
    # Strong
    "BEL": 1945, "COL": 1930, "MAR": 1920, "SUI": 1915,
    "URY": 1905, "CRO": 1900, "USA": 1875, "NOR": 1865,
    "JPN": 1860, "TUR": 1850, "SEN": 1840, "MEX": 1835,
    "ECU": 1825, "AUT": 1820, "CAN": 1815, "SWE": 1810,
    "KOR": 1805, "CZE": 1800,
    # Mid
    "ALG": 1790, "GHA": 1780, "AUS": 1775, "CIV": 1770,
    "TUN": 1755, "EGY": 1750, "RSA": 1745, "BIH": 1740,
    "SCO": 1730, "KSA": 1725, "IRN": 1720, "IRQ": 1715,
    "COD": 1710, "QAT": 1700, "CPV": 1690,
    # Lower
    "PAR": 1685, "UZB": 1680, "JOR": 1670,
    "NZL": 1660, "HAI": 1655, "PAN": 1650, "CUW": 1600,
}


class EloSystem:
    def __init__(self):
        self.ratings: Dict[int, float] = {}
        self.tla_to_id: Dict[str, int] = {}
        self.id_to_tla: Dict[int, str] = {}
        self.id_to_name: Dict[int, str] = {}

    def initialize(self, teams: list):
        for t in teams:
            tid = t["id"]
            tla = t.get("tla", "")
            name = t.get("name", "")
            self.tla_to_id[tla] = tid
            self.id_to_tla[tid] = tla
            self.id_to_name[tid] = name
            if tid not in self.ratings:
                self.ratings[tid] = SEED_ELO.get(tla, 1700.0)

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, 1700.0)

    def get_tla(self, team_id: int) -> str:
        return self.id_to_tla.get(team_id, "")

    def _home_advantage(self, home_id: int) -> float:
        tla = self.id_to_tla.get(home_id, "")
        return WC_HOST_ADVANTAGE if tla in HOST_TEAMS else 0.0

    def win_draw_loss(self, home_id: int, away_id: int,
                      home_elo_override: float = None,
                      away_elo_override: float = None) -> Tuple[float, float, float]:
        """
        Returns (p_home_win, p_draw, p_away_win) using Elo difference.
        Accepts optional overrides for player-adjusted Elo.
        """
        r_h = home_elo_override if home_elo_override is not None else self.get_rating(home_id)
        r_a = away_elo_override if away_elo_override is not None else self.get_rating(away_id)

        dr = (r_h - r_a + self._home_advantage(home_id)) / 400
        p_h_raw = 1 / (1 + 10 ** (-dr))

        # Draw probability is highest at evenly matched contests
        imbalance = abs(2 * p_h_raw - 1)
        p_draw = 0.29 * (1 - imbalance ** 0.9)

        p_home = p_h_raw * (1 - p_draw)
        p_away = 1 - p_home - p_draw

        return max(0.02, p_home), max(0.02, p_draw), max(0.02, p_away)

    def update(self, home_id: int, away_id: int,
               home_goals: int, away_goals: int,
               stage: str = "GROUP_STAGE"):
        """Update ratings after a match result."""
        k = ELO_K.get(stage, 30)
        r_h = self.get_rating(home_id)
        r_a = self.get_rating(away_id)

        dr = (r_h - r_a + self._home_advantage(home_id)) / 400
        e_h = 1 / (1 + 10 ** (-dr))
        e_a = 1 - e_h

        if home_goals > away_goals:
            s_h, s_a = 1.0, 0.0
        elif home_goals == away_goals:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        # Goal-difference multiplier (diminishing returns)
        gd = abs(home_goals - away_goals)
        gd_mult = math.log(gd + 1) * 0.5 + 1 if gd > 0 else 1.0

        self.ratings[home_id] = r_h + k * gd_mult * (s_h - e_h)
        self.ratings[away_id] = r_a + k * gd_mult * (s_a - e_a)

    def snapshot(self) -> Dict[int, float]:
        return dict(self.ratings)

    def restore(self, snapshot: Dict[int, float]):
        self.ratings = {int(k): v for k, v in snapshot.items()}
