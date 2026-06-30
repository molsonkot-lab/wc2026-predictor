"""
Monte Carlo tournament simulator for FIFA World Cup 2026.
48 teams → Group Stage → Round of 32 → R16 → QF → SF → Final.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from models.elo import EloSystem
from models.poisson_model import elo_to_lambdas, sample_score_dc
from config import HOST_TEAMS, WC_HOST_ADVANTAGE

# Type alias for H2H map: {(team_a_id, team_b_id): elo_adjustment_for_a}
H2HMap = dict


class GroupStandings:
    def __init__(self, teams: List[int]):
        self.teams = teams
        self.pts = defaultdict(int)
        self.gf = defaultdict(int)
        self.ga = defaultdict(int)

    def add_result(self, home: int, away: int, hg: int, ag: int):
        self.gf[home] += hg; self.ga[home] += ag
        self.gf[away] += ag; self.ga[away] += hg
        if hg > ag:
            self.pts[home] += 3
        elif hg == ag:
            self.pts[home] += 1; self.pts[away] += 1
        else:
            self.pts[away] += 3

    def ranked(self) -> List[int]:
        def key(t):
            return (-self.pts[t], -(self.gf[t] - self.ga[t]), -self.gf[t])
        return sorted(self.teams, key=key)

    def third_record(self, team: int) -> Tuple[int, int, int]:
        return (self.pts[team], self.gf[team] - self.ga[team], self.gf[team])


class TournamentSimulator:
    def __init__(self, matches: List[Dict], teams: List[Dict], elo: EloSystem,
                 h2h_map: H2HMap = None):
        self.matches = matches
        self.teams = {t["id"]: t for t in teams}
        self.elo = elo
        self.h2h_map: H2HMap = h2h_map or {}
        self.groups = self._build_groups()

    def _build_groups(self) -> Dict[str, List[int]]:
        g: Dict[str, set] = defaultdict(set)
        for m in self.matches:
            if m["stage"] == "GROUP_STAGE" and m.get("group"):
                grp = m["group"]
                ht = m["homeTeam"].get("id")
                at = m["awayTeam"].get("id")
                if ht: g[grp].add(ht)
                if at: g[grp].add(at)
        return {k: list(v) for k, v in sorted(g.items())}

    def _host_adv(self, team_id: int) -> float:
        return WC_HOST_ADVANTAGE if self.teams.get(team_id, {}).get("tla", "") in HOST_TEAMS else 0.0

    def _h2h_adj(self, team_a: int, team_b: int) -> float:
        """Elo adjustment for team_a when facing team_b based on H2H history."""
        return self.h2h_map.get((team_a, team_b), 0.0)

    def _sim_match(self, home: int, away: int,
                   elo_overrides: Dict[int, float] = None) -> Tuple[int, int]:
        elo_overrides = elo_overrides or {}
        rh = elo_overrides.get(home, self.elo.get_rating(home))
        ra = elo_overrides.get(away, self.elo.get_rating(away))
        adj = self._h2h_adj(home, away)
        lam, mu = elo_to_lambdas(rh + adj, ra - adj, self._host_adv(home))
        return sample_score_dc(lam, mu)

    def _sim_knockout(self, a: int, b: int,
                      elo_overrides: Dict[int, float] = None) -> int:
        """Simulate knockout match; draw → ET → penalties."""
        hg, ag = self._sim_match(a, b, elo_overrides)
        if hg != ag:
            return a if hg > ag else b
        # Extra time — keep H2H bias
        elo_overrides = elo_overrides or {}
        rh = elo_overrides.get(a, self.elo.get_rating(a))
        ra = elo_overrides.get(b, self.elo.get_rating(b))
        adj = self._h2h_adj(a, b)
        lam, mu = elo_to_lambdas(rh + adj, ra - adj)
        et_h = np.random.poisson(lam * 0.28)
        et_a = np.random.poisson(mu * 0.28)
        if et_h != et_a:
            return a if et_h > et_a else b
        # Penalties: slight edge to stronger team, capped
        p_pen = 0.5 + (rh - ra) / 12000
        p_pen = max(0.35, min(0.65, p_pen))
        return a if np.random.random() < p_pen else b

    def simulate_once(self, fixed_results: Dict[int, Tuple[int, int]] = None,
                      elo_overrides: Dict[int, float] = None) -> Dict:
        fixed = fixed_results or {}
        overrides = elo_overrides or {}

        # ── Group Stage ──────────────────────────────────────────
        standings = {g: GroupStandings(teams) for g, teams in self.groups.items()}

        for m in self.matches:
            if m["stage"] != "GROUP_STAGE":
                continue
            grp = m.get("group")
            if not grp:
                continue
            home = m["homeTeam"].get("id")
            away = m["awayTeam"].get("id")
            if not home or not away:
                continue
            if m["id"] in fixed:
                hg, ag = fixed[m["id"]]
            else:
                hg, ag = self._sim_match(home, away, overrides)
            standings[grp].add_result(home, away, hg, ag)

        # ── Determine qualifiers ─────────────────────────────────
        group_order = sorted(standings.keys())  # GROUP_A … GROUP_L
        qualified = {}      # {group: [rank1, rank2, rank3, rank4]}
        thirds = []         # [{team, group, record}]

        for g in group_order:
            ranked = standings[g].ranked()
            qualified[g] = ranked
            third = ranked[2]
            thirds.append({
                "team": third,
                "group": g,
                "record": standings[g].third_record(third),
            })

        # Best 8 thirds by: points, GD, GF
        thirds.sort(key=lambda x: x["record"], reverse=True)
        best8 = [t["team"] for t in thirds[:8]]

        # ── Build Round-of-32 bracket ────────────────────────────
        # 12 groups A-L; winners pair with runners-up from offset groups.
        # Winners A-F vs Runners-up G-L; Winners G-L vs Runners-up A-F;
        # Best-8 thirds face each other (4 matches).
        n = len(group_order)   # 12
        half = n // 2          # 6
        r32 = []

        for i in range(half):          # 0-5 → A-F winners vs G-L runners-up
            w = qualified[group_order[i]][0]
            ru = qualified[group_order[i + half]][1]
            r32.append((w, ru))
        for i in range(half, n):       # 6-11 → G-L winners vs A-F runners-up
            w = qualified[group_order[i]][0]
            ru = qualified[group_order[i - half]][1]
            r32.append((w, ru))
        for i in range(0, 8, 2):       # 4 matches among best-8 thirds
            r32.append((best8[i], best8[i + 1]))

        # ── Simulate knockout rounds ──────────────────────────────
        def play_round(pairs):
            return [self._sim_knockout(a, b, overrides) for a, b in pairs]

        def to_pairs(lst):
            return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]

        r32_w = play_round(r32)              # 16 winners
        r16_w = play_round(to_pairs(r32_w))  # 8 winners
        qf_w  = play_round(to_pairs(r16_w))  # 4 winners
        sf_w  = play_round(to_pairs(qf_w))   # 2 winners
        champion = self._sim_knockout(sf_w[0], sf_w[1], overrides)

        # ── Progression tracking ─────────────────────────────────
        prog: Dict[int, int] = {t: 0 for t in self.teams}
        # 0=group only; 1=r32; 2=r16; 3=qf; 4=sf; 5=final; 6=champion
        for g in group_order:
            for rank, tid in enumerate(qualified[g][:2]):
                prog[tid] = max(prog[tid], 1)
        for tid in best8:
            prog[tid] = max(prog[tid], 1)
        for tid in r32_w:  prog[tid] = max(prog[tid], 2)
        for tid in r16_w:  prog[tid] = max(prog[tid], 3)
        for tid in qf_w:   prog[tid] = max(prog[tid], 4)
        for tid in sf_w:   prog[tid] = max(prog[tid], 5)
        prog[champion] = 6

        return {
            "champion": champion,
            "group_standings": {g: standings[g].ranked() for g in group_order},
            "progression": prog,
        }

    def run_simulations(self, n: int,
                        fixed_results: Dict[int, Tuple[int, int]] = None,
                        elo_overrides: Dict[int, float] = None) -> Dict[int, Dict]:
        fixed = fixed_results or {}
        overrides = elo_overrides or {}

        counts = defaultdict(lambda: [0] * 7)
        # index: 0=group_only, 1=r32, 2=r16, 3=qf, 4=sf, 5=final, 6=champion
        # counts[tid][lvl] = number of sims in which the team's deepest stage was exactly `lvl`.

        for _ in range(n):
            result = self.simulate_once(fixed, overrides)
            for tid, lvl in result["progression"].items():
                counts[tid][lvl] += 1

        probs = {}
        for tid in self.teams:
            c = counts[tid]
            # Cumulative "reached at least stage X" = sum of exact counts at >= X.
            cum = [0.0] * 7
            running = 0
            for i in range(6, -1, -1):
                running += c[i]
                cum[i] = running / n
            probs[tid] = {
                "group_qualify": cum[1],
                "r16":           cum[2],
                "qf":            cum[3],
                "sf":            cum[4],
                "final":         cum[5],
                "win":           cum[6],
            }
        return probs
