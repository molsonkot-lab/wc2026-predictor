"""
FIFA World Cup 2026 — Prediction Dashboard
Elo + Dixon-Coles Poisson + Monte Carlo + Player factors + Odds blending
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from datetime import datetime, timezone

from config import N_SIMULATIONS, HOST_TEAMS, WC_HOST_ADVANTAGE
from data.fetcher import (
    fetch_wc_matches, fetch_wc_teams, fetch_match_odds, fetch_outright_odds,
    build_odds_map, build_outright_map, get_fixed_results,
)
from data.players import get_key_players, compute_player_adjusted_elo, KEY_PLAYERS
from models.elo import EloSystem
from models.poisson_model import elo_to_lambdas, match_probabilities, most_likely_score
from models.simulator import TournamentSimulator
from models.predictor import predict_match

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ 2026 世界杯预测",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",   # 手机上默认折叠侧边栏
)

# 移动端适配 CSS
st.markdown("""
<style>
  /* 手机上让表格可横向滚动 */
  .stDataFrame { overflow-x: auto !important; }
  /* 小屏幕减少内边距 */
  @media (max-width: 768px) {
    .block-container { padding: 0.5rem 0.8rem !important; }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1.0rem !important; }
    /* 侧边栏按钮更大，手机更好点 */
    .stButton > button { min-height: 48px; font-size: 1rem; }
    /* tab 字体稍小 */
    .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 6px 8px; }
  }
  /* 进度条颜色 */
  .stProgress > div > div { background-color: #00a651; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_base_data():
    matches = fetch_wc_matches()
    teams   = fetch_wc_teams()
    return matches, teams


@st.cache_data(ttl=3600, show_spinner=False)
def load_odds_data():
    match_odds   = fetch_match_odds()
    outright     = fetch_outright_odds()
    return match_odds, outright


@st.cache_data(ttl=600, show_spinner=False)
def run_sims(_sim, n, fixed_json, override_json):
    fixed    = {int(k): tuple(v) for k, v in json.loads(fixed_json).items()}
    override = {int(k): v       for k, v in json.loads(override_json).items()}
    return _sim.run_simulations(n, fixed, override)


# ─────────────────────────────────────────────────────────────────────────────
# Session state: player status overrides
# ─────────────────────────────────────────────────────────────────────────────

if "player_statuses" not in st.session_state:
    # Initialise from registry defaults
    st.session_state.player_statuses = {}
    for tla, players in KEY_PLAYERS.items():
        for p in players:
            st.session_state.player_statuses[p["name"]] = p["status"]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ WC2026 预测")
st.sidebar.markdown("---")

n_sims = st.sidebar.select_slider(
    "模拟次数", options=[10_000, 20_000, 50_000], value=20_000,
    help="次数越高越精确，但计算越慢"
)

if st.sidebar.button("🔄 刷新实时数据", type="primary", use_container_width=True):
    fetch_wc_matches(force_refresh=True)
    fetch_match_odds(force_refresh=True)
    fetch_outright_odds(force_refresh=True)
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"更新时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("加载赛程与赔率数据..."):
    matches, teams = load_base_data()
    match_odds_raw, outright_raw = load_odds_data()

team_map  = {t["id"]: t for t in teams}
odds_map  = build_odds_map(match_odds_raw)
out_map   = build_outright_map(outright_raw)
fixed     = get_fixed_results(matches)

# Build and update Elo
elo = EloSystem()
elo.initialize(teams)
for m in sorted(matches, key=lambda x: x["utcDate"]):
    if m["status"] == "FINISHED":
        hg, ag = fixed[m["id"]]
        elo.update(m["homeTeam"]["id"], m["awayTeam"]["id"], hg, ag, m.get("stage", "GROUP_STAGE"))

# Build player-adjusted Elo overrides
elo_overrides = {}
for tid, t in team_map.items():
    tla = t.get("tla", "")
    base = elo.get_rating(tid)
    adj  = compute_player_adjusted_elo(base, tla, st.session_state.player_statuses)
    if abs(adj - base) > 0.5:
        elo_overrides[tid] = adj

# Build simulator
sim = TournamentSimulator(matches, teams, elo)

# Run Monte Carlo
with st.spinner(f"运行 {n_sims:,} 次赛程模拟..."):
    probs = run_sims(
        sim, n_sims,
        json.dumps({str(k): list(v) for k, v in fixed.items()}),
        json.dumps({str(k): v       for k, v in elo_overrides.items()}),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Build master dataframe
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for tid, t in team_map.items():
    p   = probs.get(tid, {})
    tla = t.get("tla", "")
    elo_val = elo_overrides.get(tid, elo.get_rating(tid))
    out_prob = out_map.get(t["name"].lower())
    rows.append({
        "team_id":      tid,
        "Team":         t["name"],
        "TLA":          tla,
        "Elo":          round(elo_val),
        "冠军概率%":     round(p.get("win", 0)           * 100, 1),
        "进决赛%":       round(p.get("final", 0)         * 100, 1),
        "进四强%":       round(p.get("sf", 0)            * 100, 1),
        "进八强%":       round(p.get("qf", 0)            * 100, 1),
        "进16强%":       round(p.get("r16", 0)           * 100, 1),
        "出线%":         round(p.get("group_qualify", 0) * 100, 1),
        "赔率冠军概率%": round(out_prob * 100, 1) if out_prob else None,
    })

df = pd.DataFrame(rows).sort_values("冠军概率%", ascending=False).reset_index(drop=True)
df.index += 1

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚽ FIFA 世界杯 2026 — 智能预测看板")
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("已进行比赛", len(fixed))
col_h2.metric("剩余比赛", len([m for m in matches if m["status"] != "FINISHED"]))
col_h3.metric("模拟次数", f"{n_sims:,}")
col_h4.metric("当前冠军大热", df.iloc[0]["Team"] if len(df) else "—")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 冠军赔率",
    "📊 小组赛",
    "🎯 比赛预测",
    "📈 赛程路径",
    "👥 球员状态",
    "🔢 完整排名",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: Championship Odds
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🏆 冠军概率（蒙特卡洛 × 次模拟）")

    top20 = df.head(20).copy()

    fig = px.bar(
        top20, x="Team", y="冠军概率%",
        color="冠军概率%",
        color_continuous_scale="Viridis",
        text="冠军概率%",
        title="夺冠概率 Top 20",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=480, xaxis_tickangle=-40, showlegend=False,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Comparison with bookmaker odds
    cmp = df[df["赔率冠军概率%"].notna()][
        ["Team", "冠军概率%", "赔率冠军概率%", "Elo"]
    ].head(16).copy()
    if not cmp.empty:
        st.subheader("模型 vs 赌盘对比")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="模型预测", x=cmp["Team"], y=cmp["冠军概率%"],
                              marker_color="steelblue"))
        fig2.add_trace(go.Bar(name="赌盘赔率", x=cmp["Team"], y=cmp["赔率冠军概率%"],
                              marker_color="tomato"))
        fig2.update_layout(barmode="group", height=400, xaxis_tickangle=-40,
                           title="模型 vs 赌盘冠军概率比较 (%)")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2: Group Stage
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 小组赛 — 当前积分榜 & 晋级概率")

    groups = sim.groups
    group_letters = sorted(groups.keys())

    for row_start in range(0, len(group_letters), 3):
        cols = st.columns(3)
        for ci, g in enumerate(group_letters[row_start:row_start + 3]):
            with cols[ci]:
                letter = g.replace("GROUP_", "")
                st.markdown(f"#### 第 {letter} 组")
                g_rows = []
                for tid in groups[g]:
                    t = team_map.get(tid, {})
                    p = probs.get(tid, {})
                    tla = t.get("tla", "")
                    adj_elo = elo_overrides.get(tid, elo.get_rating(tid))
                    # Count actual played matches
                    gp = sum(
                        1 for m in matches
                        if m["status"] == "FINISHED" and m.get("group") == g
                        and (m["homeTeam"].get("id") == tid or m["awayTeam"].get("id") == tid)
                    )
                    pts_actual = 0
                    for m in matches:
                        if m["status"] != "FINISHED" or m.get("group") != g:
                            continue
                        hid = m["homeTeam"].get("id")
                        aid = m["awayTeam"].get("id")
                        sc = m["score"]["fullTime"]
                        hg, ag = sc.get("home", 0) or 0, sc.get("away", 0) or 0
                        if hid == tid:
                            pts_actual += 3 if hg > ag else (1 if hg == ag else 0)
                        elif aid == tid:
                            pts_actual += 3 if ag > hg else (1 if ag == hg else 0)

                    g_rows.append({
                        "球队":     t.get("shortName", t.get("name", "")),
                        "Elo":      round(adj_elo),
                        "已赛":     gp,
                        "积分":     pts_actual,
                        "出线%":    f"{p.get('group_qualify',0)*100:.0f}%",
                    })

                g_rows.sort(key=lambda x: (-x["积分"], -x["Elo"]))
                gdf = pd.DataFrame(g_rows)
                st.dataframe(gdf, use_container_width=True, hide_index=True, height=180)

# ═══════════════════════════════════════════════════════════════════
# TAB 3: Match Predictions
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 比赛预测与分析")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_played = st.checkbox("显示已完赛", value=False)
    with col_f2:
        stage_opts = ["全部", "GROUP_STAGE", "ROUND_OF_32", "ROUND_OF_16",
                      "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
        stage_filter = st.selectbox("阶段筛选", stage_opts)
    with col_f3:
        days_ahead = st.slider("显示未来几天比赛", 1, 7, 3)

    now = datetime.now(timezone.utc)
    pred_rows = []
    for m in sorted(matches, key=lambda x: x["utcDate"]):
        if m["status"] == "FINISHED" and not show_played:
            continue
        if stage_filter != "全部" and m.get("stage") != stage_filter:
            continue

        home_id = m["homeTeam"].get("id")
        away_id = m["awayTeam"].get("id")
        if not home_id or not away_id:
            continue

        ht = team_map.get(home_id, {})
        at = team_map.get(away_id, {})
        home_tla = ht.get("tla", "")
        away_tla = at.get("tla", "")

        adj_h = elo_overrides.get(home_id, elo.get_rating(home_id))
        adj_a = elo_overrides.get(away_id, elo.get_rating(away_id))

        host_adv = WC_HOST_ADVANTAGE if home_tla in HOST_TEAMS else 0.0
        lam, mu = elo_to_lambdas(adj_h, adj_a, host_adv)

        # Match odds
        name_key = frozenset([ht.get("name", ""), at.get("name", "")])
        odds_pr = None
        for ok, ov in odds_map.items():
            if ok == name_key:
                odds_pr = ov
                break
        # Also try short name
        short_key = frozenset([ht.get("shortName", ""), at.get("shortName", "")])
        if odds_pr is None:
            for ok, ov in odds_map.items():
                if ok == short_key:
                    odds_pr = ov
                    break

        p_h, p_d, p_a = match_probabilities(lam, mu)
        if odds_pr:
            from models.poisson_model import blend_with_odds
            p_h, p_d, p_a = blend_with_odds((p_h, p_d, p_a), odds_pr)

        ml_h, ml_a = most_likely_score(lam, mu)
        date_str = m["utcDate"][:10]
        stage_label = m.get("stage", "").replace("_", " ").title()
        group_label = m.get("group", "").replace("GROUP_", "组 ") if m.get("group") else stage_label

        result_str = ""
        outcome_icon = ""
        if m["status"] == "FINISHED":
            sc = m["score"]["fullTime"]
            result_str = f"{sc.get('home',0)} – {sc.get('away',0)}"

        pred_rows.append({
            "日期":       date_str,
            "阶段":       group_label,
            "主队":       ht.get("shortName", ht.get("name", "?")),
            "主队胜%":    f"{p_h*100:.0f}%",
            "平局%":      f"{p_d*100:.0f}%",
            "客队胜%":    f"{p_a*100:.0f}%",
            "客队":       at.get("shortName", at.get("name", "?")),
            "预测比分":   f"{ml_h}:{ml_a}",
            "实际比分":   result_str,
            "xG主/客":    f"{lam:.2f}/{mu:.2f}",
            "有赔率":     "✅" if odds_pr else "—",
            "_home_id":   home_id,
            "_away_id":   away_id,
            "_home_tla":  home_tla,
            "_away_tla":  away_tla,
            "_home_name": ht.get("name", ""),
            "_away_name": at.get("name", ""),
            "_odds_pr":   odds_pr,
            "_adj_h":     adj_h,
            "_adj_a":     adj_a,
            "_host_adv":  host_adv,
        })

    display_cols = ["日期","阶段","主队","主队胜%","平局%","客队胜%","客队","预测比分","实际比分","xG主/客","有赔率"]
    pdf_display = pd.DataFrame(pred_rows)[display_cols] if pred_rows else pd.DataFrame()

    if not pdf_display.empty:
        st.dataframe(pdf_display, use_container_width=True, height=520, hide_index=True)
    else:
        st.info("没有符合条件的比赛。")

    # ── Detailed prediction for a single match ──
    st.markdown("---")
    st.subheader("🔍 单场深度分析")

    upcoming = [r for r in pred_rows if not r["实际比分"]]
    if upcoming:
        match_labels = [f"{r['日期']} | {r['主队']} vs {r['客队']} [{r['阶段']}]" for r in upcoming]
        sel = st.selectbox("选择比赛", match_labels)
        sel_idx = match_labels.index(sel)
        r = upcoming[sel_idx]

        result = predict_match(
            home_id=r["_home_id"], away_id=r["_away_id"],
            home_name=r["_home_name"], away_name=r["_away_name"],
            home_tla=r["_home_tla"], away_tla=r["_away_tla"],
            elo=elo,
            player_statuses=st.session_state.player_statuses,
            odds_probs=r["_odds_pr"],
            home_adv_elo=r["_host_adv"],
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"{r['_home_name']} 胜", f"{result['p_home']*100:.1f}%")
        col_b.metric("平局", f"{result['p_draw']*100:.1f}%")
        col_c.metric(f"{r['_away_name']} 胜", f"{result['p_away']*100:.1f}%")

        col_d, col_e = st.columns(2)
        col_d.metric("预期比分 (xG)", f"{result['xg_home']:.2f} – {result['xg_away']:.2f}")
        col_e.metric("最大概率比分", f"{result['predicted_home']} – {result['predicted_away']}")

        with st.expander("📋 完整预测分析", expanded=True):
            st.markdown(result["reasoning"])

# ═══════════════════════════════════════════════════════════════════
# TAB 4: Tournament Path
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 各队赛程晋级路径概率")

    stage_labels = ["出线%", "进16强%", "进八强%", "进四强%", "进决赛%", "冠军概率%"]
    stage_keys   = ["group_qualify", "r16", "qf", "sf", "final", "win"]

    top_n = st.slider("显示前 N 支球队", 4, 16, 8)
    top_teams = df.head(top_n)

    fig_path = go.Figure()
    for _, row in top_teams.iterrows():
        tid = row["team_id"]
        p   = probs.get(tid, {})
        y_vals = [p.get(k, 0) * 100 for k in stage_keys]
        fig_path.add_trace(go.Scatter(
            x=stage_labels, y=y_vals,
            mode="lines+markers",
            name=row["Team"],
            line=dict(width=2),
        ))

    fig_path.update_layout(
        title="各阶段晋级概率（%）",
        yaxis_title="概率 (%)",
        height=480,
        legend=dict(orientation="v"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_path, use_container_width=True)

    # Heatmap
    st.subheader("热力图 — Top 20 队伍 × 各阶段")
    top20_ids = df.head(20)["team_id"].tolist()
    top20_names = df.head(20)["Team"].tolist()
    heat_data = []
    for tid in top20_ids:
        p = probs.get(tid, {})
        heat_data.append([round(p.get(k, 0) * 100, 1) for k in stage_keys])

    fig_heat = px.imshow(
        heat_data,
        x=stage_labels, y=top20_names,
        color_continuous_scale="YlGn",
        aspect="auto",
        title="晋级概率热力图 (%)",
        text_auto=True,
    )
    fig_heat.update_layout(height=580)
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5: Player Status
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("👥 核心球员状态管理")
    st.info("修改球员状态后，预测将自动重新计算。伤号、停赛球员会降低对应队伍的 Elo 评分。")

    tla_options = sorted(KEY_PLAYERS.keys())
    team_names_by_tla = {t.get("tla", ""): t["name"] for t in teams}

    sel_tla = st.selectbox(
        "选择球队",
        tla_options,
        format_func=lambda x: f"{x} — {team_names_by_tla.get(x, x)}"
    )

    players = get_key_players(sel_tla)
    if players:
        st.markdown(f"**{team_names_by_tla.get(sel_tla, sel_tla)} 核心球员**")
        col_hdr1, col_hdr2, col_hdr3, col_hdr4 = st.columns([3, 2, 2, 4])
        col_hdr1.markdown("**球员**")
        col_hdr2.markdown("**位置**")
        col_hdr3.markdown("**影响力 (Elo)**")
        col_hdr4.markdown("**状态**")

        changed = False
        for p in players:
            name = p["name"]
            c1, c2, c3, c4 = st.columns([3, 2, 2, 4])
            c1.write(name)
            c2.write(p["position"])
            c3.write(f"-{p['impact']} pts")
            current = st.session_state.player_statuses.get(name, p["status"])
            new_status = c4.radio(
                f"_{name}",
                ["FIT", "DOUBTFUL", "OUT"],
                index=["FIT", "DOUBTFUL", "OUT"].index(current),
                horizontal=True,
                label_visibility="collapsed",
            )
            if new_status != st.session_state.player_statuses.get(name):
                st.session_state.player_statuses[name] = new_status
                changed = True

        if changed:
            st.cache_data.clear()
            st.rerun()

        # Show team Elo impact
        base_elo = elo.get_rating(
            next((t["id"] for t in teams if t.get("tla") == sel_tla), 0)
        )
        adj_elo = compute_player_adjusted_elo(base_elo, sel_tla, st.session_state.player_statuses)
        st.markdown("---")
        colx, coly = st.columns(2)
        colx.metric("基础 Elo", f"{base_elo:.0f}")
        coly.metric("球员调整后 Elo", f"{adj_elo:.0f}", delta=f"{adj_elo - base_elo:.0f}")
    else:
        st.info(f"{team_names_by_tla.get(sel_tla, sel_tla)} 暂无核心球员数据（可在 data/players.py 中添加）")

    # Quick overview of all OUT players
    out_players = [
        (name, status)
        for name, status in st.session_state.player_statuses.items()
        if status in ("OUT", "DOUBTFUL")
    ]
    if out_players:
        st.markdown("---")
        st.subheader("🚨 当前伤病/停赛名单")
        for name, status in sorted(out_players):
            icon = "❌" if status == "OUT" else "⚠️"
            st.write(f"{icon} **{name}** — {status}")

# ═══════════════════════════════════════════════════════════════════
# TAB 6: Full Rankings
# ═══════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔢 全部 48 队完整排名")

    display_df = df[[
        "Team", "TLA", "Elo", "冠军概率%", "进决赛%",
        "进四强%", "进八强%", "进16强%", "出线%", "赔率冠军概率%"
    ]].copy()

    def colour_pct(val):
        if isinstance(val, (int, float)) and val > 0:
            alpha = min(val / 40, 0.85)
            return f"background-color: rgba(0,140,0,{alpha:.2f}); color: white"
        return ""

    pct_cols = ["冠军概率%", "进决赛%", "进四强%", "进八强%", "进16强%", "出线%"]
    styled = (
        display_df.style
        .applymap(colour_pct, subset=pct_cols)
        .format({c: "{:.1f}%" for c in pct_cols + ["赔率冠军概率%"]},
                na_rep="—")
        .format({"Elo": "{:.0f}"})
    )
    st.dataframe(styled, use_container_width=True, height=1400)
