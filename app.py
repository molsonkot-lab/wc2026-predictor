"""
FIFA World Cup 2026 — 预测看板（极简版）
胜平负 + 比分 + 冠军，移动端友好。
模型：Elo + Dixon-Coles(进球相关性) + 蒙特卡洛 + 市场赔率锚定 + 玄学(可选)。
"""
import json
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import plotly.express as px

from config import HOST_TEAMS, WC_HOST_ADVANTAGE
from data.fetcher import (
    fetch_wc_matches, fetch_wc_teams, fetch_match_odds, fetch_outright_odds,
    build_odds_map, build_outright_map, get_fixed_results,
)
from data.players import compute_player_adjusted_elo, KEY_PLAYERS
from data import predictions as plog
from models.elo import EloSystem
from models.simulator import TournamentSimulator
from models.predictor import predict_match
from models.calibration import calibrate_elo_to_market
from models.metaphysics import metaphysics_reading, apply_tilt

st.set_page_config(page_title="⚽ 2026世界杯预测", page_icon="⚽",
                   layout="centered", initial_sidebar_state="collapsed")

# ── 移动端紧凑样式 ──────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 720px;}
[data-testid="stMetricValue"] {font-size: 1.4rem;}
h1 {font-size: 1.5rem !important;}
</style>
""", unsafe_allow_html=True)


# ── 数据加载（缓存）─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_base():
    return fetch_wc_matches(), fetch_wc_teams()


@st.cache_data(ttl=3600, show_spinner=False)
def load_odds():
    return fetch_match_odds(), fetch_outright_odds()


@st.cache_data(ttl=600, show_spinner=False)
def calibrate(_sim, _team_map, out_json, base_json, fixed_json, weight):
    out_map = json.loads(out_json)
    base    = {int(k): v        for k, v in json.loads(base_json).items()}
    fixed   = {int(k): tuple(v) for k, v in json.loads(fixed_json).items()}
    return calibrate_elo_to_market(_sim, out_map, _team_map,
                                   market_weight=weight, base_overrides=base,
                                   fixed_results=fixed)


@st.cache_data(ttl=600, show_spinner=False)
def run_sims(_sim, n, fixed_json, override_json):
    fixed    = {int(k): tuple(v) for k, v in json.loads(fixed_json).items()}
    override = {int(k): v        for k, v in json.loads(override_json).items()}
    return _sim.run_simulations(n, fixed, override)


# ── 侧边栏控制 ─────────────────────────────────────────────────
st.sidebar.title("⚙️ 设置")
n_sims = st.sidebar.select_slider("模拟次数", options=[5_000, 10_000, 20_000],
                                  value=10_000, help="越高越精确，越慢")
market_weight = st.sidebar.slider(
    "市场锚定强度", 0.0, 1.0, 0.7, 0.1,
    help="0=纯模型；1=完全贴合博彩夺冠赔率。推荐0.7。")
mystic_strength = st.sidebar.slider(
    "🔮 玄学影响力", 0.0, 1.0, 0.0, 0.1,
    help="默认0=不影响真实预测。调大才让五行/风水/大运对概率施加偏置（纯娱乐）。")
if st.sidebar.button("🔄 刷新实时数据", use_container_width=True):
    fetch_wc_matches(force_refresh=True)
    fetch_match_odds(force_refresh=True)
    fetch_outright_odds(force_refresh=True)
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"更新: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

# ── 加载与建模 ─────────────────────────────────────────────────
with st.spinner("加载数据..."):
    matches, teams = load_base()
    match_odds_raw, outright_raw = load_odds()

team_map = {t["id"]: t for t in teams}
odds_map = build_odds_map(match_odds_raw)
out_map  = build_outright_map(outright_raw)
fixed    = get_fixed_results(matches)

# Elo（含已结束比赛的更新）
elo = EloSystem()
elo.initialize(teams)
for m in sorted(matches, key=lambda x: x["utcDate"]):
    if m["status"] == "FINISHED" and m["id"] in fixed:
        hg, ag = fixed[m["id"]]
        elo.update(m["homeTeam"]["id"], m["awayTeam"]["id"], hg, ag, m.get("stage", "GROUP_STAGE"))

# 球员伤停调整
player_statuses = {p["name"]: p["status"] for ps in KEY_PLAYERS.values() for p in ps}
base_overrides = {}
for tid, t in team_map.items():
    adj = compute_player_adjusted_elo(elo.get_rating(tid), t.get("tla", ""), player_statuses)
    if abs(adj - elo.get_rating(tid)) > 0.5:
        base_overrides[tid] = adj

sim = TournamentSimulator(matches, teams, elo)
fixed_json = json.dumps({str(k): list(v) for k, v in fixed.items()})

# 市场锚定 → 校准后的 Elo
with st.spinner("市场赔率校准中..."):
    if out_map and market_weight > 0:
        overrides, cal_info = calibrate(
            sim, team_map, json.dumps(out_map),
            json.dumps({str(k): v for k, v in base_overrides.items()}),
            fixed_json, market_weight)
    else:
        overrides, cal_info = base_overrides, {"status": "skipped"}

with st.spinner(f"运行 {n_sims:,} 次赛程模拟..."):
    probs = run_sims(sim, n_sims, fixed_json,
                     json.dumps({str(k): v for k, v in overrides.items()}))

st.title("⚽ 2026 世界杯预测")

tab_champ, tab_match, tab_review = st.tabs(["🏆 冠军", "⚽ 比赛预测", "📊 复盘"])

# ════════════════════════ 冠军预测 ════════════════════════
with tab_champ:
    champ_rows = []
    for tid, t in team_map.items():
        p = probs.get(tid, {})
        champ_rows.append({
            "球队": t.get("name", "?"),
            "夺冠%": round(p.get("win", 0) * 100, 1),
            "出线%": round(p.get("group_qualify", 0) * 100, 0),
        })
    cdf = pd.DataFrame(champ_rows).sort_values("夺冠%", ascending=False).reset_index(drop=True)
    top = cdf.head(12)

    fig = px.bar(top[::-1], x="夺冠%", y="球队", orientation="h",
                 text="夺冠%", color="夺冠%", color_continuous_scale="YlOrRd")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=440, margin=dict(l=0, r=10, t=10, b=0),
                      coloraxis_showscale=False, yaxis_title="", xaxis_title="夺冠概率 %")
    st.plotly_chart(fig, use_container_width=True)

    champ = cdf.iloc[0]
    st.success(f"🥇 夺冠热门：**{champ['球队']}**（{champ['夺冠%']}%）")
    if market_weight > 0 and cal_info.get("status") == "ok":
        st.caption(f"已用博彩赔率校准（锚定强度 {market_weight:.1f}，覆盖 {cal_info.get('resolved',0)} 队）")
    with st.expander("完整夺冠榜"):
        st.dataframe(cdf, use_container_width=True, hide_index=True, height=400)

# ════════════════════════ 比赛预测 ════════════════════════
with tab_match:
    def _host_adv(tla):
        return WC_HOST_ADVANTAGE if tla in HOST_TEAMS else 0.0

    upcoming = [m for m in matches if m["status"] != "FINISHED"
                and m["homeTeam"].get("id") and m["awayTeam"].get("id")]
    upcoming.sort(key=lambda x: x["utcDate"])

    if not upcoming:
        st.info("暂无待预测的比赛。")
    else:
        labels = []
        for m in upcoming[:60]:
            d = m["utcDate"][:10]
            labels.append(f"{d} | {m['homeTeam'].get('name','?')} vs {m['awayTeam'].get('name','?')}")
        sel = st.selectbox("选择比赛", labels)
        m = upcoming[labels.index(sel)]

        ht, at = m["homeTeam"], m["awayTeam"]
        h_name, a_name = ht.get("name", "?"), at.get("name", "?")
        odds_pr = odds_map.get(frozenset([h_name, a_name]))
        res = predict_match(
            home_id=ht["id"], away_id=at["id"], home_name=h_name, away_name=a_name,
            home_tla=ht.get("tla", ""), away_tla=at.get("tla", ""),
            elo=elo, player_statuses=player_statuses, odds_probs=odds_pr,
            home_adv_elo=_host_adv(ht.get("tla", "")),
        )

        # 玄学 tilt（默认 strength=0 时完全不改）
        reading = metaphysics_reading(h_name, a_name, m.get("venue", ""), m["utcDate"][:10])
        p_h, p_d, p_a = apply_tilt(res["p_home"], res["p_draw"], res["p_away"],
                                   reading["bias"], mystic_strength)

        # 记录预测（仅首次，供日后复盘）
        plog.log_prediction(m["id"], h_name, a_name, p_h, p_d, p_a,
                            res["predicted_home"], res["predicted_away"], m["utcDate"])

        st.markdown(f"### {h_name} vs {a_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{h_name} 胜", f"{p_h*100:.0f}%")
        c2.metric("平局", f"{p_d*100:.0f}%")
        c3.metric(f"{a_name} 胜", f"{p_a*100:.0f}%")

        st.metric("🎯 预测比分", f"{res['predicted_home']} : {res['predicted_away']}")
        st.caption(f"预期进球 xG：{res['xg_home']:.2f} – {res['xg_away']:.2f}"
                   + ("　|　含博彩赔率" if odds_pr else ""))

        if mystic_strength > 0:
            st.info(reading["verdict"])
        else:
            with st.expander("🔮 玄学一览（不影响上方数字）"):
                st.write(reading["verdict"])

# ════════════════════════ 复盘 ════════════════════════
with tab_review:
    st.caption("已结束比赛 vs 赛前预测的对比，用真实结果校验模型准确度。")
    reviews = plog.reconcile(fixed, team_map)
    s = plog.summary(reviews)
    if not s:
        st.info("还没有可复盘的比赛。比赛结束后会自动出现胜负/比分命中率。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("胜负命中率", f"{s['outcome_acc']*100:.0f}%", help=f"共 {s['n']} 场")
        c2.metric("比分命中率", f"{s['score_acc']*100:.0f}%")
        c3.metric("Brier", f"{s['avg_brier']:.3f}", help="越低越好")
        c4.metric("LogLoss", f"{s['avg_logloss']:.3f}", help="越低越好")
        st.markdown("---")
        for r in reviews:
            hit = "✅" if r["outcome_hit"] else "❌"
            sc = "🎯" if r["score_hit"] else ""
            probs_str = f"{r['p_home']*100:.0f}/{r['p_draw']*100:.0f}/{r['p_away']*100:.0f}"
            st.markdown(
                f"{hit} **{r['home_name']} {r['actual_home']}–{r['actual_away']} {r['away_name']}** {sc}　"
                f"<small>预测比分 {r['pred_home']}:{r['pred_away']}　胜平负 {probs_str}%</small>",
                unsafe_allow_html=True)
