"""
场地环境对进球数的调整（高温 + 海拔）—— 2026 世界杯专用侧面数据。

2026 世界杯在美/墨/加 16 座城市、6-7 月举行。两类有实证支持的环境因素会影响
进球水平（且未必充分反映在 Elo / 赔率里）：

  • 高温高湿：降低跑动强度与比赛节奏 → 进球略减。带空调的封闭/可开合顶棚球场
    可基本抵消（多座美国球场如此）。
  • 高海拔：空气稀薄、体能消耗大，整体回合节奏与有效跑动下降 → 进球略减
    （墨西哥城 ~2240m、瓜达拉哈拉 ~1566m 最显著）。

返回一个乘子 factor，作用于 AVG_GOALS_PER_TEAM（两队同乘，只影响“总进球水平”，
不偏向任何一方）。factor<1 表示该场环境压低进球。

数据为公开可查的球场海拔与气候常识，属粗粒度近似；应配合复盘 Brier/LogLoss 回测，
确认确实改善后再加大权重。
"""
from typing import Dict, Tuple

# city key → (altitude_m, goal_factor, note)
# goal_factor 已综合海拔 + 6/7月气候 + 是否封闭/空调球场。
HOST_CITIES: Dict[str, Dict] = {
    # ── 墨西哥（高海拔主导）──────────────────────────────────────
    "mexico_city":  {"alt": 2240, "factor": 0.92, "note": "墨西哥城 海拔2240m，空气稀薄、节奏下降"},
    "guadalajara":  {"alt": 1566, "factor": 0.95, "note": "瓜达拉哈拉 海拔1566m，中等海拔"},
    "monterrey":    {"alt": 540,  "factor": 0.95, "note": "蒙特雷 6/7月酷热高湿，露天"},
    # ── 美国（高温主导；带顶棚/空调者中性）──────────────────────
    "atlanta":      {"alt": 320,  "factor": 1.00, "note": "亚特兰大 奔驰体育场可开合顶棚+空调，环境中性"},
    "dallas":       {"alt": 150,  "factor": 1.00, "note": "达拉斯 AT&T球场封闭空调，抵消酷热"},
    "houston":      {"alt": 30,   "factor": 1.00, "note": "休斯顿 NRG球场封闭空调，抵消湿热"},
    "kansas_city":  {"alt": 270,  "factor": 0.96, "note": "堪萨斯城 露天，仲夏高温"},
    "los_angeles":  {"alt": 30,   "factor": 1.00, "note": "洛杉矶 SoFi半封闭，气候温和"},
    "miami":        {"alt": 2,    "factor": 0.95, "note": "迈阿密 露天，高温高湿"},
    "new_york":     {"alt": 7,    "factor": 0.98, "note": "纽约/新泽西 MetLife露天，夏季偏热"},
    "boston":       {"alt": 30,   "factor": 1.00, "note": "波士顿/福克斯堡 露天，气候温和"},
    "philadelphia": {"alt": 12,   "factor": 0.98, "note": "费城 露天，夏季湿热"},
    "san_francisco":{"alt": 3,    "factor": 1.00, "note": "旧金山湾区 Levi's露天，气候温和"},
    "seattle":      {"alt": 5,    "factor": 1.00, "note": "西雅图 露天，凉爽温和"},
    # ── 加拿大（凉爽，中性）──────────────────────────────────────
    "toronto":      {"alt": 76,   "factor": 1.00, "note": "多伦多 BMO露天，气候温和"},
    "vancouver":    {"alt": 3,    "factor": 1.00, "note": "温哥华 BC Place封闭，气候温和"},
}

# 常见别名 / 中文名 / 球场名 → 标准 key
_ALIASES = {
    "墨西哥城": "mexico_city", "azteca": "mexico_city", "estadio azteca": "mexico_city",
    "mexico": "mexico_city",
    "瓜达拉哈拉": "guadalajara", "akron": "guadalajara",
    "蒙特雷": "monterrey", "bbva": "monterrey",
    "亚特兰大": "atlanta", "mercedes-benz": "atlanta",
    "达拉斯": "dallas", "at&t": "dallas", "arlington": "dallas",
    "休斯顿": "houston", "休斯敦": "houston", "nrg": "houston",
    "堪萨斯城": "kansas_city", "kansas city": "kansas_city", "arrowhead": "kansas_city",
    "洛杉矶": "los_angeles", "los angeles": "los_angeles", "sofi": "los_angeles", "inglewood": "los_angeles",
    "迈阿密": "miami", "hard rock": "miami",
    "纽约": "new_york", "纽约/新泽西": "new_york", "new york": "new_york",
    "new york/new jersey": "new_york", "metlife": "new_york", "east rutherford": "new_york",
    "波士顿": "boston", "foxborough": "boston", "gillette": "boston",
    "费城": "philadelphia", "lincoln financial": "philadelphia",
    "旧金山": "san_francisco", "san francisco": "san_francisco", "bay area": "san_francisco",
    "santa clara": "san_francisco", "levi's": "san_francisco",
    "西雅图": "seattle", "lumen": "seattle",
    "多伦多": "toronto", "bmo": "toronto",
    "温哥华": "vancouver", "bc place": "vancouver",
}

NEUTRAL = (1.0, "环境中性（未指定承办城市）")


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def resolve_city(venue: str) -> str:
    """把任意场地/城市字符串映射到标准 city key，匹配不到返回空串。"""
    v = _normalize(venue)
    if not v:
        return ""
    if v in HOST_CITIES:
        return v
    if v in _ALIASES:
        return _ALIASES[v]
    # 子串模糊匹配（城市名出现在更长的字符串里）
    for alias, key in _ALIASES.items():
        if alias in v:
            return key
    for key in HOST_CITIES:
        if key.replace("_", " ") in v:
            return key
    return ""


def venue_goal_factor(venue: str) -> Tuple[float, str]:
    """返回 (goal_factor, 说明文字)。匹配不到城市时返回中性 (1.0, …)。"""
    key = resolve_city(venue)
    if not key:
        return NEUTRAL
    c = HOST_CITIES[key]
    return c["factor"], c["note"]


def city_options() -> list:
    """供 UI 下拉：[(显示名, key), …]，按 factor 升序（影响大的在前）。"""
    items = sorted(HOST_CITIES.items(), key=lambda kv: kv[1]["factor"])
    return [(c["note"].split(" ")[0], key) for key, c in items]
