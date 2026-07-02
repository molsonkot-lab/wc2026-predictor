"""
比赛 → 承办城市自动匹配。

football-data.org 的 /matches 接口 venue 字段全部为 null（实测 104 场皆然），
但 FIFA 官方赛程里每个场次的球场是固定的（与对阵无关，按场次编号定死）。
本表由官方赛程/多家媒体核对整理（2026-07-02），淘汰赛全部场次已覆盖：

  数据来源：FIFA.com 赛程、Wikipedia 2026 FIFA World Cup knockout stage、
  Sky Sports 逐日赛程（UK时间换算UTC核对）、各场票务/媒体报道交叉验证。

主键用 football-data.org 的 match_id（稳定）；utcDate 作为兜底键，
防止 API 换 ID。city key 与 models/conditions.py 的 HOST_CITIES 一致，
同时喂给高温/海拔进球修正和玄学场地风水。
"""

# match_id → conditions.py 的 city key
VENUE_BY_MATCH_ID = {
    # ── 1/16 决赛 (LAST_32, 6.28–7.3) ────────────────────────────
    537424: "dallas",          # 科特迪瓦 vs 挪威        AT&T Stadium, Arlington
    537416: "new_york",        # 法国 vs 瑞典            MetLife, East Rutherford
    537425: "mexico_city",     # 墨西哥 vs 厄瓜多尔      Estadio Azteca
    537426: "atlanta",         # 英格兰 vs 刚果(金)      Mercedes-Benz Stadium
    537422: "seattle",         # 比利时 vs 塞内加尔      Lumen Field
    537421: "san_francisco",   # 美国 vs 波黑            Levi's Stadium, Santa Clara
    537420: "los_angeles",     # 西班牙 vs 奥地利        SoFi Stadium, Inglewood
    537419: "toronto",         # 葡萄牙 vs 克罗地亚      BMO Field
    537429: "vancouver",       # 瑞士 vs 阿尔及利亚      BC Place
    537428: "dallas",          # 澳大利亚 vs 埃及        AT&T Stadium, Arlington
    537427: "miami",           # 阿根廷 vs 佛得角        Hard Rock Stadium
    537430: "kansas_city",     # 哥伦比亚 vs 加纳        Arrowhead Stadium
    # ── 1/8 决赛 (LAST_16, 7.4–7.7；场馆按官方赛程场次编号固定) ──
    537376: "houston",         # M90 加拿大 vs 摩洛哥    NRG Stadium
    537375: "philadelphia",    # M89                     Lincoln Financial Field
    537377: "new_york",        # M91 巴西 vs …           MetLife
    537378: "mexico_city",     # M92 墨西哥 vs 英格兰    Estadio Azteca
    537379: "dallas",          # M93                     AT&T Stadium, Arlington
    537380: "seattle",         # M94 美国 vs 比利时      Lumen Field
    537381: "atlanta",         # M95                     Mercedes-Benz Stadium
    537382: "vancouver",       # M96                     BC Place
    # ── 1/4 决赛 (7.9–7.11) ─────────────────────────────────────
    537383: "boston",          # M97 QF1                 Gillette, Foxborough
    537384: "los_angeles",     # M98 QF2                 SoFi Stadium
    537385: "miami",           # M99 QF3                 Hard Rock Stadium
    537386: "kansas_city",     # M100 QF4                Arrowhead Stadium
    # ── 半决赛 / 季军战 / 决赛 ──────────────────────────────────
    537387: "dallas",          # M101 SF1 7.14           AT&T Stadium, Arlington
    537388: "atlanta",         # M102 SF2 7.15           Mercedes-Benz Stadium
    537389: "miami",           # M103 季军战 7.18        Hard Rock Stadium
    537390: "new_york",        # M104 决赛 7.19          MetLife, East Rutherford
}

# 兜底键：utcDate → city key（剩余赛程各场开球时间互不相同，可唯一定位）
VENUE_BY_UTC = {
    "2026-06-30T17:00:00Z": "dallas",
    "2026-06-30T21:00:00Z": "new_york",
    "2026-07-01T01:00:00Z": "mexico_city",
    "2026-07-01T16:00:00Z": "atlanta",
    "2026-07-01T20:00:00Z": "seattle",
    "2026-07-02T00:00:00Z": "san_francisco",
    "2026-07-02T19:00:00Z": "los_angeles",
    "2026-07-02T23:00:00Z": "toronto",
    "2026-07-03T03:00:00Z": "vancouver",
    "2026-07-03T18:00:00Z": "dallas",
    "2026-07-03T22:00:00Z": "miami",
    "2026-07-04T01:30:00Z": "kansas_city",
    "2026-07-04T17:00:00Z": "houston",
    "2026-07-04T21:00:00Z": "philadelphia",
    "2026-07-05T20:00:00Z": "new_york",
    "2026-07-06T00:00:00Z": "mexico_city",
    "2026-07-06T19:00:00Z": "dallas",
    "2026-07-07T00:00:00Z": "seattle",
    "2026-07-07T16:00:00Z": "atlanta",
    "2026-07-07T20:00:00Z": "vancouver",
    "2026-07-09T20:00:00Z": "boston",
    "2026-07-10T19:00:00Z": "los_angeles",
    "2026-07-11T21:00:00Z": "miami",
    "2026-07-12T01:00:00Z": "kansas_city",
    "2026-07-14T19:00:00Z": "dallas",
    "2026-07-15T19:00:00Z": "atlanta",
    "2026-07-18T21:00:00Z": "miami",
    "2026-07-19T19:00:00Z": "new_york",
}


def venue_city_key(match: dict) -> str:
    """按官方赛程返回比赛承办城市 key；匹配不到返回空串（UI 退回手动选择）。"""
    key = VENUE_BY_MATCH_ID.get(match.get("id"))
    if key:
        return key
    return VENUE_BY_UTC.get(match.get("utcDate", ""), "")
