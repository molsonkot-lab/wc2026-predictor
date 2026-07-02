"""
玄学加成层（娱乐功能）—— v2「有据可依」版。

⚠️ 重要：风水、五行八卦、干支历法对真实比赛结果**没有任何统计学预测效力**。
本模块纯属趣味。只有当用户把「玄学影响力」滑块调到 >0（默认 0.5）时，
才会对最终概率施加可控偏置——这是用户自己的选择。

v2 的"玄学依据"从 md5 哈希升级为可解释的传统规则（全部确定性，无随机噪声）：
  1. 队伍五行 —— 按地理方位定盘（真实堪舆逻辑）：
       欧洲居西属金 · 亚洲居东属木 · 非洲炎南属火
       南美川泽属水 · 大洋洲环海属水 · 中北美居中属土（东道主之地）
  2. 比赛日干支 —— 真实干支历（以 1949-10-01 甲子日为锚，60 日一轮回），
     日干五行与队伍五行论生克：生我者吉、克我者凶、比和者稳。
  3. 场地风水 —— 16 座承办城市按山川形胜定五行（墨西哥城高原厚土、
     旧金山金门属金、西雅图多雨属水……），未指定城市则不施加场地偏置。
  4. 八字流日 —— 队名×日期的确定性微调（±0.2），模拟流日起伏。

未收录的队伍退回卦象（哈希）推定，保证任何输入都有结果。
"""
import hashlib
from datetime import date
from typing import Tuple, Dict

# 玄学偏置上限系数：满强度(strength=1)+满bias(±1)时对胜率/xG 的最大相对偏移。
TILT_MAX = 0.40

WU_XING = ["金", "木", "水", "火", "土"]
# 五行相克：key 克 value（被克方处于劣势）
KE = {"金": "木", "木": "土", "土": "水", "水": "火", "火": "金"}
# 五行相生：key 生 value
SHENG = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
BAGUA = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]

# ── 干支历（真实历法，非哈希）────────────────────────────────────
STEMS = "甲乙丙丁戊己庚辛壬癸"
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
STEM_ELEM = ["木", "木", "火", "火", "土", "土", "金", "金", "水", "水"]
BRANCH_ELEM = ["水", "土", "木", "木", "土", "火", "火", "土", "金", "金", "土", "水"]
_GANZHI_EPOCH = date(1949, 10, 1)   # 公认甲子日锚点

# ── 队伍五行：按大洲方位定盘 ─────────────────────────────────────
_CONTINENT_ELEM = {
    "欧洲": ("金", "居西方，西方属金"),
    "亚洲": ("木", "居东方，东方属木"),
    "非洲": ("火", "地处炎南，南方属火"),
    "南美": ("水", "川泽纵横（亚马逊），属水"),
    "大洋洲": ("水", "四海环抱，属水"),
    "中北美": ("土", "本届主场居中，中央属土"),
}
_TEAMS_BY_CONTINENT = {
    "欧洲": [
        "france", "法国", "spain", "西班牙", "england", "英格兰",
        "portugal", "葡萄牙", "netherlands", "荷兰", "germany", "德国",
        "belgium", "比利时", "switzerland", "瑞士", "croatia", "克罗地亚",
        "norway", "挪威", "turkey", "türkiye", "土耳其", "austria", "奥地利",
        "sweden", "瑞典", "czechia", "czech republic", "捷克",
        "bosnia-herzegovina", "bosnia and herzegovina", "波黑",
        "scotland", "苏格兰", "italy", "意大利", "denmark", "丹麦",
        "poland", "波兰", "ukraine", "乌克兰", "wales", "威尔士",
        "serbia", "塞尔维亚", "hungary", "匈牙利", "romania", "罗马尼亚",
        "greece", "希腊", "slovakia", "斯洛伐克", "slovenia", "斯洛文尼亚",
        "republic of ireland", "ireland", "爱尔兰", "albania", "阿尔巴尼亚",
        "north macedonia", "北马其顿", "kosovo", "科索沃",
    ],
    "亚洲": [
        "japan", "日本", "korea republic", "south korea", "韩国",
        "iran", "伊朗", "iraq", "伊拉克", "saudi arabia", "沙特阿拉伯",
        "qatar", "卡塔尔", "uzbekistan", "乌兹别克斯坦", "jordan", "约旦",
        "united arab emirates", "阿联酋", "china", "中国",
        "indonesia", "印度尼西亚", "palestine", "巴勒斯坦",
    ],
    "非洲": [
        "morocco", "摩洛哥", "senegal", "塞内加尔", "algeria", "阿尔及利亚",
        "ghana", "加纳", "ivory coast", "côte d'ivoire", "cote d'ivoire", "科特迪瓦",
        "tunisia", "突尼斯", "egypt", "埃及", "south africa", "南非",
        "congo dr", "dr congo", "刚果(金)", "刚果（金）",
        "cape verde islands", "cape verde", "佛得角",
        "nigeria", "尼日利亚", "cameroon", "喀麦隆", "mali", "马里",
        "burkina faso", "布基纳法索", "gabon", "加蓬",
    ],
    "南美": [
        "brazil", "巴西", "argentina", "阿根廷", "colombia", "哥伦比亚",
        "uruguay", "乌拉圭", "ecuador", "厄瓜多尔", "paraguay", "巴拉圭",
        "chile", "智利", "peru", "秘鲁", "venezuela", "委内瑞拉",
        "bolivia", "玻利维亚",
    ],
    "大洋洲": [
        "australia", "澳大利亚", "new zealand", "新西兰",
    ],
    "中北美": [
        "united states", "usa", "美国", "mexico", "墨西哥", "canada", "加拿大",
        "panama", "巴拿马", "haiti", "海地", "curaçao", "curacao", "库拉索",
        "costa rica", "哥斯达黎加", "jamaica", "牙买加",
        "honduras", "洪都拉斯", "suriname", "苏里南",
    ],
}
_TEAM_CONTINENT = {name: cont
                   for cont, names in _TEAMS_BY_CONTINENT.items()
                   for name in names}

# ── 场地风水：16 座承办城市按山川形胜定五行 ──────────────────────
_CITY_ELEM = {
    "mexico_city":   ("土", "高原厚土（海拔2240m）"),
    "guadalajara":   ("土", "高原之城"),
    "monterrey":     ("金", "山城工业，钢铁之都"),
    "atlanta":       ("木", "森林之城"),
    "dallas":        ("土", "中南平原"),
    "houston":       ("火", "石油炎方"),
    "kansas_city":   ("土", "中部平原腹地"),
    "los_angeles":   ("火", "阳光炎沙之地"),
    "miami":         ("火", "湿热酷暑"),
    "new_york":      ("金", "金融之都"),
    "boston":        ("水", "临海之城"),
    "philadelphia":  ("金", "自由钟鸣，钟鼎属金"),
    "san_francisco": ("金", "金门之地"),
    "seattle":       ("水", "多雨之城"),
    "toronto":       ("水", "五大湖畔"),
    "vancouver":     ("水", "海滨多雨"),
}


def _hash_int(*parts: str) -> int:
    s = "|".join(p.strip().lower() for p in parts if p)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def day_ganzhi(match_date: str) -> Tuple[str, str]:
    """真实干支历：返回 (日干支如'丙午', 日干五行)。解析失败返回 ('', '')。"""
    try:
        y, m, d = (int(x) for x in match_date[:10].split("-"))
        idx = (date(y, m, d) - _GANZHI_EPOCH).days % 60
    except (ValueError, AttributeError):
        return "", ""
    return STEMS[idx % 10] + BRANCHES[idx % 12], STEM_ELEM[idx % 10]


def team_element(name: str) -> Tuple[str, str]:
    """队伍五行：按大洲方位定盘；未收录退回卦象推定。返回 (五行, 依据)。"""
    cont = _TEAM_CONTINENT.get(_norm(name))
    if cont:
        elem, why = _CONTINENT_ELEM[cont]
        return elem, f"{cont}{why}"
    return WU_XING[_hash_int(name, "wuxing") % 5], "未入方位盘，依卦象推定"


def team_bagua(name: str) -> str:
    return BAGUA[_hash_int(name, "bagua") % 8]


def venue_element(venue: str) -> Tuple[str, str]:
    """场地五行：按承办城市山川形胜定盘；未指定/未识别返回 ('', '')。"""
    try:
        from models.conditions import resolve_city
        key = resolve_city(venue)
    except Exception:
        key = ""
    if key in _CITY_ELEM:
        elem, why = _CITY_ELEM[key]
        return elem, why
    return "", ""


def _fortune(name: str, team_elem: str, match_date: str) -> Tuple[str, float, str]:
    """
    当日运势 = 日干五行×队伍五行生克 + 八字流日微调（确定性 ±0.2）。
    返回 (大运标签, score ∈ [-0.8, 0.8], 依据说明)。
    """
    gz, day_elem = day_ganzhi(match_date)
    score, why = 0.0, "日期不明，气运持平"
    if day_elem:
        if SHENG.get(day_elem) == team_elem:
            score, why = 0.6, f"{gz}日{day_elem}气生{team_elem}，得日气相生"
        elif day_elem == team_elem:
            score, why = 0.25, f"{gz}日{day_elem}气比和，同气相扶"
        elif KE.get(team_elem) == day_elem:
            score, why = 0.15, f"{team_elem}克{gz}日{day_elem}气，克日为财"
        elif SHENG.get(team_elem) == day_elem:
            score, why = -0.3, f"{team_elem}生{gz}日{day_elem}气，泄气"
        elif KE.get(day_elem) == team_elem:
            score, why = -0.6, f"{gz}日{day_elem}气克{team_elem}，日气相克"
    # 八字流日微调（队名×日期确定性哈希，模拟流日起伏）
    score += ((_hash_int(name, match_date, "bazi") % 5) - 2) * 0.1
    if score >= 0.7:
        label = "大吉"
    elif score >= 0.4:
        label = "中吉"
    elif score >= 0.15:
        label = "小吉"
    elif score > -0.15:
        label = "平"
    elif score > -0.5:
        label = "小凶"
    else:
        label = "中凶"
    return label, score, why


def metaphysics_reading(home_name: str, away_name: str,
                        venue: str = "", match_date: str = "") -> Dict:
    """
    返回一份玄学批语 + 一个 bias ∈ [-1, 1]：
        bias > 0 → 玄学偏向主队；bias < 0 → 偏向客队。
    bias 仅在 apply_tilt / tilt_xg 中按用户滑块强度生效。
    组成：主客五行生克 ±0.5/±0.2 + 场地风水 ±0.3/±0.15 + 大运差 ×0.4。
    """
    he, he_why = team_element(home_name)
    ae, ae_why = team_element(away_name)
    ve, ve_why = venue_element(venue)

    # 五行相克：主客队元素相克关系
    elem_bias = 0.0
    note_elem = f"主队属{he}（{he_why}），客队属{ae}（{ae_why}）"
    if KE.get(he) == ae:
        elem_bias += 0.5; note_elem += f"；{he}克{ae}，主队压制"
    elif KE.get(ae) == he:
        elem_bias -= 0.5; note_elem += f"；{ae}克{he}，客队反制"
    elif SHENG.get(ae) == he:
        elem_bias -= 0.2; note_elem += f"；{ae}生{he}，客队助势"
    elif SHENG.get(he) == ae:
        elem_bias += 0.2; note_elem += f"；{he}生{ae}，主队泄气"
    else:
        note_elem += "；五行无明显生克"

    # 场地风水：与谁同属/相生者得地利。未指定城市 → 不施加场地偏置。
    venue_bias = 0.0
    if ve:
        note_venue = f"场地属{ve}（{ve_why}）"
        if ve == he:
            venue_bias += 0.3; note_venue += f"，与主队比和得地利"
        elif ve == ae:
            venue_bias -= 0.3; note_venue += f"，与客队比和得地利"
        elif SHENG.get(ve) == he:
            venue_bias += 0.15; note_venue += f"，{ve}生{he}助主队"
        elif SHENG.get(ve) == ae:
            venue_bias -= 0.15; note_venue += f"，{ve}生{ae}助客队"
        else:
            note_venue += "，与两队均无生克"
    else:
        note_venue = "未指定承办城市，场地不论"

    # 大运：真实日干支 × 队伍五行
    gz, _ = day_ganzhi(match_date)
    hf, hs, hf_why = _fortune(home_name, he, match_date)
    af, as_, af_why = _fortune(away_name, ae, match_date)
    fortune_bias = (hs - as_) * 0.4

    bias = max(-1.0, min(1.0, elem_bias + venue_bias + fortune_bias))

    return {
        "home_element": he, "away_element": ae,
        "home_bagua": team_bagua(home_name), "away_bagua": team_bagua(away_name),
        "venue_element": ve,
        "home_fortune": hf, "away_fortune": af,
        "day_ganzhi": gz,
        "bias": bias,
        "verdict": (
            f"📿 {note_elem}。{note_venue}。"
            + (f"是日{gz}。" if gz else "")
            + f"主队大运「{hf}」，客队大运「{af}」。"
            f"综合玄学{'偏主队' if bias > 0.05 else ('偏客队' if bias < -0.05 else '势均力敌')}"
            f"（bias {bias:+.2f}）。"
        ),
        "details": [
            f"五行生克 {elem_bias:+.2f}：{note_elem}",
            f"场地风水 {venue_bias:+.2f}：{note_venue}",
            f"大运流日 {fortune_bias:+.2f}：主队 {hf_why}（{hf}）；客队 {af_why}（{af}）",
        ],
    }


def apply_tilt(p_home: float, p_draw: float, p_away: float,
               bias: float, strength: float) -> Tuple[float, float, float]:
    """
    按玄学 bias 与用户设定 strength（0~1）对胜平负概率施加偏置。
    strength=0 时原样返回——真实统计预测不受任何影响。
    bias>0 抬升主队、压低客队；平局概率按比例缩放。
    """
    if strength <= 0 or bias == 0:
        return p_home, p_draw, p_away
    shift = bias * strength * TILT_MAX   # 最多 ±40% 的相对偏移（满强度+满bias）
    ph = max(1e-4, p_home * (1 + shift))
    pa = max(1e-4, p_away * (1 - shift))
    pd = max(1e-4, p_draw)
    tot = ph + pd + pa
    return ph / tot, pd / tot, pa / tot


def tilt_xg(lam: float, mu: float, bias: float, strength: float) -> Tuple[float, float]:
    """
    按玄学 bias 与 strength 对预期进球(xG)施加偏置，用于让玄学也影响"预测比分"。
    strength=0 时原样返回。bias>0 抬高主队进球、压低客队。
    """
    if strength <= 0 or bias == 0:
        return lam, mu
    f = bias * strength * TILT_MAX
    return max(0.1, lam * (1 + f)), max(0.1, mu * (1 - f))
