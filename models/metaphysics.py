"""
玄学加成层（娱乐功能）。

⚠️ 重要：风水、五行八卦、生辰八字、大运对真实比赛结果**没有任何统计学预测效力**。
本模块纯属趣味，默认 tilt=0（完全不影响统计预测数字）。只有当用户主动把
「玄学影响力」滑块调大时，才会对最终概率施加可控偏置——这是用户自己的选择。

所有"玄学"取值都由队名 / 场地 / 日期做**确定性哈希**得到，保证同样输入每次结果一致，
不引入随机噪声。
"""
import hashlib
from typing import Tuple, Dict

WU_XING = ["金", "木", "水", "火", "土"]
# 五行相克：key 克 value（被克方处于劣势）
KE = {"金": "木", "木": "土", "土": "水", "水": "火", "火": "金"}
# 五行相生：key 生 value
SHENG = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
BAGUA = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
DA_YUN = ["大吉", "中吉", "小吉", "平", "小凶", "中凶"]


def _hash_int(*parts: str) -> int:
    s = "|".join(p.strip().lower() for p in parts if p)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def team_element(name: str) -> str:
    return WU_XING[_hash_int(name, "wuxing") % 5]


def team_bagua(name: str) -> str:
    return BAGUA[_hash_int(name, "bagua") % 8]


def team_fortune(name: str, match_date: str = "") -> str:
    """生辰八字×大运：队名 + 比赛日期 → 当日运势。"""
    return DA_YUN[_hash_int(name, match_date, "dayun") % len(DA_YUN)]


def venue_fengshui(venue: str) -> str:
    """场地风水属性（用所属五行表示）。"""
    return WU_XING[_hash_int(venue or "neutral", "venue") % 5]


def _fortune_score(level: str) -> float:
    return {"大吉": 1.0, "中吉": 0.6, "小吉": 0.3, "平": 0.0,
            "小凶": -0.4, "中凶": -0.8}.get(level, 0.0)


def metaphysics_reading(home_name: str, away_name: str,
                        venue: str = "", match_date: str = "") -> Dict:
    """
    返回一份玄学批语 + 一个 bias ∈ [-1, 1]：
        bias > 0 → 玄学偏向主队；bias < 0 → 偏向客队。
    bias 仅在 apply_tilt 中按用户滑块强度生效，默认强度为 0。
    """
    he = team_element(home_name)
    ae = team_element(away_name)
    vf = venue_fengshui(venue)

    # 五行相克：主客队元素相克关系
    elem_bias = 0.0
    note_elem = f"主队属{he}、客队属{ae}"
    if KE.get(he) == ae:
        elem_bias += 0.5; note_elem += f"，{he}克{ae}，主队压制"
    elif KE.get(ae) == he:
        elem_bias -= 0.5; note_elem += f"，{ae}克{he}，客队反制"
    elif SHENG.get(ae) == he:
        elem_bias -= 0.2; note_elem += f"，{ae}生{he}，客队助势"
    elif SHENG.get(he) == ae:
        elem_bias += 0.2; note_elem += f"，{he}生{ae}，主队泄气"
    else:
        note_elem += "，五行无明显生克"

    # 场地风水：与谁同属/相生者得地利
    venue_bias = 0.0
    if vf == he:
        venue_bias += 0.3
    elif vf == ae:
        venue_bias -= 0.3
    elif SHENG.get(vf) == he:
        venue_bias += 0.15
    elif SHENG.get(vf) == ae:
        venue_bias -= 0.15

    # 大运（生辰八字）
    hf = team_fortune(home_name, match_date)
    af = team_fortune(away_name, match_date)
    fortune_bias = (_fortune_score(hf) - _fortune_score(af)) * 0.4

    bias = max(-1.0, min(1.0, elem_bias + venue_bias + fortune_bias))

    return {
        "home_element": he, "away_element": ae,
        "home_bagua": team_bagua(home_name), "away_bagua": team_bagua(away_name),
        "venue_element": vf,
        "home_fortune": hf, "away_fortune": af,
        "bias": bias,
        "verdict": (
            f"📿 {note_elem}；场地属{vf}。"
            f"主队大运「{hf}」，客队大运「{af}」。"
            f"综合玄学{'偏主队' if bias > 0.05 else ('偏客队' if bias < -0.05 else '势均力敌')}。"
        ),
    }


def apply_tilt(p_home: float, p_draw: float, p_away: float,
               bias: float, strength: float) -> Tuple[float, float, float]:
    """
    按玄学 bias 与用户设定 strength（0~1，默认 0）对胜平负概率施加偏置。
    strength=0 时原样返回——真实统计预测不受任何影响。
    bias>0 抬升主队、压低客队；平局概率按比例缩放。
    """
    if strength <= 0 or bias == 0:
        return p_home, p_draw, p_away
    shift = bias * strength * 0.15   # 最多 ±15% 的相对偏移（满强度+满bias）
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
    f = bias * strength * 0.15
    return max(0.1, lam * (1 + f)), max(0.1, mu * (1 - f))
