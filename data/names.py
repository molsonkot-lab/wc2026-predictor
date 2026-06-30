"""
球队中文名映射。按 TLA（三字代码）匹配，稳定且与 football-data.org 一致。
显示层用中文；内部逻辑（赔率匹配等）仍用 API 英文名。
"""

ZH_NAMES = {
    "FRA": "法国", "ESP": "西班牙", "ENG": "英格兰", "BRA": "巴西",
    "ARG": "阿根廷", "POR": "葡萄牙", "NED": "荷兰", "GER": "德国",
    "BEL": "比利时", "COL": "哥伦比亚", "MAR": "摩洛哥", "SUI": "瑞士",
    "URY": "乌拉圭", "CRO": "克罗地亚", "USA": "美国", "NOR": "挪威",
    "JPN": "日本", "TUR": "土耳其", "SEN": "塞内加尔", "MEX": "墨西哥",
    "ECU": "厄瓜多尔", "AUT": "奥地利", "CAN": "加拿大", "SWE": "瑞典",
    "KOR": "韩国", "CZE": "捷克", "ALG": "阿尔及利亚", "GHA": "加纳",
    "AUS": "澳大利亚", "CIV": "科特迪瓦", "TUN": "突尼斯", "EGY": "埃及",
    "RSA": "南非", "BIH": "波黑", "SCO": "苏格兰", "KSA": "沙特阿拉伯",
    "IRN": "伊朗", "IRQ": "伊拉克", "COD": "刚果(金)", "QAT": "卡塔尔",
    "CPV": "佛得角", "PAR": "巴拉圭", "UZB": "乌兹别克斯坦", "JOR": "约旦",
    "NZL": "新西兰", "HAI": "海地", "PAN": "巴拿马", "CUW": "库拉索",
}


def zh_name(team: dict) -> str:
    """返回球队中文名；TLA 未命中时退回 API 英文名。"""
    tla = team.get("tla", "") if isinstance(team, dict) else ""
    if tla in ZH_NAMES:
        return ZH_NAMES[tla]
    return (team.get("name") if isinstance(team, dict) else "") or "?"
