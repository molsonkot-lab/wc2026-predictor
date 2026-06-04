"""
Key player registry with impact weights.
impact = Elo points team loses when this player is OUT.
Status: "FIT" | "DOUBTFUL" | "OUT"
"""

# Format: {tla: [{name, position, impact, status, notes}]}
KEY_PLAYERS = {
    "FRA": [
        {"name": "Kylian Mbappé",     "position": "Forward",    "impact": 120, "status": "FIT", "notes": "Captain, primary goal threat"},
        {"name": "Antoine Griezmann", "position": "Attacking MF","impact":  70, "status": "FIT", "notes": "Set pieces, creativity"},
        {"name": "Aurélien Tchouaméni","position": "Defensive MF","impact": 55, "status": "FIT", "notes": "Defensive anchor"},
        {"name": "Mike Maignan",       "position": "Goalkeeper",  "impact": 50, "status": "FIT", "notes": "World-class keeper"},
        {"name": "William Saliba",     "position": "Defence",     "impact": 45, "status": "FIT", "notes": "CB organiser"},
    ],
    "ESP": [
        {"name": "Lamine Yamal",    "position": "Forward",    "impact": 100, "status": "FIT", "notes": "Attacking catalyst, winger"},
        {"name": "Pedri",           "position": "Midfield",   "impact":  85, "status": "FIT", "notes": "Playmaker, vision"},
        {"name": "Rodri",           "position": "Defensive MF","impact": 80, "status": "FIT", "notes": "Ballon d'Or winner, midfield engine"},
        {"name": "Álvaro Morata",   "position": "Forward",    "impact":  55, "status": "FIT", "notes": "Target striker"},
        {"name": "Dani Carvajal",   "position": "Defence",    "impact":  45, "status": "FIT", "notes": "Right-back, experience"},
    ],
    "ENG": [
        {"name": "Jude Bellingham",  "position": "Attacking MF","impact": 110, "status": "FIT", "notes": "Talisman, goals from midfield"},
        {"name": "Bukayo Saka",      "position": "Forward",    "impact":  85, "status": "FIT", "notes": "Right winger, key chance creator"},
        {"name": "Phil Foden",       "position": "Attacking MF","impact":  80, "status": "FIT", "notes": "Creative spark"},
        {"name": "Harry Kane",       "position": "Forward",    "impact":  85, "status": "FIT", "notes": "All-time top scorer, aerial threat"},
        {"name": "Declan Rice",      "position": "Defensive MF","impact": 60, "status": "FIT", "notes": "Engine in midfield"},
    ],
    "BRA": [
        {"name": "Vinícius Jr.",     "position": "Forward",    "impact": 110, "status": "FIT", "notes": "Main dribbler, goals & assists"},
        {"name": "Rodrygo",          "position": "Forward",    "impact":  75, "status": "FIT", "notes": "Second attacker"},
        {"name": "Endrick",          "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Young striker, explosive"},
        {"name": "Marquinhos",       "position": "Defence",    "impact":  60, "status": "FIT", "notes": "CB captain"},
        {"name": "Casemiro",         "position": "Defensive MF","impact": 50, "status": "FIT", "notes": "Defensive shield"},
    ],
    "ARG": [
        {"name": "Lionel Messi",     "position": "Forward",    "impact": 110, "status": "FIT", "notes": "GOAT, free kicks, assists, leadership"},
        {"name": "Julián Álvarez",   "position": "Forward",    "impact":  80, "status": "FIT", "notes": "World Cup winner, press + goals"},
        {"name": "Rodrigo De Paul",  "position": "Midfield",   "impact":  65, "status": "FIT", "notes": "Engine, work rate"},
        {"name": "Enzo Fernández",   "position": "Midfield",   "impact":  60, "status": "FIT", "notes": "Creative CM"},
        {"name": "Emiliano Martínez","position": "Goalkeeper", "impact":  70, "status": "FIT", "notes": "Elite penalty saves, command"},
    ],
    "POR": [
        {"name": "Bernardo Silva",   "position": "Midfield",   "impact":  90, "status": "FIT", "notes": "Creative hub, consistency"},
        {"name": "Bruno Fernandes",  "position": "Attacking MF","impact": 80, "status": "FIT", "notes": "Direct attacks, set pieces"},
        {"name": "Rúben Dias",       "position": "Defence",    "impact":  70, "status": "FIT", "notes": "Defensive rock"},
        {"name": "Rafael Leão",      "position": "Forward",    "impact":  75, "status": "FIT", "notes": "Pace and skill on left wing"},
        {"name": "Cristiano Ronaldo","position": "Forward",    "impact":  65, "status": "FIT", "notes": "Experience + goals in big games"},
    ],
    "NED": [
        {"name": "Virgil van Dijk",   "position": "Defence",    "impact":  90, "status": "FIT", "notes": "Captain, dominant CB"},
        {"name": "Frenkie de Jong",   "position": "Midfield",   "impact":  80, "status": "FIT", "notes": "Carrying ball, passing range"},
        {"name": "Cody Gakpo",        "position": "Forward",    "impact":  75, "status": "FIT", "notes": "Left winger, goals"},
        {"name": "Xavi Simons",       "position": "Attacking MF","impact": 70, "status": "FIT", "notes": "Creativity, energy"},
        {"name": "Memphis Depay",     "position": "Forward",    "impact":  55, "status": "FIT", "notes": "Veteran striker"},
    ],
    "GER": [
        {"name": "Florian Wirtz",     "position": "Attacking MF","impact": 100, "status": "FIT", "notes": "New generation leader, creativity"},
        {"name": "Kai Havertz",       "position": "Forward",    "impact":  75, "status": "FIT", "notes": "Physical forward, finisher"},
        {"name": "Jamal Musiala",     "position": "Attacking MF","impact": 90, "status": "FIT", "notes": "Dribbling, pressing"},
        {"name": "Manuel Neuer",      "position": "Goalkeeper", "impact":  55, "status": "FIT", "notes": "Sweeper-keeper experience"},
        {"name": "Joshua Kimmich",    "position": "Midfield",   "impact":  70, "status": "FIT", "notes": "RB or DM, leadership"},
    ],
    "URY": [
        {"name": "Federico Valverde","position": "Midfield",   "impact":  95, "status": "FIT", "notes": "Box-to-box, captain material"},
        {"name": "Darwin Núñez",     "position": "Forward",    "impact":  85, "status": "FIT", "notes": "Pace + power striker"},
        {"name": "Rodrigo Bentancur","position": "Midfield",   "impact":  60, "status": "FIT", "notes": "Midfield control"},
        {"name": "Ronald Araújo",    "position": "Defence",    "impact":  70, "status": "FIT", "notes": "Physical CB"},
    ],
    "COL": [
        {"name": "James Rodríguez",  "position": "Attacking MF","impact": 85, "status": "FIT", "notes": "Playmaker, set piece magician"},
        {"name": "Luis Díaz",        "position": "Forward",    "impact":  90, "status": "FIT", "notes": "Left winger, pace"},
        {"name": "Falcao García",    "position": "Forward",    "impact":  50, "status": "FIT", "notes": "Veteran striker"},
        {"name": "Jhon Córdoba",     "position": "Forward",    "impact":  55, "status": "FIT", "notes": "Aerial striker"},
    ],
    "MAR": [
        {"name": "Achraf Hakimi",    "position": "Defence",    "impact":  90, "status": "FIT", "notes": "RB/RWB, carrying threat"},
        {"name": "Hakim Ziyech",     "position": "Attacking MF","impact": 80, "status": "FIT", "notes": "Technique, free kicks"},
        {"name": "Youssef En-Nesyri","position": "Forward",    "impact":  75, "status": "FIT", "notes": "Target man, aerial ability"},
        {"name": "Sofyan Amrabat",   "position": "Defensive MF","impact": 70, "status": "FIT", "notes": "Defensive shield, 2022 hero"},
    ],
    "CRO": [
        {"name": "Luka Modrić",      "position": "Midfield",   "impact":  90, "status": "FIT", "notes": "Captain, heart of midfield"},
        {"name": "Ivan Perišić",     "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Winger, experience"},
        {"name": "Bruno Petković",   "position": "Forward",    "impact":  50, "status": "FIT", "notes": "Super-sub mentality"},
        {"name": "Marcelo Brozović", "position": "Midfield",   "impact":  70, "status": "FIT", "notes": "Midfield engine"},
    ],
    "USA": [
        {"name": "Christian Pulisic","position": "Forward",    "impact":  95, "status": "FIT", "notes": "Captain USA, creativity"},
        {"name": "Gio Reyna",        "position": "Attacking MF","impact": 75, "status": "FIT", "notes": "Skillful, dribbling"},
        {"name": "Tyler Adams",      "position": "Defensive MF","impact": 65, "status": "FIT", "notes": "Midfield anchor"},
        {"name": "Folarin Balogun",  "position": "Forward",    "impact":  70, "status": "FIT", "notes": "Goal scorer"},
    ],
    "NOR": [
        {"name": "Erling Haaland",   "position": "Forward",    "impact": 140, "status": "FIT", "notes": "World's best striker, team = him"},
        {"name": "Martin Ødegaard",  "position": "Attacking MF","impact": 90, "status": "FIT", "notes": "Captain, creative hub"},
        {"name": "Alexander Sørloth","position": "Forward",    "impact":  65, "status": "FIT", "notes": "Alternative target man"},
    ],
    "JPN": [
        {"name": "Takefusa Kubo",    "position": "Forward",    "impact":  85, "status": "FIT", "notes": "Dribbling, creativity, La Liga pedigree"},
        {"name": "Wataru Endō",      "position": "Defensive MF","impact": 70, "status": "FIT", "notes": "Pressing engine"},
        {"name": "Ritsu Dōan",       "position": "Midfield",   "impact":  65, "status": "FIT", "notes": "Box-to-box, goals"},
        {"name": "Ayase Ueda",       "position": "Forward",    "impact":  60, "status": "FIT", "notes": "Target striker"},
    ],
    "MEX": [
        {"name": "Santiago Giménez", "position": "Forward",    "impact":  90, "status": "FIT", "notes": "Prolific Eredivisie scorer"},
        {"name": "Edson Álvarez",    "position": "Defensive MF","impact": 70, "status": "FIT", "notes": "Midfield shield"},
        {"name": "Hirving Lozano",   "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Pace on flank"},
    ],
    "SEN": [
        {"name": "Sadio Mané",       "position": "Forward",    "impact":  90, "status": "FIT", "notes": "Captain, veteran forward"},
        {"name": "Idrissa Gueye",    "position": "Midfield",   "impact":  60, "status": "FIT", "notes": "Press, tackle"},
        {"name": "Ismaïla Sarr",     "position": "Forward",    "impact":  70, "status": "FIT", "notes": "Winger, pace"},
    ],
    "CAN": [
        {"name": "Alphonso Davies",  "position": "Defence",    "impact": 100, "status": "FIT", "notes": "LB/LW, pace + crossing"},
        {"name": "Jonathan David",   "position": "Forward",    "impact":  90, "status": "FIT", "notes": "Top scorer, clinical"},
        {"name": "Tajon Buchanan",   "position": "Midfield",   "impact":  60, "status": "FIT", "notes": "Winger"},
    ],
    "BEL": [
        {"name": "Kevin De Bruyne",  "position": "Midfield",   "impact":  95, "status": "FIT", "notes": "Creative genius (aging golden gen)"},
        {"name": "Romelu Lukaku",    "position": "Forward",    "impact":  75, "status": "FIT", "notes": "Physical striker, goals"},
        {"name": "Leandro Trossard", "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Left winger"},
    ],
    "AUT": [
        {"name": "Marcel Sabitzer",  "position": "Midfield",   "impact":  75, "status": "FIT", "notes": "Midfield engine"},
        {"name": "Marko Arnautović", "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Veteran striker"},
        {"name": "David Alaba",      "position": "Defence",    "impact":  80, "status": "FIT", "notes": "CB/LB, leadership"},
    ],
    "ECU": [
        {"name": "Moisés Caicedo",   "position": "Midfield",   "impact":  90, "status": "FIT", "notes": "World-class DM, transfer record"},
        {"name": "Enner Valencia",   "position": "Forward",    "impact":  70, "status": "FIT", "notes": "Veteran captain striker"},
    ],
    "TUR": [
        {"name": "Hakan Çalhanoğlu", "position": "Midfield",   "impact":  90, "status": "FIT", "notes": "DM/CM, penalty taker"},
        {"name": "Arda Güler",       "position": "Attacking MF","impact": 85, "status": "FIT", "notes": "Young talent, Real Madrid"},
        {"name": "Kerem Aktürkoğlu", "position": "Forward",    "impact":  65, "status": "FIT", "notes": "Winger"},
    ],
    "SUI": [
        {"name": "Granit Xhaka",     "position": "Midfield",   "impact":  80, "status": "FIT", "notes": "Captain, midfield general"},
        {"name": "Xherdan Shaqiri",  "position": "Midfield",   "impact":  60, "status": "FIT", "notes": "Big-match player"},
        {"name": "Ruben Vargas",     "position": "Forward",    "impact":  55, "status": "FIT", "notes": "Winger"},
    ],
}

# Default key player list for teams not in registry
DEFAULT_PLAYERS = []


def get_key_players(tla: str) -> list:
    return KEY_PLAYERS.get(tla, DEFAULT_PLAYERS)


def compute_player_adjusted_elo(base_elo: float, tla: str, statuses: dict) -> float:
    """
    Adjust team Elo based on player availability.
    statuses: {player_name: "FIT"|"DOUBTFUL"|"OUT"}
    """
    from config import PLAYER_IMPACT
    players = get_key_players(tla)
    adjustment = 0.0
    for p in players:
        name = p["name"]
        status = statuses.get(name, p["status"])
        factor = PLAYER_IMPACT.get(status, 0.0)
        adjustment += p["impact"] * factor
    return base_elo - adjustment
