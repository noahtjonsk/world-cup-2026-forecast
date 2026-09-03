import pandas as pd
from src.schema import CANON_FIXTURE_COLS, CANON_LINEUP_COLS, CANON_INJURY_COLS

BASE = "https://v3.football.api-sports.io"

def build_request(endpoint, params, api_key):
    """Build (url, headers, params) for an API-Football v3 call."""
    url = f"{BASE}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    return url, headers, params

def fetch(endpoint, params, api_key):
    """Thin network wrapper (not unit-tested). Returns parsed JSON."""
    import requests
    url, headers, params = build_request(endpoint, params, api_key)
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize_fixtures(payload):
    """Parse the API-Football /fixtures response into the canonical fixtures schema."""
    rows = []
    for item in payload.get("response", []):
        fx, lg, tm = item["fixture"], item["league"], item["teams"]
        rows.append({
            "fixture_id": fx["id"],
            "date": fx["date"],
            "competition": lg["name"],
            "season": str(lg["season"]),
            "stage": lg.get("round"),
            "home_team": tm["home"]["name"],
            "away_team": tm["away"]["name"],
            "status": fx["status"]["short"],
            "venue": (fx.get("venue") or {}).get("name"),
            "neutral": False,
            "source": "apifootball",
        })
    if not rows:
        return pd.DataFrame(columns=CANON_FIXTURE_COLS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df[CANON_FIXTURE_COLS]

def normalize_lineups(payload, fixture_id):
    """Parse /fixtures/lineups into per-player rows with starter/sub flag."""
    rows = []
    for block in payload.get("response", []):
        team = block["team"]["name"]
        formation = block.get("formation")
        for is_starter, group in ((True, "startXI"), (False, "substitutes")):
            for entry in block.get(group, []):
                pl = entry["player"]
                rows.append({
                    "fixture_id": fixture_id,
                    "team": team,
                    "player": pl["name"],
                    "position": pl.get("pos"),
                    "is_starter": is_starter,
                    "formation": formation,
                    "source": "apifootball",
                })
    if not rows:
        return pd.DataFrame(columns=CANON_LINEUP_COLS)
    return pd.DataFrame(rows)[CANON_LINEUP_COLS]

# API-Football team names -> the repo's canonical names (matches/fixtures vocabulary).
# Names not in the map pass through unchanged; the fetch script validates the final
# names against the known 48 and fails loudly on anything unresolved.
APIFOOTBALL_TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

FINISHED = {"FT", "AET", "PEN"}


def normalize_results(payload, name_map=None):
    """Parse an API-Football /fixtures response into FINISHED results:
    [date, home_team, away_team, home_score, away_score, status] with team names
    mapped through the alias table. Kickoff datetimes stay UTC-naive, alignment
    to the repo's local fixture dates happens in results.align_results_to_fixtures."""
    name_map = APIFOOTBALL_TEAM_ALIASES if name_map is None else name_map
    rows = []
    for item in payload.get("response", []):
        fx, tm, goals = item["fixture"], item["teams"], item.get("goals", {})
        status = (fx.get("status") or {}).get("short")
        if status not in FINISHED:
            continue
        rows.append({
            "date": fx["date"],
            "home_team": name_map.get(tm["home"]["name"], tm["home"]["name"]),
            "away_team": name_map.get(tm["away"]["name"], tm["away"]["name"]),
            "home_score": goals.get("home"),
            "away_score": goals.get("away"),
            "status": status,
        })
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "status"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df


def normalize_injuries(payload):
    """Parse /injuries into the canonical injuries schema."""
    rows = []
    for item in payload.get("response", []):
        pl, tm = item["player"], item["team"]
        fx = item.get("fixture") or {}
        rows.append({
            "date": fx.get("date"),
            "team": tm["name"],
            "player": pl["name"],
            "reason": pl.get("reason"),
            "status": pl.get("type"),
            "source": "apifootball",
        })
    if not rows:
        return pd.DataFrame(columns=CANON_INJURY_COLS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    return df[CANON_INJURY_COLS]
