"""Compute per-player and per-team metrics from StatsBomb event data.

Produces player_values (event counts weighted into a value per player) and
team_style (action-type shares and mean expected goals per shot). Works directly
against statsbombpy, with no SPADL or socceraction dependency."""
import sys, os
sys.path.insert(0, ".")
import pandas as pd
from statsbombpy import sb
from src.ingest.run import persist_tables
from src.schema import CANON_PLAYER_STAT_COLS, CANON_TEAM_STYLE_COLS

COMPETITIONS = [
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
    (55, 43, "Euros 2020"),
    (55, 282, "Euros 2024"),
    (223, 282, "Copa America 2024"),
]

# Event types we care about
EVENT_METRICS = [
    ("Pass", "passes_p90"),
    ("Carry", "carries_p90"),
    ("Pressure", "pressures_p90"),
    ("Dribble", "dribbles_p90"),
    ("Shot", "shots_p90"),
    ("Ball Recovery", "ball_recoveries_p90"),
    ("Interception", "interceptions_p90"),
    ("Block", "blocks_p90"),
    ("Clearance", "clearances_p90"),
    ("Foul Committed", "fouls_committed_p90"),
    ("Foul Won", "fouls_won_p90"),
    ("Duel", "duels_p90"),
    ("Miscontrol", "miscontrols_p90"),
    ("Dispossessed", "dispossessed_p90"),
]


def fetch_all_data():
    """Fetch all events."""
    all_data = []
    for comp_id, season_id, label in COMPETITIONS:
        print(f"\nFetching: {label} (comp={comp_id}, season={season_id})")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            print(f"  Matches: {len(matches)}")
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for _, m in matches.iterrows():
            try:
                events = sb.events(match_id=m["match_id"])
                all_data.append((m, events))
                if len(all_data) % 50 == 0:
                    print(f"  Fetched {len(all_data)} matches...")
            except Exception as e:
                print(f"  Match {m['match_id']} error: {e}")
                continue

    print(f"\nTotal matches: {len(all_data)}")
    return all_data


def compute_player_values(all_data):
    """Compute per-player per-90 event counts and xG from raw events."""
    print("\nComputing player values...")
    player_stats = {}  # (player, team) -> {metric: sum, minutes: X}

    for m, events in all_data:
        for _, ev in events.iterrows():
            player = ev.get("player")
            team = ev.get("team")
            if pd.isna(player) or pd.isna(team):
                continue

            key = (str(player), str(team))
            if key not in player_stats:
                player_stats[key] = {"minutes": 0}
                for _, metric_name in EVENT_METRICS:
                    player_stats[key][metric_name] = 0
                player_stats[key]["xg"] = 0

            ps = player_stats[key]
            ps["minutes"] = max(ps["minutes"], ev.get("minute", 0))

            ev_type = ev.get("type", "")
            for ev_name, metric_name in EVENT_METRICS:
                if ev_type == ev_name:
                    ps[metric_name] += 1

            # xG from shot events
            if ev_type == "Shot":
                xg = ev.get("shot_statsbomb_xg", 0)
                if not pd.isna(xg):
                    ps["xg"] += float(xg)

    # Convert to rows
    rows = []
    for (player, team), ps in player_stats.items():
        mins = ps["minutes"]
        p90 = max(mins, 90) / 90  # avoid division by zero for very low minutes
        for _, metric_name in EVENT_METRICS:
            val = ps[metric_name] / p90
            if val > 0:
                rows.append({
                    "player": player, "team": team, "season": "2022",
                    "position": None, "metric": metric_name, "value": val,
                    "source": "statsbomb_events",
                })
        xg_p90 = ps["xg"] / p90
        if xg_p90 > 0:
            rows.append({
                "player": player, "team": team, "season": "2022",
                "position": None, "metric": "xg_p90", "value": xg_p90,
                "source": "statsbomb_events",
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df[CANON_PLAYER_STAT_COLS]
    print(f"Player values: {len(df):,} rows, {df['player'].nunique():,} players")
    return df


def compute_team_style(all_data):
    """Compute team-level action-type shares and mean xG per shot."""
    print("\nComputing team style...")
    team_counts = {}  # team -> {event_type: count, total: N, xg: sum, shots: N}

    for m, events in all_data:
        for _, ev in events.iterrows():
            team = ev.get("team")
            if pd.isna(team):
                continue
            team = str(team)
            if team not in team_counts:
                team_counts[team] = {"total": 0, "xg": 0, "shots": 0}
                for ev_name, _ in EVENT_METRICS:
                    team_counts[team][ev_name] = 0

            tc = team_counts[team]
            tc["total"] += 1
            ev_type = ev.get("type", "")
            if ev_type in tc:
                tc[ev_type] += 1
            if ev_type == "Shot":
                tc["shots"] += 1
                xg = ev.get("shot_statsbomb_xg", 0)
                if not pd.isna(xg):
                    tc["xg"] += float(xg)

    rows = []
    for team, tc in team_counts.items():
        total = tc["total"]
        if total < 10:
            continue
        for ev_name, metric_name in EVENT_METRICS:
            share = tc[ev_name] / total
            rows.append({
                "team": team, "season": "2022",
                "metric": f"share_{metric_name.replace('_p90','')}",
                "value": share, "source": "statsbomb_events",
            })
        if tc["shots"] > 0:
            rows.append({
                "team": team, "season": "2022",
                "metric": "mean_xg_per_shot",
                "value": tc["xg"] / tc["shots"],
                "source": "statsbomb_events",
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df[CANON_TEAM_STYLE_COLS]
    print(f"Team style: {len(df):,} rows, {df['team'].nunique():,} teams")
    return df


def main():
    print("=" * 60)
    print("Event-Based Player & Team Metrics Pipeline")
    print("=" * 60)

    all_data = fetch_all_data()
    if not all_data:
        print("No data.")
        return

    player_values = compute_player_values(all_data)
    team_style = compute_team_style(all_data)

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    persist_tables({"player_values": player_values, "team_style": team_style}, out_dir=out_dir)

    print(f"\nDone! Files in {out_dir}/")
    print(f"  player_values: {len(player_values):,} rows")
    print(f"  team_style: {len(team_style):,} rows")


if __name__ == "__main__":
    main()
