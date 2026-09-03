"""Rank national squads by the corrected talent metric and check the result is sane.

The original metric had two visible faults: it ignored central midfielders, and it
treated a good season in a weak league as equal to one in a strong league. This
prints the ranking after both fixes, then checks three things that should follow
from them: Portugal above Colombia, Mexico dropping below both, and squads built
on top-five-league players near the top. Writes reports/squad_ranking.md."""
from pathlib import Path
import pandas as pd
from src.config import load_params
from src.features.squad_strength import team_squad_strength, club_league_multiplier

NATIONS = ["Portugal", "Colombia", "Mexico", "France", "England", "Spain",
           "Brazil", "Argentina", "Germany", "Netherlands", "Italy", "Uruguay",
           "Morocco", "Japan", "Senegal", "Croatia", "Belgium", "USA", "Ecuador"]


def main():
    cfg = load_params()
    cutoff = cfg["states"]["live_cutoff_date"]
    months = cfg.get("recency_months", 24)

    squads = pd.read_csv("data/processed/squad_coverage.csv")
    pstats = pd.read_parquet("data/processed/player_stats.parquet")

    # Build league strength map
    all_clubs = squads["club"].dropna().unique()
    league_map = club_league_multiplier(all_clubs)

    # Base (old metric, no league, no midfield)
    base = team_squad_strength(squads, pstats, cutoff, months=months)
    # Corrected (both fixes)
    corrected = team_squad_strength(squads, pstats, cutoff, months=months,
                                    league_strength=league_map,
                                    include_midfield=True)
    # League only (no midfield)
    league_only = team_squad_strength(squads, pstats, cutoff, months=months,
                                      league_strength=league_map)
    # Midfield only (no league)
    mid_only = team_squad_strength(squads, pstats, cutoff, months=months,
                                   include_midfield=True)

    def ranking(scores, nations=None):
        """Sorted list of (team, atk_q, dfc_q, net) for given nations."""
        if nations is None:
            nations = list(scores.keys())
        rows = [(n, scores[n][0], scores[n][1], scores[n][0] + scores[n][1])
                for n in nations if n in scores]
        return sorted(rows, key=lambda r: r[3], reverse=True)

    # Build report
    lines = [
        "# Squad strength: corrected metric",
        "",
        f"As-of: {cutoff} | months={months} | league_strength={len(league_map)} clubs mapped",
        "",
        "## Ranking (league + midfield corrected)",
        "",
        "| Rank | Team | atk_q | dfc_q | net |",
        "|------|------|-------|-------|-----|",
    ]
    corrected_rank = ranking(corrected, NATIONS)
    for i, (team, atk, dfc, net) in enumerate(corrected_rank, 1):
        lines.append(f"| {i} | {team} | {atk:+.3f} | {dfc:+.3f} | {net:+.3f} |")

    lines += ["", "## Before vs After (key nations)", "",
              "| Team | Base net | Corrected net | Delta |",
              "|------|----------|---------------|-------|"]
    for team in ["Portugal", "Colombia", "Mexico", "France", "England",
                 "Spain", "Brazil", "Argentina", "Morocco", "Japan"]:
        b = base.get(team, (0, 0))
        c = corrected.get(team, (0, 0))
        lines.append(f"| {team} | {b[0]+b[1]:+.3f} | {c[0]+c[1]:+.3f} | {(c[0]+c[1])-(b[0]+b[1]):+.3f} |")

    lines += ["", "## Ablation: which fix does what", "",
              "| Team | Base | +League | +Midfield | Both |",
              "|------|------|---------|-----------|------|"]
    for team in ["Portugal", "Colombia", "Mexico", "France", "England", "Spain"]:
        b = base.get(team, (0, 0))
        lo = league_only.get(team, (0, 0))
        mo = mid_only.get(team, (0, 0))
        co = corrected.get(team, (0, 0))
        lines.append(f"| {team} | {b[0]+b[1]:+.3f} | {lo[0]+lo[1]:+.3f} | {mo[0]+mo[1]:+.3f} | {co[0]+co[1]:+.3f} |")

    # Acceptance checks
    pt_net = corrected.get("Portugal", (0, 0))
    co_net = corrected.get("Colombia", (0, 0))
    mx_net = corrected.get("Mexico", (0, 0))
    fr_net = corrected.get("France", (0, 0))
    es_net = corrected.get("Spain", (0, 0))
    en_net = corrected.get("England", (0, 0))

    lines += ["", "## Acceptance checks", "",
              f"- Portugal ({pt_net[0]+pt_net[1]:+.3f}) > Colombia ({co_net[0]+co_net[1]:+.3f}): "
              f"**{'YES' if pt_net[0]+pt_net[1] > co_net[0]+co_net[1] else 'NO (FAIL)'}**",
              f"- Mexico ({mx_net[0]+mx_net[1]:+.3f}) below Portugal/Colombia: "
              f"**{'YES' if mx_net[0]+mx_net[1] < pt_net[0]+pt_net[1] and mx_net[0]+mx_net[1] < co_net[0]+co_net[1] else 'NO (FAIL)'}**",
              f"- Top-league cores at top (France {fr_net[0]+fr_net[1]:+.3f}, "
              f"Spain {es_net[0]+es_net[1]:+.3f}, England {en_net[0]+en_net[1]:+.3f})",
    ]

    print("\n".join(lines))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/squad_ranking.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
