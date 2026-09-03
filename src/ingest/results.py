import pandas as pd
from src.schema import CANON_MATCH_COLS
from src.utils.ids import make_match_id

def load_results(path):
    """Load the martj42 international-results CSV into the canonical schema."""
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["date"])
    df = pd.DataFrame({
        "date": dates,
        "competition": raw["tournament"].astype(str),
        "season": dates.dt.year.astype(str),
        "home_team": raw["home_team"].astype(str),
        "away_team": raw["away_team"].astype(str),
        "home_score": raw["home_score"].astype("Int64"),
        "away_score": raw["away_score"].astype("Int64"),
        "stage": None,
        "neutral": raw["neutral"].astype(bool),
        "source": "results",
    })
    df["match_id"] = [make_match_id(d, h, a)
                      for d, h, a in zip(df["date"], df["home_team"], df["away_team"])]
    return df[CANON_MATCH_COLS]


def align_results_to_fixtures(results, fixtures, tolerance_days=1):
    """Match fetched finished results to OUR fixtures table and emit canonical match
    rows that take the FIXTURE's date/competition/season/stage/neutral, so match_id
    and dates stay consistent with the rest of the repo even when the API's UTC
    kickoff rolls past midnight (a US evening match is 'tomorrow' in UTC).

    Matching: ordered (home, away) pair first, then the swapped pair (scores swapped
    too, defensive against orientation differences), date within `tolerance_days`.
    Returns (matched: CANON_MATCH_COLS frame, unmatched: the input rows that found no
    fixture, report these loudly upstream, never silently drop them)."""
    fx = fixtures.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    matched_rows, unmatched = [], []
    for _, r in results.iterrows():
        rdate = pd.Timestamp(r["date"])
        hs, as_ = r["home_score"], r["away_score"]
        cand = fx[(fx["home_team"] == r["home_team"]) & (fx["away_team"] == r["away_team"])]
        if cand.empty:                                       # orientation flipped?
            cand = fx[(fx["home_team"] == r["away_team"]) & (fx["away_team"] == r["home_team"])]
            hs, as_ = as_, hs
        cand = cand[(cand["date"] - rdate.normalize()).abs() <= pd.Timedelta(days=tolerance_days)]
        if cand.empty:
            unmatched.append(r)
            continue
        f = cand.iloc[0]
        matched_rows.append({
            "match_id": make_match_id(f["date"], f["home_team"], f["away_team"]),
            "date": f["date"], "competition": f["competition"], "season": f["season"],
            "home_team": f["home_team"], "away_team": f["away_team"],
            "home_score": int(hs), "away_score": int(as_),
            "stage": f["stage"], "neutral": bool(f["neutral"]), "source": "apifootball",
        })
    matched = (pd.DataFrame(matched_rows, columns=CANON_MATCH_COLS) if matched_rows
               else pd.DataFrame(columns=CANON_MATCH_COLS))
    unmatched = pd.DataFrame(unmatched) if unmatched else pd.DataFrame(columns=results.columns)
    return matched, unmatched


def append_new_matches(matches, new_rows):
    """Append canonical match rows, deduplicated on match_id (existing rows win).
    Idempotent: re-appending the same results never double-counts. Sorted by date."""
    out = pd.concat([matches, new_rows], ignore_index=True)
    out = out.drop_duplicates("match_id", keep="first")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def mark_fixtures_finished(fixtures, results):
    """Set status='FT' on fixtures whose (home, away) pairing appears in `results`
    (either orientation), played fixtures then drop out of upcoming_fixtures."""
    pairs = set(zip(results["home_team"], results["away_team"]))
    pairs |= {(a, h) for h, a in pairs}
    fx = fixtures.copy()
    played = [(h, a) in pairs for h, a in zip(fx["home_team"], fx["away_team"])]
    fx.loc[played, "status"] = "FT"
    return fx
