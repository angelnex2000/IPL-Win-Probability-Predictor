from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TEAM_REPLACEMENTS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}


DEFAULT_MATCHES_PATH = Path(
    r"C:\Users\Angel\Desktop\ML Project\IPL Win Probability project\dataset\matches.csv"
)
DEFAULT_DELIVERIES_PATH = Path(
    r"C:\Users\Angel\Desktop\ML Project\IPL Win Probability project\dataset\deliveries.csv"
)


GROUP_COLUMN = "match_id"

FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "city",
    "venue",
    "target",
    "runs_left",
    "balls_left",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
]

CATEGORICAL_COLUMNS = ["batting_team", "bowling_team", "city", "venue"]
NUMERIC_COLUMNS = [
    "target",
    "runs_left",
    "balls_left",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
]


def normalize_team_name(team: object) -> object:
    if pd.isna(team):
        return team
    return TEAM_REPLACEMENTS.get(str(team), str(team))


def _normalize_team_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(normalize_team_name)
    return df


def load_raw_data(
    matches_path: str | Path = DEFAULT_MATCHES_PATH,
    deliveries_path: str | Path = DEFAULT_DELIVERIES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    matches = _normalize_team_columns(matches, ["team1", "team2", "toss_winner", "winner"])
    deliveries = _normalize_team_columns(deliveries, ["batting_team", "bowling_team"])
    return matches, deliveries


def build_training_frame(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    clean_matches = matches[
        (matches["dl_applied"].fillna(0).astype(int) == 0)
        & (matches["result"].fillna("") == "normal")
        & matches["winner"].notna()
    ][["id", "city", "venue", "winner"]].copy()

    first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id", as_index=False)["total_runs"]
        .sum()
        .rename(columns={"match_id": "id", "total_runs": "first_innings_runs"})
    )
    match_targets = clean_matches.merge(first_innings, on="id", how="inner")
    match_targets["target"] = match_targets["first_innings_runs"] + 1

    chase = deliveries[deliveries["inning"] == 2].copy()
    chase = chase.merge(
        match_targets[["id", "city", "venue", "winner", "target"]],
        left_on="match_id",
        right_on="id",
        how="inner",
    )

    chase["legal_delivery"] = ((chase["wide_runs"] == 0) & (chase["noball_runs"] == 0)).astype(int)
    chase["balls_bowled"] = chase.groupby("match_id")["legal_delivery"].cumsum()
    chase["current_score"] = chase.groupby("match_id")["total_runs"].cumsum()
    chase["is_wicket"] = chase["player_dismissed"].notna().astype(int)
    chase["wickets_lost"] = chase.groupby("match_id")["is_wicket"].cumsum()

    chase["runs_left"] = chase["target"] - chase["current_score"]
    chase["balls_left"] = 120 - chase["balls_bowled"]
    chase["wickets_remaining"] = 10 - chase["wickets_lost"]

    overs_completed = chase["balls_bowled"] / 6
    chase["current_run_rate"] = np.where(overs_completed > 0, chase["current_score"] / overs_completed, 0.0)
    overs_left = chase["balls_left"] / 6
    chase["required_run_rate"] = np.where(overs_left > 0, chase["runs_left"] / overs_left, 0.0)
    chase["result"] = (chase["batting_team"] == chase["winner"]).astype(int)

    training = chase[
        (chase["balls_bowled"] > 0)
        & (chase["balls_left"] >= 0)
        & (chase["wickets_remaining"] >= 0)
        & (chase["runs_left"] >= 0)
    ][[GROUP_COLUMN] + FEATURE_COLUMNS + ["result"]].copy()

    training.replace([np.inf, -np.inf], np.nan, inplace=True)
    training.dropna(subset=FEATURE_COLUMNS + ["result"], inplace=True)
    return training


def get_reference_values(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict[str, list[str]]:
    teams = sorted(
        pd.concat([deliveries["batting_team"], deliveries["bowling_team"]])
        .dropna()
        .map(normalize_team_name)
        .unique()
        .tolist()
    )
    cities = sorted(matches["city"].dropna().unique().tolist())
    venues = sorted(matches["venue"].dropna().unique().tolist())
    return {"teams": teams, "cities": cities, "venues": venues}
