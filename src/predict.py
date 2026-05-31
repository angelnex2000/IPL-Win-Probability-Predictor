from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, normalize_team_name


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ipl_win_probability.joblib"


def overs_to_balls(overs: float) -> int:
    completed_overs = int(overs)
    balls = int(round((overs - completed_overs) * 10))
    if balls < 0 or balls > 5:
        raise ValueError("Overs must use cricket notation, for example 12.4 means 12 overs and 4 balls.")
    return completed_overs * 6 + balls


def make_feature_row(
    batting_team: str,
    bowling_team: str,
    city: str,
    venue: str,
    target: int,
    score: int,
    overs: float,
    wickets: int,
) -> pd.DataFrame:
    balls_bowled = overs_to_balls(overs)
    balls_left = max(120 - balls_bowled, 0)
    runs_left = max(target - score, 0)
    wickets_remaining = max(10 - wickets, 0)
    current_run_rate = score / (balls_bowled / 6) if balls_bowled else 0.0
    required_run_rate = runs_left / (balls_left / 6) if balls_left else 0.0

    row = {
        "batting_team": normalize_team_name(batting_team),
        "bowling_team": normalize_team_name(bowling_team),
        "city": city,
        "venue": venue,
        "target": target,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_remaining": wickets_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def load_model(model_path: str | Path = MODEL_PATH):
    model_bundle = joblib.load(model_path)
    return model_bundle


def predict_win_probability(model_bundle: dict, feature_row: pd.DataFrame) -> dict[str, float]:
    model = model_bundle["model"]
    probabilities = model.predict_proba(feature_row)[0]
    class_to_probability = dict(zip(model.classes_, probabilities))
    batting_probability = float(class_to_probability.get(1, 0.0))
    bowling_probability = 1.0 - batting_probability
    return {
        "batting_team_win_probability": round(batting_probability * 100, 2),
        "bowling_team_win_probability": round(bowling_probability * 100, 2),
    }
