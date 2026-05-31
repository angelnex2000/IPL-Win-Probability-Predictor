from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import MODEL_PATH, load_model, make_feature_row, predict_win_probability


app = FastAPI(title="IPL Win Probability API", version="1.0.0")


class PredictionRequest(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    venue: str
    target: int = Field(gt=0, description="Runs needed to win before chase starts.")
    score: int = Field(ge=0, description="Current chasing team score.")
    overs: float = Field(ge=0, le=20, description="Cricket notation, for example 12.4.")
    wickets: int = Field(ge=0, le=10, description="Chasing team wickets lost.")


@lru_cache(maxsize=1)
def model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train_model")
    return load_model(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "IPL Win Probability API is running"}


@app.get("/metadata")
def metadata() -> dict:
    bundle = model_bundle()
    return {
        "accuracy": round(float(bundle["accuracy"]), 4),
        "teams": bundle["reference_values"]["teams"],
        "cities": bundle["reference_values"]["cities"],
        "venues": bundle["reference_values"]["venues"],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    if request.batting_team == request.bowling_team:
        raise HTTPException(status_code=400, detail="Batting team and bowling team must be different.")
    if request.score > request.target:
        raise HTTPException(status_code=400, detail="Score cannot be greater than target.")

    try:
        row = make_feature_row(**request.model_dump())
        probabilities = predict_win_probability(model_bundle(), row)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **probabilities,
        "batting_team": request.batting_team,
        "bowling_team": request.bowling_team,
    }
