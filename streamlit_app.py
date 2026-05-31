from __future__ import annotations

import requests
import streamlit as st

from src.predict import MODEL_PATH, load_model, make_feature_row, predict_win_probability


API_URL = "http://127.0.0.1:8000"


@st.cache_resource
def cached_model() -> dict:
    return load_model(MODEL_PATH)


def local_predict(payload: dict) -> dict:
    row = make_feature_row(**payload)
    return predict_win_probability(cached_model(), row)


def api_predict(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


st.set_page_config(page_title="IPL Win Probability", page_icon="IP", layout="wide")

st.title("IPL Win Probability")
st.caption("Random Forest model trained on historical IPL chase data")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.train_model` first.")
    st.stop()

bundle = cached_model()
reference_values = bundle["reference_values"]
teams = reference_values["teams"]
cities = reference_values["cities"]
venues = reference_values["venues"]

left, right = st.columns([1, 1])

with left:
    batting_team = st.selectbox("Batting team", teams, index=0)
    bowling_options = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox("Bowling team", bowling_options, index=0)
    city = st.selectbox("City", cities, index=0)
    venue = st.selectbox("Venue", venues, index=0)

with right:
    target = st.number_input("Target", min_value=1, max_value=300, value=180, step=1)
    score = st.number_input("Current score", min_value=0, max_value=int(target), value=80, step=1)
    overs = st.number_input("Overs completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    wickets = st.slider("Wickets lost", min_value=0, max_value=10, value=3)

payload = {
    "batting_team": batting_team,
    "bowling_team": bowling_team,
    "city": city,
    "venue": venue,
    "target": int(target),
    "score": int(score),
    "overs": float(overs),
    "wickets": int(wickets),
}

if st.button("Predict", type="primary"):
    try:
        result = api_predict(payload) or local_predict(payload)
        batting_prob = result["batting_team_win_probability"]
        bowling_prob = result["bowling_team_win_probability"]

        metric_cols = st.columns(2)
        metric_cols[0].metric(f"{batting_team} win probability", f"{batting_prob:.2f}%")
        metric_cols[1].metric(f"{bowling_team} win probability", f"{bowling_prob:.2f}%")
        st.progress(int(round(batting_prob)))
        st.caption(f"Model validation accuracy: {bundle['accuracy']:.2%}")
    except ValueError as exc:
        st.error(str(exc))
