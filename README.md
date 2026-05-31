# IPL Win Probability

End-to-end IPL win probability project using historical ball-by-ball data:

Historical Data -> Feature Engineering -> Random Forest -> Training -> FastAPI -> Streamlit UI -> Win Probability

## Project Files

- `src/features.py` - cleans IPL data and creates chase-state training rows.
- `src/train_model.py` - trains and saves a Random Forest model.
- `src/predict.py` - shared prediction helper.
- `api/main.py` - FastAPI service.
- `streamlit_app.py` - Streamlit user interface.
- `models/` - generated model output after training.

## Setup

```powershell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train The Model

The defaults point at the CSV files you provided.

```powershell
python -m src.train_model
```

Or pass custom paths:

```powershell
python -m src.train_model --matches "path\to\matches.csv" --deliveries "path\to\deliveries.csv"
```

## Run FastAPI

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

Example request:

```json
{
  "batting_team": "Royal Challengers Bangalore",
  "bowling_team": "Sunrisers Hyderabad",
  "city": "Hyderabad",
  "venue": "Rajiv Gandhi International Stadium, Uppal",
  "target": 208,
  "score": 110,
  "overs": 12.4,
  "wickets": 3
}
```

## Run Streamlit

In another terminal, keep FastAPI running and start:

```powershell
streamlit run streamlit_app.py
```

The UI can call FastAPI, and also falls back to the local model if the API is not running.

## Run With Docker

Build the image:

```powershell
docker build -t ipl-win-probability .
```

Run the Streamlit app:

```powershell
docker run --rm -p 8501:8501 ipl-win-probability
```

Open `http://127.0.0.1:8501`.
