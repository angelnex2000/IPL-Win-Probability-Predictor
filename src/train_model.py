from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import (
    CATEGORICAL_COLUMNS,
    DEFAULT_DELIVERIES_PATH,
    DEFAULT_MATCHES_PATH,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_training_frame,
    get_reference_values,
    load_raw_data,
)


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ipl_win_probability.joblib"


def train(matches_path: Path, deliveries_path: Path, model_path: Path) -> dict:
    matches, deliveries = load_raw_data(matches_path, deliveries_path)
    training = build_training_frame(matches, deliveries)

    X = training[FEATURE_COLUMNS]
    y = training["result"]
    groups = training["match_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=8,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    bundle = {
        "model": model,
        "accuracy": accuracy,
        "feature_columns": FEATURE_COLUMNS,
        "reference_values": get_reference_values(matches, deliveries),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IPL win probability Random Forest model.")
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--deliveries", type=Path, default=DEFAULT_DELIVERIES_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train(args.matches, args.deliveries, args.model_path)
    print(f"Saved model: {args.model_path}")
    print(f"Validation accuracy: {bundle['accuracy']:.4f}")


if __name__ == "__main__":
    main()
