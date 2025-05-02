
#!/usr/bin/env python3
"""
modeling.py
----------
A lightweight predictive model for Los Angeles crime data (2020–present).

Goal
----
Classify whether a reported incident is **violent** (homicide, rape, robbery,
aggravated assault) or **non‑violent** based on temporal and geographic
features.

Why this model?
  • A binary violent/ non‑violent flag is actionable for resource allocation.
  • Features are readily available (no PII, no text parsing beyond a lookup).
  • RandomForest handles non‑linearities and is quick to train.

Output
------
* Prints accuracy and classification report on a held‑out test set.
* Persists the fitted model to `rf_violent_crime.joblib` for reuse.

Usage
-----
    python modeling.py /path/to/Crime_Data_from_2020_to_Present.csv

Or import and call `train_and_evaluate()` from your notebooks.

Dependencies: pandas, scikit‑learn, joblib
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_wrangling import load_and_clean


# --- 1. Target engineering -------------------------------------------------
_VIOLENT_KEYWORDS = {
    "HOMICIDE", "RAPE", "ROBBERY", "AGGRAVATED ASSAULT",
    "SHOTS FIRED", "MANSLAUGHTER", "KIDNAPPING"
}


def _is_violent(desc: str) -> int:
    """Return 1 if description contains a violent keyword."""
    if pd.isna(desc):
        return 0
    desc = str(desc).upper()
    return int(any(word in desc for word in _VIOLENT_KEYWORDS))


# --- 2. Training routine ----------------------------------------------------
def train_and_evaluate(raw_csv: str | Path) -> None:
    """Train RandomForest and report performance."""
    df = load_and_clean(raw_csv)

    # Create target variable
    df["violent"] = df["crm_cd_desc"].apply(_is_violent)

    # Feature engineering: hour, dayofweek, lat, lon
    df["hour"] = df["occurrence_datetime"].dt.hour
    df["dayofweek"] = df["occurrence_datetime"].dt.dayofweek

    features = ["hour", "dayofweek"] + [
        c for c in df.columns if c.startswith("lat") or c.startswith("lon")
    ]
    X = df[features]
    y = df["violent"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Column types
    numeric_features = [col for col in X.columns if X[col].dtype != "object"]
    categorical_features = [col for col in X.columns if X[col].dtype == "object"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("rf", clf)])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # Persist model
    joblib.dump(pipe, "rf_violent_crime.joblib")
    print("✔ Model saved to rf_violent_crime.joblib")


# --- 3. CLI entry point -----------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train violent‑crime classifier.")
    parser.add_argument("raw_csv", help="Path to raw LA crime CSV file")
    args = parser.parse_args()
    train_and_evaluate(args.raw_csv)


if __name__ == "__main__":
    main()
