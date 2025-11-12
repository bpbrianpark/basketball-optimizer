"""Model training workflow skeleton using scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingArtifacts:
    model: Pipeline
    metrics: dict[str, Any]
    model_path: Path


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    """Create a scikit-learn pipeline for shooting classification."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feature_columns),
        ]
    )

    classifier = RandomForestClassifier(n_estimators=200, random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(df: pd.DataFrame, label_column: str, model_dir: Path) -> TrainingArtifacts:
    """Train the classifier and persist the fitted pipeline."""
    feature_columns = [col for col in df.columns if col not in {label_column, "video_id"}]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns], df[label_column], test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(feature_columns)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = classification_report(y_test, predictions, output_dict=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "shooting_form_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    return TrainingArtifacts(model=pipeline, metrics=metrics, model_path=model_path)


def load_model(model_path: Path) -> Pipeline:
    """Load a persisted pipeline from disk."""
    return joblib.load(model_path)
