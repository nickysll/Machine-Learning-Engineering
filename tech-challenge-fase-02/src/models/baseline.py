from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluation.metrics import compute_classification_metrics

TRAIN_FILE = Path("data/features/train.csv")
TEST_FILE = Path("data/features/test.csv")
REPORTS_DIR = Path("reports")
BASELINE_REPORT_FILE = REPORTS_DIR / "baseline_metrics.json"

FEATURE_COLUMNS = ["user_id_idx", "item_id_idx"]
TARGET_COLUMN = "target"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a feature dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with features and target.
    """
    return pd.read_csv(file_path)


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataset into feature matrix and target vector.

    Args:
        data: DataFrame containing model features and target.

    Returns:
        Feature matrix and target vector.
    """
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    return features, target


def build_dummy_baseline() -> DummyClassifier:
    """Create a dummy baseline classifier.

    Returns:
        DummyClassifier using the most frequent class strategy.
    """
    return DummyClassifier(strategy="most_frequent")


def build_logistic_baseline() -> Pipeline:
    """Create a logistic regression baseline pipeline.

    Returns:
        Scikit-Learn pipeline with one-hot encoding and logistic regression.
    """
    return Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train and evaluate a baseline model.

    Args:
        model_name: Name used to identify the model.
        model: Scikit-Learn compatible estimator.
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary with model name and evaluation metrics.
    """
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    return {
        "model": model_name,
        "metrics": metrics,
    }


def save_report(results: list[dict[str, Any]]) -> None:
    """Save baseline evaluation results as a JSON file.

    Args:
        results: List with model names and evaluation metrics.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    BASELINE_REPORT_FILE.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def run_baselines() -> None:
    """Run baseline models and save evaluation metrics."""
    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)

    x_train, y_train = split_features_and_target(train_data)
    x_test, y_test = split_features_and_target(test_data)

    results = [
        evaluate_model(
            model_name="dummy_most_frequent",
            model=build_dummy_baseline(),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        ),
        evaluate_model(
            model_name="logistic_regression",
            model=build_logistic_baseline(),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        ),
    ]

    save_report(results)

    print(f"Baseline metrics saved to: {BASELINE_REPORT_FILE}")


if __name__ == "__main__":
    run_baselines()
