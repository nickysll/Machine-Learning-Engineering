import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

TEST_PATH = Path("data/raw/test.parquet")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"


def main() -> None:
    print("Loading test dataset and model...")

    test_df = pd.read_parquet(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    x_test = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()