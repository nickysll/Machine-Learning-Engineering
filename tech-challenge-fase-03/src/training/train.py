from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/raw/train.parquet")
TEST_PATH = Path("data/raw/test.parquet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test datasets."""

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    required_columns = {TEXT_COLUMN, TARGET_COLUMN}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            f"Training dataset must contain columns: {required_columns}"
        )

    if not required_columns.issubset(test_df.columns):
        raise ValueError(
            f"Test dataset must contain columns: {required_columns}"
        )

    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Create the text classification pipeline."""

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def main() -> None:
    print("Loading dataset...")

    train_df, test_df = load_data()

    x_train = train_df[TEXT_COLUMN]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    print("\nTraining model...")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    print("Training completed.")

    print("\nEvaluating model...")

    predictions = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()