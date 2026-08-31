from pathlib import Path

import pandas as pd

TRAIN_PATH = Path("data/raw/train.parquet")
TEST_PATH = Path("data/raw/test.parquet")

REQUIRED_COLUMNS = {
    "condition_label",
    "medical_abstract",
}

EXPECTED_CLASSES = {1, 2, 3, 4, 5}


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(
            f"{path} does not contain the required columns: "
            f"{REQUIRED_COLUMNS}"
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains missing values.")

    classes = set(df["condition_label"].unique())

    if not classes.issubset(EXPECTED_CLASSES):
        raise ValueError(
            f"{path} contains unexpected classes: {classes}"
        )

    print(f"{path} validated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Classes: {sorted(classes)}")


def main() -> None:
    print("Validating datasets...\n")

    validate_file(TRAIN_PATH)
    validate_file(TEST_PATH)

    print("\nDataset validation completed successfully.")


if __name__ == "__main__":
    main()