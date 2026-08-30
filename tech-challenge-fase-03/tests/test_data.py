from pathlib import Path

import pandas as pd

TRAIN_PATH = Path("data/raw/train.parquet")
TEST_PATH = Path("data/raw/test.parquet")

REQUIRED_COLUMNS = {
    "condition_label",
    "medical_abstract",
}


def test_dataset_files_exist() -> None:
    assert TRAIN_PATH.exists()
    assert TEST_PATH.exists()


def test_train_schema() -> None:
    df = pd.read_parquet(TRAIN_PATH)

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_test_schema() -> None:
    df = pd.read_parquet(TEST_PATH)

    assert REQUIRED_COLUMNS.issubset(df.columns)


def test_train_has_no_missing_values() -> None:
    df = pd.read_parquet(TRAIN_PATH)

    assert df["condition_label"].isna().sum() == 0
    assert df["medical_abstract"].isna().sum() == 0


def test_target_classes() -> None:
    df = pd.read_parquet(TRAIN_PATH)

    assert set(df["condition_label"].unique()) == {1, 2, 3, 4, 5}