from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROCESSED_FILE = Path("data/processed/interactions.csv")
FEATURES_DIR = Path("data/features")

TRAIN_FILE = FEATURES_DIR / "train.csv"
TEST_FILE = FEATURES_DIR / "test.csv"
USER_MAPPING_FILE = FEATURES_DIR / "user_mapping.csv"
ITEM_MAPPING_FILE = FEATURES_DIR / "item_mapping.csv"


def load_interactions() -> pd.DataFrame:
    """Load processed interactions dataset."""
    return pd.read_csv(PROCESSED_FILE)


def create_mapping(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Create numeric mapping for an identifier column."""
    unique_values = data[column].drop_duplicates().sort_values()
    return pd.DataFrame({column: unique_values, f"{column}_idx": range(len(unique_values))})


def apply_mappings(
    data: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Apply user and item numeric mappings."""
    data = data.merge(user_mapping, on="user_id", how="left")
    return data.merge(item_mapping, on="item_id", how="left")


def select_model_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Select columns used by recommendation models."""
    return data[["user_id_idx", "item_id_idx", "rating", "target"]]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test datasets."""
    return train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["target"],
    )


def save_outputs(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
) -> None:
    """Save feature engineering outputs."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)
    user_mapping.to_csv(USER_MAPPING_FILE, index=False)
    item_mapping.to_csv(ITEM_MAPPING_FILE, index=False)


def build_features() -> None:
    """Run feature engineering pipeline."""
    interactions = load_interactions()

    user_mapping = create_mapping(interactions, "user_id")
    item_mapping = create_mapping(interactions, "item_id")

    mapped_data = apply_mappings(interactions, user_mapping, item_mapping)
    model_data = select_model_columns(mapped_data)

    train_data, test_data = split_data(model_data)
    save_outputs(train_data, test_data, user_mapping, item_mapping)

    print(f"Train data saved to: {TRAIN_FILE}")
    print(f"Test data saved to: {TEST_FILE}")
    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")


if __name__ == "__main__":
    build_features()