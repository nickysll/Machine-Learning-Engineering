import pandas as pd

from data.feature_engineering import (
    apply_mappings,
    create_mapping,
    select_model_columns,
)


def test_create_mapping_creates_sequential_indices() -> None:
    """Test if create_mapping generates sorted sequential indices."""
    data = pd.DataFrame({"user_id": [3, 1, 3, 2]})

    result = create_mapping(data, "user_id")

    assert result["user_id"].tolist() == [1, 2, 3]
    assert result["user_id_idx"].tolist() == [0, 1, 2]


def test_apply_mappings_adds_user_and_item_indices() -> None:
    """Test if user and item mappings are applied to the dataset."""
    data = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "rating": [5.0],
            "target": [1],
        }
    )
    user_mapping = pd.DataFrame({"user_id": [1], "user_id_idx": [0]})
    item_mapping = pd.DataFrame({"item_id": [10], "item_id_idx": [0]})

    result = apply_mappings(data, user_mapping, item_mapping)

    assert "user_id_idx" in result.columns
    assert "item_id_idx" in result.columns
    assert result.loc[0, "user_id_idx"] == 0
    assert result.loc[0, "item_id_idx"] == 0


def test_select_model_columns_returns_expected_columns() -> None:
    """Test if only model input columns are selected."""
    data = pd.DataFrame(
        {
            "user_id_idx": [0],
            "item_id_idx": [1],
            "rating": [5.0],
            "target": [1],
            "item_title": ["Movie A"],
        }
    )

    result = select_model_columns(data)

    assert result.columns.tolist() == [
        "user_id_idx",
        "item_id_idx",
        "rating",
        "target",
    ]
