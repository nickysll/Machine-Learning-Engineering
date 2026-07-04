import pandas as pd

from data.preprocess import clean_ratings, rename_columns


def test_clean_ratings_creates_binary_target() -> None:
    """Test if clean_ratings creates the expected binary target."""
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [10, 20, 30],
            "rating": [5.0, 3.0, 4.0],
            "timestamp": [111, 222, 333],
        }
    )

    result = clean_ratings(ratings)

    assert "target" in result.columns
    assert result["target"].tolist() == [1, 0, 1]


def test_rename_columns_uses_project_convention() -> None:
    """Test if raw column names are renamed to project conventions."""
    data = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [10],
            "title": ["Movie A"],
        }
    )

    result = rename_columns(data)

    assert "user_id" in result.columns
    assert "item_id" in result.columns
    assert "item_title" in result.columns
