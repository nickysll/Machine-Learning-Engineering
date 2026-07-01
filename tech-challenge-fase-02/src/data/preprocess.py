from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw/movielens_100k")
PROCESSED_DATA_DIR = Path("data/processed")

RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "interactions.csv"


def load_ratings() -> pd.DataFrame:
    """Load user-item interactions from raw ratings file."""
    return pd.read_csv(RATINGS_FILE)


def load_movies() -> pd.DataFrame:
    """Load movie metadata from raw movies file."""
    return pd.read_csv(MOVIES_FILE)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings data and create binary target."""
    cleaned = ratings.copy()
    cleaned = cleaned.dropna()
    cleaned["target"] = (cleaned["rating"] >= 4).astype(int)
    return cleaned


def merge_movie_metadata(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ratings with movie metadata."""
    return ratings.merge(movies, on="movieId", how="left")


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to project naming convention."""
    return data.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "title": "item_title",
        }
    )


def save_processed_data(data: pd.DataFrame) -> None:
    """Save processed interactions dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)


def preprocess() -> None:
    """Run preprocessing pipeline."""
    ratings = load_ratings()
    movies = load_movies()
    cleaned_ratings = clean_ratings(ratings)
    interactions = merge_movie_metadata(cleaned_ratings, movies)
    interactions = rename_columns(interactions)
    save_processed_data(interactions)

    print(f"Processed data saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(interactions)}")


if __name__ == "__main__":
    preprocess()