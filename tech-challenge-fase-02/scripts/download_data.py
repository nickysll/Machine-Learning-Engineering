from pathlib import Path
import shutil

import kagglehub


DATASET_NAME = "abhikjha/movielens-100k"
RAW_DATA_DIR = Path("data/raw/movielens_100k")


def download_dataset() -> None:
    """Download MovieLens 100K dataset from Kaggle and copy it to data/raw."""
    downloaded_path = Path(kagglehub.dataset_download(DATASET_NAME))

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in downloaded_path.iterdir():
        destination = RAW_DATA_DIR / file_path.name

        if file_path.is_file():
            shutil.copy2(file_path, destination)

    print(f"Dataset downloaded to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_dataset()