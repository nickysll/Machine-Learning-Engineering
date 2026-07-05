import shutil
from pathlib import Path

import kagglehub

DATASET_NAME = "abhikjha/movielens-100k"
RAW_DATA_DIR = Path("data/raw/movielens_100k")


def clear_raw_data_dir() -> None:
    """Remove previous dataset files from raw data directory."""
    if RAW_DATA_DIR.exists():
        shutil.rmtree(RAW_DATA_DIR)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def copy_dataset_files(source_path: Path) -> None:
    """Copy all dataset files recursively to the raw data directory."""
    files = [path for path in source_path.rglob("*") if path.is_file()]

    if not files:
        raise FileNotFoundError(f"No files found in {source_path}")

    for file_path in files:
        destination = RAW_DATA_DIR / file_path.name
        shutil.copy2(file_path, destination)
        print(f"Copied: {destination}")


def download_dataset() -> None:
    """Download MovieLens 100K dataset from Kaggle."""
    print("Starting dataset download...")

    downloaded_path = Path(kagglehub.dataset_download(DATASET_NAME))

    print(f"Kaggle cache path: {downloaded_path}")

    clear_raw_data_dir()
    copy_dataset_files(downloaded_path)

    print(f"Dataset downloaded to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_dataset()
