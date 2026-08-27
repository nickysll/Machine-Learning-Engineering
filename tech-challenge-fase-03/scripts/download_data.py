from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "TimSchopf/medical_abstracts"
RAW_DATA_DIR = Path("data/raw")


def main() -> None:
    """Download and save the Medical Abstracts dataset locally."""

    print(f"Downloading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDataset loaded successfully.")
    print(dataset)

    for split in dataset.keys():
        output_path = RAW_DATA_DIR / f"{split}.parquet"

        dataframe = dataset[split].to_pandas()
        dataframe.to_parquet(output_path, index=False)

        print(
            f"\nSplit '{split}' saved to: {output_path}"
            f"\nRows: {len(dataframe)}"
            f"\nColumns: {list(dataframe.columns)}"
        )

    print("\nDataset download completed successfully.")


if __name__ == "__main__":
    main()