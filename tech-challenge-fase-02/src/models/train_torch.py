from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import compute_classification_metrics
from src.models.factory import build_model

TRAIN_FILE = Path("data/features/train.csv")
TEST_FILE = Path("data/features/test.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_FILE = MODELS_DIR / "recommender_net.pt"
REPORT_FILE = REPORTS_DIR / "torch_metrics.json"

FEATURE_COLUMNS = ["user_id_idx", "item_id_idx"]
TARGET_COLUMN = "target"

SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 3


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interactions."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize dataset tensors.

        Args:
            data: DataFrame with user, item and target columns.
        """
        self.user_ids = torch.tensor(data["user_id_idx"].values, dtype=torch.long)
        self.item_ids = torch.tensor(data["item_id_idx"].values, dtype=torch.long)
        self.targets = torch.tensor(data[TARGET_COLUMN].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.targets)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one interaction sample."""
        return self.user_ids[index], self.item_ids[index], self.targets[index]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get available training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)


def split_train_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation sets."""
    train_data, validation_data = train_test_split(
        data,
        test_size=0.1,
        random_state=SEED,
        stratify=data[TARGET_COLUMN],
    )

    return train_data, validation_data


def create_dataloader(
    data: pd.DataFrame,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader from interaction DataFrame."""
    dataset = InteractionDataset(data)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def get_model_sizes(dataframes: list[pd.DataFrame]) -> tuple[int, int]:
    """Get number of users and items across datasets."""
    full_data = pd.concat(dataframes, ignore_index=True)
    num_users = int(full_data["user_id_idx"].max()) + 1
    num_items = int(full_data["item_id_idx"].max()) + 1

    return num_users, num_items


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    for user_ids, item_ids, targets in dataloader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(user_ids, item_ids)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    """Collect targets, probabilities and average loss."""
    model.eval()
    targets_list: list[int] = []
    probabilities: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for user_ids, item_ids, targets in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            targets = targets.to(device)

            logits = model(user_ids, item_ids)
            loss = loss_fn(logits, targets)
            probs = torch.sigmoid(logits)

            targets_list.extend(targets.cpu().numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            total_loss += loss.item()

    return targets_list, probabilities, total_loss / len(dataloader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float | None]:
    """Evaluate model and compute classification metrics."""
    y_true, y_proba, loss = collect_predictions(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    y_pred = [int(probability >= 0.5) for probability in y_proba]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    metrics["loss"] = float(loss)

    return metrics


def save_outputs(
    model: nn.Module,
    metrics: dict[str, Any],
) -> None:
    """Save trained model and metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_FILE)

    REPORT_FILE.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def run_training() -> None:
    """Train and evaluate PyTorch recommendation model."""
    set_seed(SEED)

    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)
    train_data, validation_data = split_train_validation(train_data)

    train_loader = create_dataloader(train_data, shuffle=True)
    validation_loader = create_dataloader(validation_data, shuffle=False)
    test_loader = create_dataloader(test_data, shuffle=False)

    num_users, num_items = get_model_sizes([train_data, validation_data, test_data])
    device = get_device()

    model = build_model(
        model_name="recommender_net",
        num_users=num_users,
        num_items=num_items,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        validation_metrics = evaluate_model(
            model=model,
            dataloader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        validation_loss = validation_metrics["loss"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    save_outputs(
        model=model,
        metrics={
            "model": "recommender_net",
            "metrics": test_metrics,
        },
    )

    print(f"PyTorch metrics saved to: {REPORT_FILE}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    run_training()
