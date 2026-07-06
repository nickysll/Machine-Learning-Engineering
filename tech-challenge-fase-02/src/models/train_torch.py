from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import mlflow
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

TARGET_COLUMN = "target"
EXPERIMENT_NAME = "tech-challenge-recommender"


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training.

    Returns:
        Parsed training arguments.
    """
    parser = argparse.ArgumentParser(description="Train PyTorch recommender model.")
    parser.add_argument("--run-name", type=str, default="recommender_net")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.2)

    return parser.parse_args()


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


def split_train_validation(
    data: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation sets."""
    train_data, validation_data = train_test_split(
        data,
        test_size=0.1,
        random_state=seed,
        stratify=data[TARGET_COLUMN],
    )

    return train_data, validation_data


def create_dataloader(
    data: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader from interaction DataFrame."""
    dataset = InteractionDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
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


def configure_mlflow() -> None:
    """Configure MLflow tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_params(
    args: argparse.Namespace,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> None:
    """Log training parameters to MLflow."""
    mlflow.log_params(
        {
            "model_name": "recommender_net",
            "num_users": num_users,
            "num_items": num_items,
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout_rate": args.dropout_rate,
        }
    )


def log_metrics_to_mlflow(
    metrics: dict[str, float | None],
    prefix: str,
) -> None:
    """Log metrics to MLflow with a prefix."""
    valid_metrics = {
        f"{prefix}_{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
        if metric_value is not None
    }

    mlflow.log_metrics(valid_metrics)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train model using early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
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

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("validation_loss", validation_loss, step=epoch)

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

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break


def run_training() -> None:
    """Train and evaluate PyTorch recommendation model."""
    args = parse_args()
    set_seed(args.seed)
    configure_mlflow()

    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)
    train_data, validation_data = split_train_validation(train_data, args.seed)

    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    validation_loader = create_dataloader(
        validation_data,
        args.batch_size,
        shuffle=False,
    )
    test_loader = create_dataloader(test_data, args.batch_size, shuffle=False)

    num_users, num_items = get_model_sizes([train_data, validation_data, test_data])
    device = get_device()

    with mlflow.start_run(run_name=args.run_name):
        log_training_params(args, num_users, num_items, device)

        model = build_model(
            model_name="recommender_net",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
        ).to(device)

        train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            args=args,
            device=device,
        )

        model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
        loss_fn = nn.BCEWithLogitsLoss()
        test_metrics = evaluate_model(model, test_loader, loss_fn, device)

        final_report = {
            "model": "recommender_net",
            "metrics": test_metrics,
            "parameters": vars(args),
        }

        save_outputs(model=model, metrics=final_report)
        log_metrics_to_mlflow(test_metrics, prefix="test")
        mlflow.log_artifact(str(REPORT_FILE))
        mlflow.log_artifact(str(MODEL_FILE))

    print(f"PyTorch metrics saved to: {REPORT_FILE}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    run_training()
