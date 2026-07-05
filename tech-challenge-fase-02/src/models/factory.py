from __future__ import annotations

from typing import Literal

from torch import nn

from src.models.recommender import RecommenderNet

ModelName = Literal["recommender_net"]


def build_model(
    model_name: ModelName,
    num_users: int,
    num_items: int,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Create a model instance from a model name.

    Args:
        model_name: Name of the model architecture.
        num_users: Number of unique users.
        num_items: Number of unique items.
        embedding_dim: Size of user and item embeddings.
        hidden_dim: Number of units in hidden layers.
        dropout_rate: Dropout probability used for regularization.

    Returns:
        PyTorch model instance.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name == "recommender_net":
        return RecommenderNet(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    raise ValueError(f"Unsupported model name: {model_name}")
