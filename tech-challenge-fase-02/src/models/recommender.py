from __future__ import annotations

import torch
from torch import nn


class RecommenderNet(nn.Module):
    """Neural recommender model based on user and item embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize recommendation neural network.

        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            embedding_dim: Size of user and item embeddings.
            hidden_dim: Number of units in hidden layers.
            dropout_rate: Dropout probability used for regularization.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass.

        Args:
            user_ids: Tensor with encoded user identifiers.
            item_ids: Tensor with encoded item identifiers.

        Returns:
            Tensor with raw logits for binary classification.
        """
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)

        features = torch.cat([user_embedding, item_embedding], dim=1)

        return self.network(features).squeeze(1)
