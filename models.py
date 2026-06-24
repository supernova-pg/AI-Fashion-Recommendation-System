from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SiameseConfig:
    input_dim: int = 2816  # 2048 (ResNet50) + 768 (DistilBERT)
    hidden_dim: int = 512
    embedding_dim: int = 256
    dropout_p: float = 0.3
    margin: float = 1.0


class SiameseEncoder(nn.Module):
    """Projection head over frozen multimodal features."""

    def __init__(self, cfg: SiameseConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_p),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class SiameseNetwork(nn.Module):
    def __init__(self, cfg: SiameseConfig) -> None:
        super().__init__()
        self.encoder = SiameseEncoder(cfg)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        dist = F.pairwise_distance(z1, z2, keepdim=False)
        return z1, z2, dist


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss for labels: 1=positive match, 0=negative."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        pos_loss = labels * torch.pow(distances, 2)
        neg_loss = (1.0 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return torch.mean(pos_loss + neg_loss)
