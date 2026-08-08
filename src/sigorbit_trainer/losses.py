"""Trainer-owned metric-learning objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, scale: float, margin: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        cosine = embeddings @ F.normalize(self.weight, dim=1).T
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels[:, None], 1.0)
        return torch.cos(theta + self.margin * one_hot) * self.scale


def orientation_loss(
    clean_pose: Tensor, rotated_pose: Tensor, angles: Tensor
) -> tuple[Tensor, Tensor]:
    clean_target = torch.zeros_like(clean_pose)
    clean_target[:, 0] = 1.0
    rotated_target = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    clean = 1.0 - (clean_pose * clean_target).sum(dim=1)
    rotated_dot = (rotated_pose * rotated_target).sum(dim=1).clamp(-1.0, 1.0)
    loss = 0.5 * (clean.mean() + (1.0 - rotated_dot).mean())
    mae_degrees = torch.rad2deg(torch.acos(rotated_dot)).mean()
    return loss, mae_degrees


def embedding_consistency(clean: Tensor, rotated: Tensor) -> Tensor:
    clean_unit = F.normalize(clean, dim=1)
    rotated_unit = F.normalize(rotated, dim=1)
    return (1.0 - (clean_unit * rotated_unit).sum(dim=1)).mean()
