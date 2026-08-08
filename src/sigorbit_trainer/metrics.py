"""Signer-disjoint retrieval metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .augment import rotate_tensor
from .data import DatasetBundle, SignatureDataset


@dataclass(frozen=True)
class ValidationMetrics:
    top1: float
    top5: float
    median_margin: float
    fragile_percent: float
    count: int

    def score(self, top1_floor: float = 100.0) -> tuple[float, float]:
        """Ranking key for checkpoint selection: higher is better.

        Clamping top-1 at `top1_floor` makes every checkpoint at or above the
        floor compare purely on margin, so a one- or two-image top-1 wobble
        cannot discard a materially better-separated model. Below the floor
        top-1 still dominates, because there accuracy is a real deficit rather
        than sampling noise.
        """
        return min(round(self.top1, 2), top1_floor), self.median_margin

    def to_dict(self) -> dict[str, float | int]:
        return {
            "top1": self.top1,
            "top5": self.top5,
            "median_margin": self.median_margin,
            "fragile_percent": self.fragile_percent,
            "count": self.count,
        }


@torch.no_grad()
def embed_validation(
    model: nn.Module,
    data: DatasetBundle,
    transform: object,
    *,
    input_seed: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[Tensor, list[str]]:
    signer_map = {
        signer: index
        for index, signer in enumerate(sorted({r.signer_id for r in data.validation_records}))
    }
    dataset = SignatureDataset(
        data.source,
        data.validation_records,
        signer_map,
        transform,  # type: ignore[arg-type]
        run_seed=input_seed,
        stage="validation",
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    embeddings: list[Tensor] = []
    model.eval()
    for tensor, _ in loader:
        output = model(tensor.to(device, non_blocking=True))
        if isinstance(output, tuple):
            output = output[0]
        embeddings.append(F.normalize(output.float(), dim=1).cpu())
    return torch.cat(embeddings), [record.signer_id for record in data.validation_records]


def leave_one_out(
    embeddings: Tensor, signer_ids: list[str], fragile_margin: float
) -> ValidationMetrics:
    if embeddings.shape[0] != len(signer_ids):
        raise ValueError("embedding/label length mismatch")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, signer in enumerate(signer_ids):
        groups[signer].append(index)
    signers = sorted(groups)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("leave-one-out requires two samples per validation signer")
    array = F.normalize(embeddings, dim=1).numpy()
    full = array @ array.T
    np.fill_diagonal(full, -np.inf)
    signer_scores = np.stack([full[:, groups[signer]].max(axis=1) for signer in signers], axis=1)
    truth = np.array([signers.index(signer) for signer in signer_ids])
    order = np.argsort(-signer_scores, axis=1)
    ranks = (order == truth[:, None]).argmax(axis=1) + 1
    genuine = signer_scores[np.arange(len(truth)), truth]
    impostor_scores = signer_scores.copy()
    impostor_scores[np.arange(len(truth)), truth] = -np.inf
    margins = genuine - impostor_scores.max(axis=1)
    return ValidationMetrics(
        top1=float((ranks == 1).mean() * 100.0),
        top5=float((ranks <= min(5, len(signers))).mean() * 100.0),
        median_margin=float(np.median(margins)),
        fragile_percent=float((margins < fragile_margin).mean() * 100.0),
        count=len(truth),
    )


@torch.no_grad()
def rotation_retrieval(
    model: nn.Module,
    data: DatasetBundle,
    transform: object,
    angles_degrees: tuple[float, ...],
    *,
    split: str = "validation",
    padding_value: float,
    fragile_margin: float,
    input_seed: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, dict[str, float | int]]:
    """Per-angle leave-one-out retrieval, reported but never used for selection."""
    records = data.source.records(split)
    if not records:
        raise ValueError(f"split has no genuine records: {split}")
    signer_map = {
        signer: index for index, signer in enumerate(sorted({r.signer_id for r in records}))
    }
    dataset = SignatureDataset(
        data.source,
        records,
        signer_map,
        transform,  # type: ignore[arg-type]
        run_seed=input_seed,
        stage=f"rotation-retrieval-{split}",
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    signer_ids = [record.signer_id for record in records]
    result: dict[str, dict[str, float | int]] = {}
    model.eval()
    for degrees in angles_degrees:
        embeddings: list[Tensor] = []
        radians = float(torch.deg2rad(torch.tensor(float(degrees))))
        for tensor, _ in loader:
            tensor = tensor.to(device, non_blocking=True)
            angle = torch.full((tensor.shape[0],), radians, device=device, dtype=tensor.dtype)
            rotated = rotate_tensor(tensor, angle, padding_value)
            output = model(rotated)
            if isinstance(output, tuple):
                output = output[0]
            embeddings.append(F.normalize(output.float(), dim=1).cpu())
        metrics = leave_one_out(torch.cat(embeddings), signer_ids, fragile_margin)
        result[str(float(degrees))] = metrics.to_dict()
    return result


@torch.no_grad()
def rotation_consistency(
    model: nn.Module,
    embeddings: Tensor,
    data: DatasetBundle,
    transform: object,
    angles_degrees: tuple[float, ...],
    *,
    padding_value: float,
    input_seed: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, float]:
    signer_map = {
        signer: index
        for index, signer in enumerate(sorted({r.signer_id for r in data.validation_records}))
    }
    dataset = SignatureDataset(
        data.source,
        data.validation_records,
        signer_map,
        transform,  # type: ignore[arg-type]
        run_seed=input_seed,
        stage="rotation-validation",
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    result: dict[str, float] = {}
    model.eval()
    for degrees in angles_degrees:
        similarities: list[Tensor] = []
        offset = 0
        for tensor, _ in loader:
            tensor = tensor.to(device, non_blocking=True)
            angle = torch.full(
                (tensor.shape[0],),
                torch.deg2rad(torch.tensor(degrees)).item(),
                device=device,
                dtype=tensor.dtype,
            )
            rotated = rotate_tensor(tensor, angle, padding_value)
            output = model(rotated)
            if isinstance(output, tuple):
                output = output[0]
            reference = embeddings[offset : offset + tensor.shape[0]].to(device)
            similarities.append((F.normalize(output.float(), dim=1) * reference).sum(dim=1).cpu())
            offset += tensor.shape[0]
        result[str(float(degrees))] = float(torch.cat(similarities).mean())
    return result
