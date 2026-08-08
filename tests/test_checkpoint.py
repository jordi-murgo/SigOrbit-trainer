from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sigorbit_trainer.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from sigorbit_trainer.losses import ArcFace


def test_safetensors_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    head = ArcFace(2, 2, 8.0, 0.2)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    embeddings = F.normalize(model(inputs), dim=1)
    loss = F.cross_entropy(head(embeddings, labels), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = save_checkpoint(
        tmp_path / "checkpoints",
        run_id="test-run",
        stage="backbone",
        epoch_completed=0,
        global_step=1,
        model_kind="backbone",
        model=model,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        config_sha256="a" * 64,
        dataset_sha256="synthetic:" + "b" * 64,
        class_map_sha256="c" * 64,
        architecture_sha256="d" * 64,
        best_score=(50.0, 0.1),
        metrics={"loss": float(loss.detach())},
        mark_best=True,
        device=torch.device("cpu"),
    )
    loaded = load_checkpoint(path)
    assert loaded.metadata.global_step == 1
    assert not list(path.glob("*.pt"))

    restored_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    restored_head = ArcFace(2, 2, 8.0, 0.2)
    restored_optimizer = torch.optim.AdamW(
        list(restored_model.parameters()) + list(restored_head.parameters()), lr=0.01
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
    restore_checkpoint(
        loaded,
        restored_model,
        restored_head,
        restored_optimizer,
        restored_scheduler,
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = save_checkpoint(
        tmp_path / "checkpoints",
        run_id="test-run",
        stage="pose",
        epoch_completed=0,
        global_step=0,
        model_kind="canonicalized",
        model=model,
        head=None,
        optimizer=optimizer,
        scheduler=None,
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        class_map_sha256="c" * 64,
        architecture_sha256="d" * 64,
        best_score=None,
        metrics={},
        mark_best=False,
        device=torch.device("cpu"),
    )
    tensor_path = path / "tensors.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint(path)


def test_symlinked_checkpoint_directory_is_rejected(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = save_checkpoint(
        tmp_path / "checkpoints",
        run_id="test-run",
        stage="pose",
        epoch_completed=0,
        global_step=0,
        model_kind="canonicalized",
        model=model,
        head=None,
        optimizer=optimizer,
        scheduler=None,
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        class_map_sha256="c" * 64,
        architecture_sha256="d" * 64,
        best_score=None,
        metrics={},
        mark_best=False,
        device=torch.device("cpu"),
    )
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        load_checkpoint(alias)
