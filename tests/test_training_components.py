import torch

from sigorbit_trainer.augment import rotate_tensor
from sigorbit_trainer.losses import embedding_consistency, orientation_loss
from sigorbit_trainer.sampler import PKBatchSampler


def test_pk_sampler_is_epoch_addressable() -> None:
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    first = list(PKBatchSampler(labels, 2, 2, seed=9, stage="backbone", epoch=3))
    repeated = list(PKBatchSampler(labels, 2, 2, seed=9, stage="backbone", epoch=3))
    changed = list(PKBatchSampler(labels, 2, 2, seed=9, stage="backbone", epoch=4))
    assert first == repeated
    assert first != changed


def test_pose_and_consistency_losses_have_zero_identity_case() -> None:
    clean_pose = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    angles = torch.tensor([0.0, torch.pi / 2])
    rotated_pose = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    loss, mae = orientation_loss(clean_pose, rotated_pose, angles)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert float(loss) == 0.0
    assert float(mae) == 0.0
    assert float(embedding_consistency(embeddings, embeddings)) == 0.0


def test_rotation_padding_is_explicit() -> None:
    tensor = torch.ones(1, 1, 17, 17)
    angle = torch.tensor([torch.pi / 4])
    gray = rotate_tensor(tensor, angle, 0.0)
    white = rotate_tensor(tensor, angle, 1.0)
    assert float(white[0, 0, 0, 0]) > float(gray[0, 0, 0, 0])
