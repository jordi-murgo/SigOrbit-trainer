"""Signature-specific augmentation with explicit historical conventions."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

INK_THRESHOLD = 200


def ink_bbox(
    image: Image.Image, threshold: int = INK_THRESHOLD
) -> tuple[int, int, int, int] | None:
    array = np.asarray(image.convert("L"))
    mask = array < threshold
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


class FramingJitter:
    def __init__(
        self,
        fill_width: tuple[float, float] = (0.45, 0.97),
        fill_height: tuple[float, float] = (0.35, 0.90),
        probability: float = 0.85,
    ) -> None:
        self.fill_width = fill_width
        self.fill_height = fill_height
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        if random.random() > self.probability:
            return image
        bounding_box = ink_bbox(image)
        if bounding_box is None:
            return image
        ink = image.crop(bounding_box)
        fill_width = random.uniform(*self.fill_width)
        fill_height = random.uniform(*self.fill_height)
        canvas_width = max(ink.width + 2, round(ink.width / fill_width))
        canvas_height = max(ink.height + 2, round(ink.height / fill_height))
        canvas = Image.new("L", (canvas_width, canvas_height), 255)
        offset_x = random.randint(0, canvas_width - ink.width)
        offset_y = random.randint(0, canvas_height - ink.height)
        canvas.paste(ink, (offset_x, offset_y))
        return canvas


class DiscreteRotation:
    def __init__(self, angles_degrees: Sequence[float]) -> None:
        if not angles_degrees:
            raise ValueError("at least one discrete angle is required")
        self.angles_degrees = tuple(float(angle) for angle in angles_degrees)

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.choice(self.angles_degrees)
        if angle == 0:
            return image
        return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=255)


def build_train_transform(
    input_size: int, *, discrete_angles: Sequence[float] | None
) -> transforms.Compose:
    operations: list[object] = [FramingJitter()]
    if discrete_angles is not None:
        operations.append(DiscreteRotation(discrete_angles))
    operations.extend(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=4,
                interpolation=InterpolationMode.NEAREST,
                fill=255,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(operations)


def build_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def stateless_seed(run_seed: int, stage: str, epoch: int, token: str) -> int:
    payload = f"sigorbit-trainer-v1:{run_seed}:{stage}:{epoch}:{token}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def sample_angles(
    batch_size: int,
    maximum_degrees: float,
    *,
    run_seed: int,
    stage: str,
    epoch: int,
    batch_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stateless_seed(run_seed, stage, epoch, str(batch_index)))
    unit = torch.rand(batch_size, generator=generator, dtype=torch.float32)
    radians = (unit * 2.0 - 1.0) * math.radians(maximum_degrees)
    return radians.to(device=device, dtype=dtype)


def rotate_tensor(tensor: Tensor, angles_radians: Tensor, padding_value: float) -> Tensor:
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("rotate_tensor expects Bx1xHxW grayscale tensors")
    if angles_radians.shape != (tensor.shape[0],):
        raise ValueError("angle count must equal batch size")
    cosine, sine = torch.cos(angles_radians), torch.sin(angles_radians)
    affine = torch.zeros(tensor.shape[0], 2, 3, device=tensor.device, dtype=tensor.dtype)
    affine[:, 0, 0] = cosine
    affine[:, 0, 1] = -sine
    affine[:, 1, 0] = sine
    affine[:, 1, 1] = cosine
    grid = F.affine_grid(affine, list(tensor.shape), align_corners=False)
    sampled = F.grid_sample(
        tensor,
        grid,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=False,
    )
    if padding_value != 0.0:
        mask = F.grid_sample(
            torch.ones_like(tensor),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled + (1.0 - mask).clamp(0.0, 1.0) * padding_value
    return sampled
