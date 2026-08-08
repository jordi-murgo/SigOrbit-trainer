"""Pinned SigOrbit 0.1 architecture adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from sigorbit import CanonicalizedEncoder, SteerableEncoder
from sigorbit import ModelConfig as SigOrbitModelConfig
from torch import Tensor, nn

from .config import ModelConfig

_DERIVED_SUFFIXES = (".filter", ".expanded_bias")


def sigorbit_model_config(config: ModelConfig) -> SigOrbitModelConfig:
    return SigOrbitModelConfig(
        input_size=config.input_size,
        rotations=config.group_order,
        widths=config.widths,
        embedding_dim=config.embedding_dim,
        dropout=0.3,
    )


def architecture_sha256(config: ModelConfig) -> str:
    data = sigorbit_model_config(config).to_checkpoint_dict()
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_backbone(config: ModelConfig) -> SteerableEncoder:
    return SteerableEncoder(sigorbit_model_config(config))


def build_canonicalized(config: ModelConfig) -> CanonicalizedEncoder:
    return CanonicalizedEncoder(sigorbit_model_config(config))


def portable_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in module.state_dict().items()
        if not key.endswith(_DERIVED_SUFFIXES)
    }


def load_portable_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    invalid_missing = [key for key in missing if not key.endswith(_DERIVED_SUFFIXES)]
    if unexpected or invalid_missing:
        raise ValueError(
            f"incompatible model tensors: missing={invalid_missing}, unexpected={list(unexpected)}"
        )


def assert_model_contract(model: nn.Module, config: ModelConfig, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        probe = torch.ones(2, 1, config.input_size, config.input_size, device=device)
        output = model(probe)
        if isinstance(output, tuple):
            output = output[0]
        if output.shape != (2, config.embedding_dim):
            raise RuntimeError(f"unexpected encoder output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError("encoder preflight produced non-finite values")
        norms = torch.linalg.vector_norm(output, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
            raise RuntimeError("encoder output is not unit-normalized")
