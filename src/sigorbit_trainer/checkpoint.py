"""Pickle-free recovery checkpoints using safetensors plus strict JSON."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load, save_file
from torch import Tensor, nn

from .modeling import load_portable_state, portable_state_dict

_MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    artifact_type: Literal["sigorbit-trainer-recovery"]
    run_id: str = Field(min_length=1, max_length=128)
    stage: Literal["backbone", "pose", "joint"]
    epoch_completed: int = Field(ge=-1, le=10_000)
    global_step: int = Field(ge=0)
    model_kind: Literal["backbone", "canonicalized"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(min_length=16, max_length=128)
    class_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    architecture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tensor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_type: str = Field(min_length=1, max_length=32)
    device_name: str | None
    cuda_count: int = Field(ge=0, le=1024)
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None
    rng: dict[str, Any]
    best_score: tuple[float, float] | None
    metrics: dict[str, float | int]
    created_at: datetime


class LoadedCheckpoint:
    def __init__(
        self, path: Path, metadata: CheckpointMetadata, tensors: dict[str, Tensor]
    ) -> None:
        self.path = path
        self.metadata = metadata
        self.tensors = tensors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def named_parameters(model: nn.Module, head: nn.Module | None) -> dict[str, nn.Parameter]:
    result = {f"model.{name}": parameter for name, parameter in model.named_parameters()}
    if head is not None:
        result.update({f"head.{name}": parameter for name, parameter in head.named_parameters()})
    return result


def save_checkpoint(
    checkpoint_root: Path,
    *,
    run_id: str,
    stage: Literal["backbone", "pose", "joint"],
    epoch_completed: int,
    global_step: int,
    model_kind: Literal["backbone", "canonicalized"],
    model: nn.Module,
    head: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config_sha256: str,
    dataset_sha256: str,
    class_map_sha256: str,
    architecture_sha256: str,
    best_score: tuple[float, float] | None,
    metrics: dict[str, float | int],
    mark_best: bool,
    device: torch.device,
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(checkpoint_root, 0o700)
    final_name = f"stage={stage}-epoch={epoch_completed:04d}-step={global_step:09d}"
    final_path = checkpoint_root / final_name
    if final_path.exists():
        raise FileExistsError(f"checkpoint already exists: {final_name}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoint_root))
    try:
        tensors: dict[str, Tensor] = {
            f"model::{key}": value for key, value in portable_state_dict(model).items()
        }
        if head is not None:
            tensors.update(
                {f"head::{key}": value for key, value in portable_state_dict(head).items()}
            )
        parameter_names = named_parameters(model, head)
        optimizer_json = _encode_optimizer(optimizer, parameter_names, tensors)
        rng_json = _encode_rng(tensors)
        tensor_path = temporary / "tensors.safetensors"
        save_file(tensors, tensor_path)
        os.chmod(tensor_path, 0o600)
        tensor_sha = sha256_file(tensor_path)
        device_type = device.type
        device_name = torch.cuda.get_device_name(0) if device_type == "cuda" else None
        created_at = datetime.now(timezone.utc)
        base_metadata: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "sigorbit-trainer-recovery",
            "run_id": run_id,
            "stage": stage,
            "epoch_completed": epoch_completed,
            "global_step": global_step,
            "model_kind": model_kind,
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "class_map_sha256": class_map_sha256,
            "architecture_sha256": architecture_sha256,
            "tensor_sha256": tensor_sha,
            "device_type": device_type,
            "device_name": device_name,
            "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            # NOTE: cuda_count records host topology; device_type records the
            # resolved training device for this run.
            "optimizer": optimizer_json,
            "scheduler": _json_safe(scheduler.state_dict()) if scheduler is not None else None,
            "rng": rng_json,
            "best_score": best_score,
            "metrics": metrics,
            "created_at": created_at,
        }
        placeholder_metadata = CheckpointMetadata.model_validate(
            {**base_metadata, "metadata_sha256": "0" * 64}
        )
        canonical_without_sha = json.dumps(
            placeholder_metadata.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).replace(f'"metadata_sha256":"{"0" * 64}"', '"metadata_sha256":"' + "0" * 64 + '"')
        metadata_sha = hashlib.sha256(canonical_without_sha.encode()).hexdigest()
        metadata = CheckpointMetadata.model_validate(
            {**base_metadata, "metadata_sha256": metadata_sha}
        )
        metadata_path = temporary / "checkpoint.json"
        _write_private_json(metadata_path, metadata.model_dump(mode="json"))
        _fsync_directory(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(checkpoint_root)
        _write_pointer(checkpoint_root / "latest.json", final_name, tensor_sha, metadata_sha)
        if mark_best:
            _write_pointer(
                checkpoint_root / f"best-{stage}.json", final_name, tensor_sha, metadata_sha
            )
        return final_path
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    _reject_any_symlink(path.expanduser().absolute())
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("checkpoint must be a non-symlink directory")
    metadata_path = path / "checkpoint.json"
    tensor_path = path / "tensors.safetensors"
    for file_path in (metadata_path, tensor_path):
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError("checkpoint contains missing or symlinked files")
    if (
        metadata_path.stat().st_size > 1024 * 1024
        or tensor_path.stat().st_size > _MAX_CHECKPOINT_BYTES
    ):
        raise ValueError("checkpoint exceeds size limit")
    with metadata_path.open("rb") as handle:
        metadata_bytes = handle.read(1024 * 1024 + 1)
    with tensor_path.open("rb") as handle:
        tensor_bytes = handle.read(_MAX_CHECKPOINT_BYTES + 1)
    if len(metadata_bytes) > 1024 * 1024 or len(tensor_bytes) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds size limit")
    metadata = CheckpointMetadata.model_validate_json(metadata_bytes)
    if hashlib.sha256(tensor_bytes).hexdigest() != metadata.tensor_sha256:
        raise ValueError("checkpoint tensor digest mismatch")
    computed_canonical = json.dumps(
        metadata.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).replace(
        f'"metadata_sha256":"{metadata.metadata_sha256}"', '"metadata_sha256":"' + "0" * 64 + '"'
    )
    if hashlib.sha256(computed_canonical.encode()).hexdigest() != metadata.metadata_sha256:
        raise ValueError("checkpoint metadata digest mismatch")
    tensors = load(tensor_bytes)
    if not all(
        torch.isfinite(value).all() for value in tensors.values() if value.is_floating_point()
    ):
        raise ValueError("checkpoint contains non-finite tensors")
    return LoadedCheckpoint(path, metadata, tensors)


def restore_checkpoint(
    loaded: LoadedCheckpoint,
    model: nn.Module,
    head: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    model_state = {
        key.removeprefix("model::"): value
        for key, value in loaded.tensors.items()
        if key.startswith("model::")
    }
    load_portable_state(model, model_state)
    head_state = {
        key.removeprefix("head::"): value
        for key, value in loaded.tensors.items()
        if key.startswith("head::")
    }
    if head is None and head_state:
        raise ValueError("checkpoint has a head but runtime does not")
    if head is not None:
        load_portable_state(head, head_state)
    _decode_optimizer(
        optimizer, named_parameters(model, head), loaded.metadata.optimizer, loaded.tensors
    )
    if scheduler is not None:
        if loaded.metadata.scheduler is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(loaded.metadata.scheduler)
    elif loaded.metadata.scheduler is not None:
        raise ValueError("checkpoint scheduler mismatch")
    _decode_rng(loaded.metadata.rng, loaded.tensors)


def resolve_pointer(checkpoint_root: Path, pointer: str = "latest") -> Path:
    pointer_path = checkpoint_root / f"{pointer}.json"
    payload = json.loads(pointer_path.read_text())
    if not isinstance(payload, dict) or set(payload) != {
        "checkpoint",
        "tensor_sha256",
        "metadata_sha256",
    }:
        raise ValueError("invalid checkpoint pointer")
    checkpoint_name = payload["checkpoint"]
    tensor_sha256 = payload["tensor_sha256"]
    metadata_sha256 = payload["metadata_sha256"]
    if (
        not isinstance(checkpoint_name, str)
        or not isinstance(tensor_sha256, str)
        or not isinstance(metadata_sha256, str)
    ):
        raise ValueError("invalid checkpoint pointer values")
    candidate = (checkpoint_root / checkpoint_name).resolve(strict=True)
    if not candidate.is_relative_to(checkpoint_root.resolve()):
        raise ValueError("checkpoint pointer escapes root")
    loaded = load_checkpoint(candidate)
    if loaded.metadata.tensor_sha256 != tensor_sha256:
        raise ValueError("checkpoint pointer tensor digest mismatch")
    if loaded.metadata.metadata_sha256 != metadata_sha256:
        raise ValueError("checkpoint pointer metadata digest mismatch")
    return candidate


def _encode_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, nn.Parameter],
    tensors: dict[str, Tensor],
) -> dict[str, Any]:
    reverse = {id(parameter): name for name, parameter in parameters.items()}
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        encoded = {key: _json_safe(value) for key, value in group.items() if key != "params"}
        encoded["params"] = [reverse[id(parameter)] for parameter in group["params"]]
        groups.append(encoded)
    non_tensors: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        name = reverse[id(parameter)]
        encoded_state: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, Tensor):
                tensors[f"optimizer::{name}::{key}"] = value.detach().cpu().contiguous()
            else:
                encoded_state[key] = _json_safe(value)
        non_tensors[name] = encoded_state
    return {
        "class": type(optimizer).__name__,
        "param_groups": groups,
        "non_tensor_state": non_tensors,
    }


def _decode_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, nn.Parameter],
    payload: dict[str, Any],
    tensors: dict[str, Tensor],
) -> None:
    if payload.get("class") != type(optimizer).__name__:
        raise ValueError("optimizer class mismatch")
    groups = payload.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("optimizer parameter-group mismatch")
    optimizer.state.clear()
    for target, saved in zip(optimizer.param_groups, groups, strict=True):
        names = saved.get("params")
        current_names = [
            next(name for name, parameter in parameters.items() if parameter is item)
            for item in target["params"]
        ]
        if names != current_names:
            raise ValueError("optimizer parameter names mismatch")
        for key, value in saved.items():
            if key != "params":
                target[key] = value
    non_tensors = payload.get("non_tensor_state", {})
    for name, parameter in parameters.items():
        state: dict[str, Any] = dict(non_tensors.get(name, {}))
        prefix = f"optimizer::{name}::"
        for tensor_name, tensor in tensors.items():
            if tensor_name.startswith(prefix):
                state[tensor_name.removeprefix(prefix)] = tensor.to(parameter.device)
        if state:
            optimizer.state[parameter] = state


def _encode_rng(tensors: dict[str, Tensor]) -> dict[str, Any]:
    tensors["rng::torch_cpu"] = torch.random.get_rng_state().contiguous()
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for index, state in enumerate(torch.cuda.get_rng_state_all() if cuda_count else []):
        tensors[f"rng::cuda::{index}"] = state.cpu().contiguous()
    numpy_state: tuple[str, np.ndarray[Any, Any], int, int, float] = np.random.get_state()  # type: ignore[assignment]
    return {
        "python": _json_safe(random.getstate()),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "cuda_count": cuda_count,
    }


def _decode_rng(payload: dict[str, Any], tensors: dict[str, Tensor]) -> None:
    random.setstate(_nested_tuple(payload["python"]))
    numpy_state = payload["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.random.set_rng_state(tensors["rng::torch_cpu"])
    if payload["cuda_count"]:
        if torch.cuda.device_count() != payload["cuda_count"]:
            raise ValueError("CUDA topology differs from recovery checkpoint")
        torch.cuda.set_rng_state_all(
            [tensors[f"rng::cuda::{index}"] for index in range(payload["cuda_count"])]
        )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"unsupported recovery metadata value: {type(value).__name__}")


def _write_pointer(path: Path, checkpoint: str, tensor_sha256: str, metadata_sha256: str) -> None:
    _write_private_json(
        path,
        {
            "checkpoint": checkpoint,
            "tensor_sha256": tensor_sha256,
            "metadata_sha256": metadata_sha256,
        },
    )


def _write_private_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        + b"\n"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_any_symlink(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinked checkpoint paths are rejected")
        if not current.exists():
            break


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
