"""Three-stage SigOrbit training state machine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sigorbit import CanonicalizedEncoder, SteerableEncoder, load_model
from torch import nn
from torch.utils.data import DataLoader

from .augment import build_eval_transform, build_train_transform, rotate_tensor, sample_angles
from .checkpoint import (
    LoadedCheckpoint,
    load_checkpoint,
    resolve_pointer,
    restore_checkpoint,
    save_checkpoint,
    sha256_file,
)
from .config import TrainerConfig, canonical_config, config_sha256
from .data import DatasetBundle, LocalManifestSource, SignatureDataset, load_data
from .losses import ArcFace, embedding_consistency, orientation_loss
from .metrics import (
    ValidationMetrics,
    embed_validation,
    leave_one_out,
    rotation_consistency,
    rotation_retrieval,
)
from .modeling import (
    architecture_sha256,
    assert_model_contract,
    build_backbone,
    build_canonicalized,
    load_portable_state,
    portable_state_dict,
    sigorbit_model_config,
)
from .sampler import PKBatchSampler


class RunResult:
    def __init__(
        self, output_dir: Path, checkpoint: Path, sha256: str, metrics: dict[str, Any]
    ) -> None:
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.sha256 = sha256
        self.metrics = metrics


def _log(message: str) -> None:
    """Timestamped progress line; runs last hours, so wall clock beats step counts.

    Layout follows spike-signature's finetune_equivariant.py / finetune_canonicalized.py
    so their logs and these stay directly comparable line by line.
    """
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def _stage_banner(
    stage: str, epochs: int, steps: int, config: TrainerConfig, data: DatasetBundle
) -> None:
    persons, samples = config.sampler.persons_per_batch, config.sampler.samples_per_person
    _log(
        f"=== {stage}: {epochs} epochs x {steps} steps, PK {persons}x{samples}={persons * samples}, "
        f"{len(data.train_records)} train imgs / {len(data.class_map)} signers, "
        f"{config.model.input_size}px, {config.runtime.precision}"
        f"{', deterministic' if config.run.deterministic else ''} ==="
    )


def _validation_line(validation: ValidationMetrics) -> str:
    return (
        f"val top-1 {validation.top1:.1f}%  top-5 {validation.top5:.1f}%  "
        f"margin {validation.median_margin:+.4f}  fragile {validation.fragile_percent:.1f}%"
    )


def _assert_determinism_supported(config: TrainerConfig, device: torch.device) -> None:
    """Reject a config that cannot finish, before it burns hours of GPU time.

    The joint stage backpropagates through the canonicalizer's grid_sample, and
    grid_sampler_2d_backward_cuda has no deterministic implementation. Left
    unchecked, run.deterministic on CUDA trains the whole backbone and pose
    stages and only then dies at the first joint step.
    """
    if config.run.deterministic and device.type == "cuda":
        raise ValueError(
            "run.deterministic = true is not supported on CUDA: the SO(2) "
            "canonicalizer backpropagates through grid_sample, and "
            "grid_sampler_2d_backward_cuda has no deterministic implementation, "
            "so the joint stage cannot run. Set run.deterministic = false "
            "(seeds are still fixed, and runtime.precision is unaffected), or "
            "set runtime.device = \"cpu\"."
        )



def run_training(config: TrainerConfig, *, source_config: Path | None = None) -> RunResult:
    _configure_runtime(config)
    device = _select_device(config)
    _assert_determinism_supported(config, device)
    data = load_data(config.data, config.sampler, config.run.seed)
    output = _prepare_output(config, data)
    checkpoint_root = output / "checkpoints"
    run_id = str(uuid.uuid4())
    config_digest = config_sha256(config)
    architecture_digest = architecture_sha256(config.model)
    metrics_path = output / "metrics.jsonl"
    _write_private_json(output / "config.resolved.json", json.loads(canonical_config(config)))
    if source_config is not None:
        source_digest = hashlib.sha256(source_config.resolve(strict=True).read_bytes()).hexdigest()
        _write_private_json(output / "config.source.json", {"sha256": source_digest})
    _write_private_json(
        output / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_digest,
            "architecture_sha256": architecture_digest,
            "dataset": data.summary,
            "environment": _environment(device),
            "environment_history": [_environment(device)],
            "claim_status": "attempted-reproduction"
            if config.data.kind == "local_manifest"
            else "synthetic-smoke",
        },
    )

    try:
        backbone = build_backbone(config.model).to(device)
        assert_model_contract(backbone, config.model, device)
        head = ArcFace(
            config.model.embedding_dim,
            len(data.class_map),
            config.backbone.arc_scale,
            config.backbone.arc_margin,
        ).to(device)
        eval_transform = build_eval_transform(config.model.input_size)
        backbone, head, global_step = _train_backbone(
            config,
            data,
            backbone,
            head,
            eval_transform,
            device,
            checkpoint_root,
            metrics_path,
            run_id,
            config_digest,
            architecture_digest,
        )

        model = build_canonicalized(config.model).to(device)
        load_portable_state(model.backbone, portable_state_dict(backbone))
        global_step = _train_pose(
            config,
            data,
            model,
            head,
            device,
            checkpoint_root,
            metrics_path,
            run_id,
            config_digest,
            architecture_digest,
            global_step,
        )
        model, head, global_step = _train_joint(
            config,
            data,
            model,
            head,
            eval_transform,
            device,
            checkpoint_root,
            metrics_path,
            run_id,
            config_digest,
            architecture_digest,
            global_step,
        )

        return _finish_run(
            config,
            data,
            device,
            output,
            checkpoint_root,
            metrics_path,
            run_id,
            config_digest,
            architecture_digest,
            eval_transform,
            global_step,
        )
    except BaseException as exc:
        _write_private_json(
            output / "failure.json",
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            },
        )
        raise


def resume_training(
    config: TrainerConfig,
    *,
    checkpoint: Path,
) -> RunResult:
    _configure_runtime(config)
    device = _select_device(config)
    data = load_data(config.data, config.sampler, config.run.seed)
    output = config.run.output_dir.resolve(strict=True)
    checkpoint_root = (output / "checkpoints").resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(checkpoint_root):
        raise ValueError("resume checkpoint must belong to the configured run directory")
    if checkpoint != resolve_pointer(checkpoint_root, "latest"):
        raise ValueError("resume requires the latest recovery checkpoint")
    loaded = load_checkpoint(checkpoint)
    run_record = json.loads((output / "run.json").read_text())
    if run_record.get("status") == "completed":
        raise ValueError("completed runs cannot be resumed")
    run_id = str(run_record.get("run_id"))
    resume_history = list(run_record.get("environment_history") or [])
    resume_environment = _environment(device)
    if resume_environment not in resume_history:
        resume_history.append(resume_environment)
    _write_private_json(
        output / "run.json",
        {
            **run_record,
            "status": "resuming",
            "environment_history": resume_history,
            "last_resumed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if loaded.metadata.run_id != run_id:
        raise ValueError("checkpoint run_id does not match run directory")
    config_digest = config_sha256(config)
    architecture_digest = architecture_sha256(config.model)
    metrics_path = output / "metrics.jsonl"
    eval_transform = build_eval_transform(config.model.input_size)

    try:
        stage = loaded.metadata.stage
        if stage == "backbone":
            backbone = build_backbone(config.model).to(device)
            head = ArcFace(
                config.model.embedding_dim,
                len(data.class_map),
                config.backbone.arc_scale,
                config.backbone.arc_margin,
            ).to(device)
            backbone, head, global_step = _train_backbone(
                config,
                data,
                backbone,
                head,
                eval_transform,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                resume=loaded,
            )
            model = build_canonicalized(config.model).to(device)
            load_portable_state(model.backbone, portable_state_dict(backbone))
            global_step = _train_pose(
                config,
                data,
                model,
                head,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                global_step,
            )
            model, head, global_step = _train_joint(
                config,
                data,
                model,
                head,
                eval_transform,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                global_step,
            )
        elif stage == "pose":
            model = build_canonicalized(config.model).to(device)
            head = ArcFace(
                config.model.embedding_dim,
                len(data.class_map),
                config.backbone.arc_scale,
                config.backbone.arc_margin,
            ).to(device)
            global_step = _train_pose(
                config,
                data,
                model,
                head,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                loaded.metadata.global_step,
                resume=loaded,
            )
            model, head, global_step = _train_joint(
                config,
                data,
                model,
                head,
                eval_transform,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                global_step,
            )
        else:
            model = build_canonicalized(config.model).to(device)
            head = ArcFace(
                config.model.embedding_dim,
                len(data.class_map),
                config.joint.arc_scale,
                config.joint.arc_margin,
            ).to(device)
            model, head, global_step = _train_joint(
                config,
                data,
                model,
                head,
                eval_transform,
                device,
                checkpoint_root,
                metrics_path,
                run_id,
                config_digest,
                architecture_digest,
                loaded.metadata.global_step,
                resume=loaded,
            )
        return _finish_run(
            config,
            data,
            device,
            output,
            checkpoint_root,
            metrics_path,
            run_id,
            config_digest,
            architecture_digest,
            eval_transform,
            global_step,
        )
    except BaseException as exc:
        _write_private_json(
            output / "failure.json",
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            },
        )
        raise


def evaluate_checkpoint(
    config: TrainerConfig,
    *,
    checkpoint: Path,
    split: str,
    allow_test_split: bool,
) -> dict[str, Any]:
    """Evaluate an exported SigOrbit checkpoint; test requires an explicit opt-in."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if split == "test" and not allow_test_split:
        raise ValueError(
            "the test split is release-candidate only; pass allow_test_split to confirm"
        )
    _configure_runtime(config)
    device = _select_device(config)
    data = load_data(config.data, config.sampler, config.run.seed)
    checkpoint = checkpoint.resolve(strict=True)
    model, info = load_model(
        checkpoint,
        device="cpu",
        expected_sha256=sha256_file(checkpoint),
        strict_release_config=config.model.release_compatible,
    )
    model = model.to(device).eval()
    eval_transform = build_eval_transform(config.model.input_size)
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
        eval_transform,
        run_seed=config.run.seed,
        stage=f"evaluate-{split}",
        epoch=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.validation_batch_size,
        shuffle=False,
        num_workers=config.runtime.workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    with torch.no_grad():
        for tensor, _ in loader:
            output = model(tensor.to(device, non_blocking=True))
            embeddings.append(F.normalize(output.float(), dim=1).cpu())
    stacked = torch.cat(embeddings)
    signer_ids = [record.signer_id for record in records]
    clean = leave_one_out(stacked, signer_ids, config.evaluation.fragile_margin)
    return {
        "split": split,
        "model_id": info.model_id,
        "checkpoint_sha256": info.sha256,
        "samples": len(records),
        "signers": len(signer_map),
        "clean": clean.to_dict(),
        "rotation_retrieval": rotation_retrieval(
            model,
            data,
            eval_transform,
            config.evaluation.rotation_angles_degrees,
            split=split,
            padding_value=config.joint.tensor_rotation_padding,
            fragile_margin=config.evaluation.fragile_margin,
            input_seed=config.run.seed,
            device=device,
            batch_size=config.runtime.validation_batch_size,
            workers=config.runtime.workers,
        ),
        "claim_status": (
            "attempted-reproduction" if config.data.kind == "local_manifest" else "synthetic-smoke"
        ),
    }


def _finish_run(
    config: TrainerConfig,
    data: DatasetBundle,
    device: torch.device,
    output: Path,
    checkpoint_root: Path,
    metrics_path: Path,
    run_id: str,
    config_digest: str,
    architecture_digest: str,
    eval_transform: object,
    global_step: int,
) -> RunResult:
    best_joint = load_checkpoint(resolve_pointer(checkpoint_root, "best-joint"))
    best_model = build_canonicalized(config.model).to(device)
    best_head = ArcFace(
        config.model.embedding_dim,
        len(data.class_map),
        config.joint.arc_scale,
        config.joint.arc_margin,
    ).to(device)
    _restore_model_head(best_joint, best_model, best_head)
    embeddings, signer_ids = embed_validation(
        best_model,
        data,
        eval_transform,
        input_seed=config.run.seed,
        device=device,
        batch_size=config.runtime.validation_batch_size,
        workers=config.runtime.workers,
    )
    clean_metrics = leave_one_out(embeddings, signer_ids, config.evaluation.fragile_margin)
    rotation_metrics = rotation_consistency(
        best_model,
        embeddings,
        data,
        eval_transform,
        config.evaluation.rotation_angles_degrees,
        padding_value=config.joint.tensor_rotation_padding,
        input_seed=config.run.seed,
        device=device,
        batch_size=config.runtime.validation_batch_size,
        workers=config.runtime.workers,
    )
    rotation_retrieval_metrics = rotation_retrieval(
        best_model,
        data,
        eval_transform,
        config.evaluation.rotation_angles_degrees,
        split="validation",
        padding_value=config.joint.tensor_rotation_padding,
        fragile_margin=config.evaluation.fragile_margin,
        input_seed=config.run.seed,
        device=device,
        batch_size=config.runtime.validation_batch_size,
        workers=config.runtime.workers,
    )
    final_metrics: dict[str, Any] = {
        "clean": clean_metrics.to_dict(),
        "rotation_embedding_cosine": rotation_metrics,
        "rotation_retrieval": rotation_retrieval_metrics,
        "selected_recovery_checkpoint": best_joint.path.name,
        "global_step": global_step,
    }
    export_path, export_sha = _export_model(
        config,
        best_model,
        output,
        config_digest,
        architecture_digest,
        data,
        final_metrics,
    )
    _append_metric(
        metrics_path,
        {"stage": "final", **final_metrics, "artifact_sha256": export_sha},
    )
    previous_record: dict[str, Any] = {}
    run_path = output / "run.json"
    if run_path.exists():
        loaded_record = json.loads(run_path.read_text())
        if isinstance(loaded_record, dict):
            previous_record = loaded_record
    environment_history = list(previous_record.get("environment_history") or [])
    current_environment = _environment(device)
    if current_environment not in environment_history:
        environment_history.append(current_environment)
    _write_private_json(
        run_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "started_at": previous_record.get("started_at"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "environment_history": environment_history,
            "config_sha256": config_digest,
            "architecture_sha256": architecture_digest,
            "dataset": data.summary,
            "environment": _environment(device),
            "claim_status": (
                "attempted-reproduction"
                if config.data.kind == "local_manifest"
                else "synthetic-smoke"
            ),
            "artifact": {"file": export_path.name, "sha256": export_sha},
            "metrics": final_metrics,
        },
    )
    failure_path = output / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    return RunResult(output, export_path, export_sha, final_metrics)


def _validate_resume(
    loaded: LoadedCheckpoint,
    *,
    stage: str,
    config_digest: str,
    dataset_digest: str,
    class_map_digest: str,
    architecture_digest: str,
    device: torch.device,
) -> None:
    metadata = loaded.metadata
    cuda_available = torch.cuda.is_available()
    expected = {
        "stage": (metadata.stage, stage),
        "config": (metadata.config_sha256, config_digest),
        "dataset": (metadata.dataset_sha256, dataset_digest),
        "class_map": (metadata.class_map_sha256, class_map_digest),
        "architecture": (metadata.architecture_sha256, architecture_digest),
        # Topology is compared unconditionally so a CPU-started run cannot be
        # silently continued on CUDA, and vice versa.
        "device_type": (metadata.device_type, device.type),
        "cuda_count": (
            metadata.cuda_count,
            torch.cuda.device_count() if cuda_available else 0,
        ),
        "device_name": (
            metadata.device_name,
            torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        ),
    }
    mismatches = [name for name, pair in expected.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError(f"recovery checkpoint mismatch: {', '.join(mismatches)}")


def _train_backbone(
    config: TrainerConfig,
    data: DatasetBundle,
    model: SteerableEncoder,
    head: ArcFace,
    eval_transform: object,
    device: torch.device,
    checkpoint_root: Path,
    metrics_path: Path,
    run_id: str,
    config_digest: str,
    architecture_digest: str,
    resume: LoadedCheckpoint | None = None,
) -> tuple[SteerableEncoder, ArcFace, int]:
    parameters = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.backbone.lr, weight_decay=config.backbone.weight_decay
    )
    steps = _steps_per_epoch(config, data)
    scheduler = _cosine_scheduler(
        optimizer, config.backbone.warmup_epochs * steps, config.backbone.epochs * steps
    )
    if resume is None:
        validation = _validate(config, data, model, eval_transform, device)
        best_score = validation.score(config.evaluation.selection_top1_floor)
        global_step = 0
        start_epoch = 0
        bad_epochs = 0
        save_checkpoint(
            checkpoint_root,
            run_id=run_id,
            stage="backbone",
            epoch_completed=-1,
            global_step=global_step,
            model_kind="backbone",
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            config_sha256=config_digest,
            dataset_sha256=data.source.fingerprint,
            class_map_sha256=data.class_map_sha256,
            architecture_sha256=architecture_digest,
            best_score=best_score,
            metrics=validation.to_dict(),
            mark_best=True,
            device=device,
        )
    else:
        _validate_resume(
            resume,
            stage="backbone",
            config_digest=config_digest,
            dataset_digest=data.source.fingerprint,
            class_map_digest=data.class_map_sha256,
            architecture_digest=architecture_digest,
            device=device,
        )
        restore_checkpoint(resume, model, head, optimizer, scheduler)
        if resume.metadata.best_score is None:
            raise ValueError("backbone recovery state lacks best_score")
        best_score = resume.metadata.best_score
        global_step = resume.metadata.global_step
        start_epoch = resume.metadata.epoch_completed + 1
        bad_epochs = int(resume.metadata.metrics.get("bad_epochs", 0))
    transform = build_train_transform(
        config.model.input_size, discrete_angles=config.backbone.discrete_rotations_deg
    )
    _stage_banner("backbone", config.backbone.epochs, steps, config, data)
    for epoch in range(start_epoch, config.backbone.epochs):
        if bad_epochs >= config.backbone.patience:
            _log(f"backbone stopped early; patience={config.backbone.patience} exhausted")
            break
        head.margin = _margin(
            epoch,
            config.backbone.margin_hold_epochs,
            config.backbone.margin_warmup_epochs,
            config.backbone.arc_margin,
        )
        loader = _train_loader(config, data, transform, "backbone", epoch, device)
        model.train()
        head.train()
        loss_sum = correct = seen = 0
        started = time.monotonic()
        for batch_index, (images, labels) in enumerate(loader):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            logits = head(model(images), labels)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite backbone loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_sum += loss.item() * labels.shape[0]
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += labels.shape[0]
            if (batch_index + 1) % config.runtime.log_every_steps == 0:
                _log(
                    f"    ep{epoch + 1} {batch_index + 1}/{len(loader)}  "
                    f"loss {loss_sum / seen:.3f}  acc {correct / seen * 100:.1f}%  "
                    f"m={head.margin:.2f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"({time.monotonic() - started:.0f}s)"
                )
        validation = _validate(config, data, model, eval_transform, device)
        improved = validation.score(config.evaluation.selection_top1_floor) > best_score
        if improved:
            best_score, bad_epochs = validation.score(config.evaluation.selection_top1_floor), 0
        else:
            bad_epochs += 1
        epoch_metrics = {
            "loss": loss_sum / seen,
            "train_accuracy": correct / seen * 100.0,
            "seconds": time.monotonic() - started,
            **validation.to_dict(),
            "bad_epochs": bad_epochs,
        }
        _append_metric(metrics_path, {"stage": "backbone", "epoch": epoch, **epoch_metrics})
        save_checkpoint(
            checkpoint_root,
            run_id=run_id,
            stage="backbone",
            epoch_completed=epoch,
            global_step=global_step,
            model_kind="backbone",
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            config_sha256=config_digest,
            dataset_sha256=data.source.fingerprint,
            class_map_sha256=data.class_map_sha256,
            architecture_sha256=architecture_digest,
            best_score=best_score,
            metrics={
                key: value
                for key, value in epoch_metrics.items()
                if isinstance(value, (int, float))
            },
            mark_best=improved,
            device=device,
        )
        _log(
            f"  epoch {epoch + 1}/{config.backbone.epochs}  "
            f"loss {epoch_metrics['loss']:.3f}  "
            f"train-acc {epoch_metrics['train_accuracy']:.1f}%  |  "
            f"{_validation_line(validation)}  "
            f"({epoch_metrics['seconds']:.0f}s)"
            f"{'  <-- best, saved' if improved else f'  (bad {bad_epochs}/{config.backbone.patience})'}"
        )
        if bad_epochs >= config.backbone.patience:
            _log(f"  early stop: no val improvement in {config.backbone.patience} epochs")
            break
    best = load_checkpoint(resolve_pointer(checkpoint_root, "best-backbone"))
    _restore_model_head(best, model, head)
    return model, head, global_step


def _train_pose(
    config: TrainerConfig,
    data: DatasetBundle,
    model: CanonicalizedEncoder,
    head: ArcFace,
    device: torch.device,
    checkpoint_root: Path,
    metrics_path: Path,
    run_id: str,
    config_digest: str,
    architecture_digest: str,
    global_step: int,
    resume: LoadedCheckpoint | None = None,
) -> int:
    canonicalizer = model.canon
    optimizer = torch.optim.AdamW(
        canonicalizer.parameters(), lr=config.pose.lr, weight_decay=config.pose.weight_decay
    )
    if resume is None:
        start_epoch = 0
    else:
        _validate_resume(
            resume,
            stage="pose",
            config_digest=config_digest,
            dataset_digest=data.source.fingerprint,
            class_map_digest=data.class_map_sha256,
            architecture_digest=architecture_digest,
            device=device,
        )
        restore_checkpoint(resume, model, head, optimizer, None)
        global_step = resume.metadata.global_step
        start_epoch = resume.metadata.epoch_completed + 1
    transform = build_train_transform(config.model.input_size, discrete_angles=None)
    _log(
        f"=== canonicalizer pretraining: {config.pose.epochs} epochs, "
        f"rot +-{config.pose.start_max_degrees:.0f}deg -> +-{config.pose.end_max_degrees:.0f}deg ==="
    )
    for epoch in range(start_epoch, config.pose.epochs):
        maximum = config.pose.start_max_degrees + (epoch / max(1, config.pose.epochs - 1)) * (
            config.pose.end_max_degrees - config.pose.start_max_degrees
        )
        loader = _train_loader(config, data, transform, "pose", epoch, device)
        canonicalizer.train()
        loss_sum = mae_sum = seen = 0
        started = time.monotonic()
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            angles = sample_angles(
                images.shape[0],
                maximum,
                run_seed=config.run.seed,
                stage="pose",
                epoch=epoch,
                batch_index=batch_index,
                device=device,
                dtype=images.dtype,
            )
            rotated = rotate_tensor(images, angles, config.joint.tensor_rotation_padding)
            _, clean_pose = canonicalizer(images)
            _, rotated_pose = canonicalizer(rotated)
            loss, mae = orientation_loss(clean_pose, rotated_pose, angles)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite pose loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(canonicalizer.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            loss_sum += loss.item() * images.shape[0]
            mae_sum += mae.item() * images.shape[0]
            seen += images.shape[0]
        metrics = {
            "loss": loss_sum / seen,
            "mae_degrees": mae_sum / seen,
            "maximum_degrees": maximum,
            "seconds": time.monotonic() - started,
        }
        _append_metric(metrics_path, {"stage": "pose", "epoch": epoch, **metrics})
        save_checkpoint(
            checkpoint_root,
            run_id=run_id,
            stage="pose",
            epoch_completed=epoch,
            global_step=global_step,
            model_kind="canonicalized",
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=None,
            config_sha256=config_digest,
            dataset_sha256=data.source.fingerprint,
            class_map_sha256=data.class_map_sha256,
            architecture_sha256=architecture_digest,
            best_score=None,
            metrics=metrics,
            mark_best=True,
            device=device,
        )
        _log(
            f"  pretrain {epoch + 1}/{config.pose.epochs}  "
            f"rot=+-{metrics['maximum_degrees']:.0f}deg  "
            f"loss {metrics['loss']:.4f}  MAE {metrics['mae_degrees']:.1f}deg  "
            f"({metrics['seconds']:.0f}s)"
        )
    return global_step


def _train_joint(
    config: TrainerConfig,
    data: DatasetBundle,
    model: CanonicalizedEncoder,
    head: ArcFace,
    eval_transform: object,
    device: torch.device,
    checkpoint_root: Path,
    metrics_path: Path,
    run_id: str,
    config_digest: str,
    architecture_digest: str,
    global_step: int,
    resume: LoadedCheckpoint | None = None,
) -> tuple[CanonicalizedEncoder, ArcFace, int]:
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.joint.backbone_lr},
            {"params": model.canon.parameters(), "lr": config.joint.canonicalizer_lr},
            {"params": head.parameters(), "lr": config.joint.head_lr},
        ],
        weight_decay=config.joint.weight_decay,
    )
    # Always use the joint stage scale so fresh and resumed runs agree.
    head.scale = config.joint.arc_scale
    steps = _steps_per_epoch(config, data)
    scheduler = _cosine_scheduler(
        optimizer, config.joint.warmup_epochs * steps, config.joint.epochs * steps
    )
    if resume is None:
        validation = _validate(config, data, model, eval_transform, device)
        best_score = validation.score(config.evaluation.selection_top1_floor)
        start_epoch = 0
        bad_epochs = 0
        save_checkpoint(
            checkpoint_root,
            run_id=run_id,
            stage="joint",
            epoch_completed=-1,
            global_step=global_step,
            model_kind="canonicalized",
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            config_sha256=config_digest,
            dataset_sha256=data.source.fingerprint,
            class_map_sha256=data.class_map_sha256,
            architecture_sha256=architecture_digest,
            best_score=best_score,
            metrics=validation.to_dict(),
            mark_best=True,
            device=device,
        )
    else:
        _validate_resume(
            resume,
            stage="joint",
            config_digest=config_digest,
            dataset_digest=data.source.fingerprint,
            class_map_digest=data.class_map_sha256,
            architecture_digest=architecture_digest,
            device=device,
        )
        restore_checkpoint(resume, model, head, optimizer, scheduler)
        if resume.metadata.best_score is None:
            raise ValueError("joint recovery state lacks best_score")
        best_score = resume.metadata.best_score
        global_step = resume.metadata.global_step
        start_epoch = resume.metadata.epoch_completed + 1
        bad_epochs = int(resume.metadata.metrics.get("bad_epochs", 0))
    transform = build_train_transform(config.model.input_size, discrete_angles=None)
    all_parameters = list(model.parameters()) + list(head.parameters())
    _stage_banner("joint", config.joint.epochs, len(data.train_records) // (
        config.sampler.persons_per_batch * config.sampler.samples_per_person
    ), config, data)
    for epoch in range(start_epoch, config.joint.epochs):
        if bad_epochs >= config.joint.patience:
            _log(f"joint stopped early; patience={config.joint.patience} exhausted")
            break
        maximum = config.joint.rotation_start_degrees + min(
            1.0, epoch / config.joint.rotation_curriculum_epochs
        ) * (config.joint.rotation_end_degrees - config.joint.rotation_start_degrees)
        head.margin = _margin(
            epoch,
            config.joint.margin_hold_epochs,
            config.joint.margin_warmup_epochs,
            config.joint.arc_margin,
        )
        loader = _train_loader(config, data, transform, "joint", epoch, device)
        model.train()
        head.train()
        total_sum = arc_sum = orient_sum = consistency_sum = mae_sum = seen = correct = 0
        started = time.monotonic()
        for batch_index, (images, labels) in enumerate(loader):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            angles = sample_angles(
                images.shape[0],
                maximum,
                run_seed=config.run.seed,
                stage="joint",
                epoch=epoch,
                batch_index=batch_index,
                device=device,
                dtype=images.dtype,
            )
            rotated_images = rotate_tensor(images, angles, config.joint.tensor_rotation_padding)
            clean_embedding, clean_pose = model(images, return_orientation=True)
            rotated_embedding, rotated_pose = model(rotated_images, return_orientation=True)
            clean_logits = head(clean_embedding, labels)
            rotated_logits = head(rotated_embedding, labels)
            arc = 0.5 * (
                F.cross_entropy(clean_logits, labels) + F.cross_entropy(rotated_logits, labels)
            )
            orient, mae = orientation_loss(clean_pose, rotated_pose, angles)
            consistency = embedding_consistency(clean_embedding, rotated_embedding)
            loss = (
                arc
                + config.joint.orientation_weight * orient
                + config.joint.consistency_weight * consistency
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(all_parameters, 5.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            batch_size = images.shape[0]
            total_sum += loss.item() * batch_size
            arc_sum += arc.item() * batch_size
            orient_sum += orient.item() * batch_size
            consistency_sum += consistency.item() * batch_size
            mae_sum += mae.item() * batch_size
            correct += int((clean_logits.argmax(dim=1) == labels).sum())
            seen += batch_size
            if (batch_index + 1) % config.runtime.log_every_steps == 0:
                _log(
                    f"    ep{epoch + 1} {batch_index + 1}/{len(loader)}  "
                    f"arc {arc_sum / seen:.3f}  acc {correct / seen * 100:.1f}%  "
                    f"m={head.margin:.2f}  orient {orient_sum / seen:.4f}  "
                    f"mae {mae_sum / seen:.1f}deg  consist {consistency_sum / seen:.4f}  "
                    f"({time.monotonic() - started:.0f}s)"
                )
        validation = _validate(config, data, model, eval_transform, device)
        improved = validation.score(config.evaluation.selection_top1_floor) > best_score
        if improved:
            best_score, bad_epochs = validation.score(config.evaluation.selection_top1_floor), 0
        else:
            bad_epochs += 1
        metrics = {
            "loss": total_sum / seen,
            "train_accuracy": correct / seen * 100.0,
            "arc_loss": arc_sum / seen,
            "orientation_loss": orient_sum / seen,
            "consistency_loss": consistency_sum / seen,
            "pose_mae_degrees": mae_sum / seen,
            "maximum_degrees": maximum,
            "seconds": time.monotonic() - started,
            **validation.to_dict(),
            "bad_epochs": bad_epochs,
        }
        _append_metric(metrics_path, {"stage": "joint", "epoch": epoch, **metrics})
        save_checkpoint(
            checkpoint_root,
            run_id=run_id,
            stage="joint",
            epoch_completed=epoch,
            global_step=global_step,
            model_kind="canonicalized",
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            config_sha256=config_digest,
            dataset_sha256=data.source.fingerprint,
            class_map_sha256=data.class_map_sha256,
            architecture_sha256=architecture_digest,
            best_score=best_score,
            metrics={
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            },
            mark_best=improved,
            device=device,
        )
        _log(
            f"  epoch {epoch + 1}/{config.joint.epochs}  "
            f"rot=+-{metrics['maximum_degrees']:.0f}deg  "
            f"arc {metrics['arc_loss']:.3f}  "
            f"train-acc {metrics['train_accuracy']:.1f}%  |  "
            f"{_validation_line(validation)}  "
            f"({metrics['seconds']:.0f}s)"
            f"{'  <-- best, saved' if improved else f'  (bad {bad_epochs}/{config.joint.patience})'}"
        )
        if bad_epochs >= config.joint.patience:
            _log(f"  early stop: no val improvement in {config.joint.patience} epochs")
            break
    best = load_checkpoint(resolve_pointer(checkpoint_root, "best-joint"))
    _log(
        f"best joint: epoch {best.metadata.epoch_completed + 1}  "
        f"top-1 {best.metadata.metrics.get('top1', float('nan')):.1f}%  "
        f"margin {best.metadata.metrics.get('median_margin', float('nan')):+.4f}  "
        f"({best.path.name})"
    )
    _restore_model_head(best, model, head)
    return model, head, global_step


def _validate(
    config: TrainerConfig,
    data: DatasetBundle,
    model: nn.Module,
    transform: object,
    device: torch.device,
) -> ValidationMetrics:
    embeddings, signer_ids = embed_validation(
        model,
        data,
        transform,
        input_seed=config.run.seed,
        device=device,
        batch_size=config.runtime.validation_batch_size,
        workers=config.runtime.workers,
    )
    return leave_one_out(embeddings, signer_ids, config.evaluation.fragile_margin)


def _train_loader(
    config: TrainerConfig,
    data: DatasetBundle,
    transform: object,
    stage: str,
    epoch: int,
    device: torch.device,
) -> DataLoader[Any]:
    labels = [data.class_map[record.signer_id] for record in data.train_records]
    dataset = SignatureDataset(
        data.source,
        data.train_records,
        data.class_map,
        transform,  # type: ignore[arg-type]
        run_seed=config.run.seed,
        stage=stage,
        epoch=epoch,
    )
    sampler = PKBatchSampler(
        labels,
        config.sampler.persons_per_batch,
        config.sampler.samples_per_person,
        seed=config.run.seed,
        stage=stage,
        epoch=epoch,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.runtime.workers,
        pin_memory=device.type == "cuda",
    )


def _steps_per_epoch(config: TrainerConfig, data: DatasetBundle) -> int:
    return len(data.train_records) // (
        config.sampler.persons_per_batch * config.sampler.samples_per_person
    )


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, warmup_steps)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _margin(epoch: int, hold: int, warmup: int, target: float) -> float:
    return target * min(1.0, max(0.0, (epoch - hold) / max(1, warmup)))


def _restore_model_head(loaded: LoadedCheckpoint, model: nn.Module, head: nn.Module) -> None:
    tensors = loaded.tensors
    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    head_state = {
        key.removeprefix("head::"): value
        for key, value in tensors.items()
        if key.startswith("head::")
    }
    load_portable_state(model, model_state)
    load_portable_state(head, head_state)


def _export_model(
    config: TrainerConfig,
    model: nn.Module,
    output: Path,
    config_digest: str,
    architecture_digest: str,
    data: DatasetBundle,
    metrics: dict[str, Any],
) -> tuple[Path, str]:
    artifact = output / config.export.filename
    temporary = artifact.with_name(f".{artifact.name}.tmp")
    source_state = portable_state_dict(model)
    # Reconstruct a fresh CPU model from the same portable state so derived
    # e2cnn buffers match the SigOrbit CPU runtime path and tolerance can be
    # applied to GPU-trained runs.
    model_for_export = build_canonicalized(config.model).to("cpu")
    load_portable_state(model_for_export, source_state)
    model_for_export.eval()
    package = {
        "format_version": 1,
        "config": sigorbit_model_config(config.model).to_checkpoint_dict(),
        "model_state_dict": portable_state_dict(model_for_export),
        "metadata": {
            "model_id": config.export.model_id,
            "preprocess_version": config.export.preprocess_version,
            "trainer": "sigorbit-trainer",
            "trainer_version": importlib.metadata.version("sigorbit-trainer"),
            "config_sha256": config_digest,
            "architecture_sha256": architecture_digest,
            "dataset_manifest_sha256": data.source.fingerprint,
            "class_map_sha256": data.class_map_sha256,
            "weight_release_status": config.export.weight_release_status,
            "public_approval_reference": config.export.public_approval_reference,
            "validation": metrics["clean"],
        },
    }
    torch.save(package, temporary)
    temporary.chmod(0o600)
    temporary.replace(artifact)
    digest = sha256_file(artifact)
    reloaded, info = load_model(
        artifact,
        device="cpu",
        expected_sha256=digest,
        strict_release_config=config.model.release_compatible,
    )
    reloaded.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.run.seed)
    probe = (
        torch.rand(2, 1, config.model.input_size, config.model.input_size, generator=generator)
        * 2.0
        - 1.0
    )
    with torch.no_grad():
        expected = model_for_export(probe)
        actual = reloaded(probe)
    max_abs_diff = float((expected - actual).abs().max())
    if max_abs_diff > 1e-5:
        raise RuntimeError(f"SigOrbit export parity failed: max_abs_diff={max_abs_diff}")
    if info.model_id != config.export.model_id:
        raise RuntimeError("SigOrbit export metadata mismatch")
    return artifact, digest


def _configure_runtime(config: TrainerConfig) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.run.seed)
    np.random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.run.seed)
    # Autotuning picks kernels by measured time, so it can vary between runs.
    torch.backends.cudnn.benchmark = not config.run.deterministic
    # TF32 is a precision choice, not a determinism one: the kernel is fixed, so
    # results stay bitwise reproducible. Disabling it costs ~3.5x on Ampere+.
    allow_tf32 = config.runtime.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.use_deterministic_algorithms(config.run.deterministic)


def _select_device(config: TrainerConfig) -> torch.device:
    requested = config.runtime.device
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.version.hip is not None:
        properties = torch.cuda.get_device_properties(0)
        architecture = getattr(properties, "gcnArchName", "")
        hip_version = tuple(int(part) for part in torch.version.hip.split(".")[:2])
        if "gfx1151" in architecture and hip_version < (7, 14):
            torch.backends.cudnn.enabled = False
            if hasattr(torch.backends, "miopen_enabled"):
                torch.backends.miopen_enabled = False
    return device


def _prepare_output(config: TrainerConfig, data: DatasetBundle) -> Path:
    output = config.run.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    repository = Path(__file__).resolve().parents[2]
    resolved_parent = output.parent.resolve()
    if output.resolve().is_relative_to(repository):
        raise ValueError("training output must be outside the source repository")
    if isinstance(data.source, LocalManifestSource):
        data_root = data.source.validated.root
        if output.resolve().is_relative_to(data_root) or data_root.is_relative_to(output.resolve()):
            raise ValueError("training output and dataset root must be separate")
    resolved_parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    return output


def _environment(device: torch.device) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "sigorbit-trainer",
        "sigorbit",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "e2cnn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device_type": device.type,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(0)
    return result


def _append_metric(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


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
    finally:
        if temporary.exists():
            temporary.unlink()
