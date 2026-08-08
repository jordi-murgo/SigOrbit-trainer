"""Strict experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

RIGHTS_ASSERTION = "I_ASSERT_AUTHORIZED_USE_OF_THIS_DATASET"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunConfig(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    output_dir: Path
    seed: int = Field(ge=0, le=2**31 - 1)
    deterministic: bool = True


class SyntheticDataConfig(StrictModel):
    train_signers: int = Field(default=4, ge=2, le=64)
    validation_signers: int = Field(default=3, ge=2, le=64)
    samples_per_signer: int = Field(default=4, ge=2, le=32)
    canvas_width: int = Field(default=160, ge=32, le=1024)
    canvas_height: int = Field(default=80, ge=32, le=1024)


class SplitConfig(StrictModel):
    """Deterministic writer-level repartitioning applied in memory."""

    strategy: Literal["manifest", "random_by_signer"] = "manifest"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0, allow_inf_nan=False)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0, allow_inf_nan=False)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_fractions(self) -> SplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to exactly 1.0")
        return self


class DataConfig(StrictModel):
    kind: Literal["local_manifest", "synthetic"]
    manifest: Path | None = None
    rights_attestation: Path | None = None
    verify_hashes: bool = True
    require_signer_disjoint_splits: bool = True
    include_forgeries: bool = False
    max_files: int = Field(default=100_000, ge=1, le=1_000_000)
    max_file_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=128 * 1024 * 1024)
    max_pixels: int = Field(default=16_777_216, ge=1024, le=67_108_864)
    synthetic: SyntheticDataConfig | None = None
    split: SplitConfig = SplitConfig()

    @model_validator(mode="after")
    def validate_kind(self) -> DataConfig:
        if self.kind == "local_manifest":
            if self.manifest is None or self.rights_attestation is None:
                raise ValueError("local_manifest requires manifest and rights_attestation")
            if self.synthetic is not None:
                raise ValueError("local_manifest cannot define synthetic")
        else:
            if self.manifest is not None or self.rights_attestation is not None:
                raise ValueError("synthetic data cannot reference external files")
            if self.synthetic is None:
                raise ValueError("synthetic data requires [data.synthetic]")
        return self


class ModelConfig(StrictModel):
    input_size: int = Field(default=257, ge=33, le=513)
    group_order: int = Field(default=8, ge=2, le=16)
    widths: tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt] = (24, 48, 96, 128)
    embedding_dim: int = Field(default=256, ge=16, le=1024)
    release_compatible: bool = True

    @model_validator(mode="after")
    def validate_model(self) -> ModelConfig:
        if self.input_size % 2 == 0:
            raise ValueError("input_size must be odd")
        if any(width > 256 for width in self.widths):
            raise ValueError("model widths must not exceed 256")
        if self.release_compatible and (
            self.input_size != 257
            or self.group_order != 8
            or self.widths != (24, 48, 96, 128)
            or self.embedding_dim != 256
        ):
            raise ValueError("release_compatible requires the SigOrbit 257/C8 v1 architecture")
        return self


class SamplerConfig(StrictModel):
    persons_per_batch: int = Field(default=8, ge=2, le=64)
    samples_per_person: int = Field(default=4, ge=2, le=32)

    @model_validator(mode="after")
    def validate_batch(self) -> SamplerConfig:
        if self.persons_per_batch * self.samples_per_person > 256:
            raise ValueError("P*K must not exceed 256")
        return self


class RuntimeConfig(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # "tf32" keeps fp32 storage and only lets conv/matmul accumulate on Ampere+
    # tensor cores. It is still bitwise reproducible run to run -- it trades
    # mantissa bits, not determinism -- and is ~3.5x faster on A100 at 257px.
    precision: Literal["fp32", "tf32"] = "fp32"
    workers: int = Field(default=6, ge=0, le=32)
    validation_batch_size: int = Field(default=16, ge=2, le=256)
    log_every_steps: int = Field(default=50, ge=1, le=100_000)


class BackboneStageConfig(StrictModel):
    epochs: int = Field(default=40, ge=1, le=500)
    lr: float = Field(default=1e-3, gt=0, le=1.0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, le=1.0, allow_inf_nan=False)
    warmup_epochs: int = Field(default=2, ge=0, le=100)
    arc_scale: float = Field(default=16.0, gt=0, le=128, allow_inf_nan=False)
    arc_margin: float = Field(default=0.35, ge=0, le=1.5, allow_inf_nan=False)
    margin_hold_epochs: int = Field(default=2, ge=0, le=100)
    margin_warmup_epochs: int = Field(default=8, ge=1, le=100)
    discrete_rotations_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
    patience: int = Field(default=8, ge=1, le=500)

    @model_validator(mode="after")
    def validate_rotations(self) -> BackboneStageConfig:
        if not 1 <= len(self.discrete_rotations_deg) <= 32:
            raise ValueError("backbone requires 1..32 discrete rotations")
        if any(
            not math.isfinite(angle) or not -180.0 <= angle <= 180.0
            for angle in self.discrete_rotations_deg
        ):
            raise ValueError("backbone rotation angles must be finite and in [-180, 180]")
        return self


class PoseStageConfig(StrictModel):
    epochs: int = Field(default=10, ge=1, le=500)
    lr: float = Field(default=3e-3, gt=0, le=1.0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, le=1.0, allow_inf_nan=False)
    start_max_degrees: float = Field(default=45.0, gt=0, le=180, allow_inf_nan=False)
    end_max_degrees: float = Field(default=180.0, gt=0, le=180, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_range(self) -> PoseStageConfig:
        if self.start_max_degrees > self.end_max_degrees:
            raise ValueError("pose start_max_degrees must not exceed end_max_degrees")
        return self


class JointStageConfig(StrictModel):
    epochs: int = Field(default=40, ge=1, le=500)
    backbone_lr: float = Field(default=3e-4, gt=0, le=1.0, allow_inf_nan=False)
    canonicalizer_lr: float = Field(default=1e-3, gt=0, le=1.0, allow_inf_nan=False)
    head_lr: float = Field(default=1e-3, gt=0, le=1.0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, le=1.0, allow_inf_nan=False)
    warmup_epochs: int = Field(default=5, ge=0, le=100)
    arc_scale: float = Field(default=16.0, gt=0, le=128, allow_inf_nan=False)
    arc_margin: float = Field(default=0.35, ge=0, le=1.5, allow_inf_nan=False)
    margin_hold_epochs: int = Field(default=2, ge=0, le=100)
    margin_warmup_epochs: int = Field(default=8, ge=1, le=100)
    rotation_start_degrees: float = Field(default=10.0, gt=0, le=180, allow_inf_nan=False)
    rotation_end_degrees: float = Field(default=180.0, gt=0, le=180, allow_inf_nan=False)
    rotation_curriculum_epochs: int = Field(default=20, ge=1, le=500)
    orientation_weight: float = Field(default=0.5, ge=0, le=10, allow_inf_nan=False)
    consistency_weight: float = Field(default=0.5, ge=0, le=10, allow_inf_nan=False)
    tensor_rotation_padding: float = Field(default=0.0, ge=-1, le=1, allow_inf_nan=False)
    patience: int = Field(default=999, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_range(self) -> JointStageConfig:
        if self.rotation_start_degrees > self.rotation_end_degrees:
            raise ValueError("joint rotation start must not exceed end")
        return self


class EvaluationConfig(StrictModel):
    rotation_angles_degrees: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 90.0, 180.0)
    fragile_margin: float = Field(default=0.05, ge=-2, le=2, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_rotations(self) -> EvaluationConfig:
        if not 1 <= len(self.rotation_angles_degrees) <= 72:
            raise ValueError("evaluation requires 1..72 rotation angles")
        if any(
            not math.isfinite(angle) or not -180.0 <= angle <= 180.0
            for angle in self.rotation_angles_degrees
        ):
            raise ValueError("evaluation rotation angles must be finite and in [-180, 180]")
        return self


class ExportConfig(StrictModel):
    filename: str = Field(default="sigorbit-model.pt", pattern=r"^[a-zA-Z0-9._-]+\.pt$")
    model_id: str = Field(default="sigorbit-c8-257-custom-v1", min_length=1, max_length=128)
    preprocess_version: str = Field(
        default="sigorbit-gray-square-257-v1", min_length=1, max_length=128
    )
    weight_release_status: Literal[
        "private-research-only", "internal-approved", "public-approved"
    ] = "private-research-only"
    public_approval_reference: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def validate_release(self) -> ExportConfig:
        if self.weight_release_status == "public-approved" and not self.public_approval_reference:
            raise ValueError("public-approved weights require public_approval_reference")
        return self


class TrainerConfig(StrictModel):
    schema_version: Literal[1]
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    sampler: SamplerConfig
    runtime: RuntimeConfig
    backbone: BackboneStageConfig
    pose: PoseStageConfig
    joint: JointStageConfig
    evaluation: EvaluationConfig
    export: ExportConfig

    @model_validator(mode="after")
    def validate_epochs(self) -> TrainerConfig:
        if self.backbone.warmup_epochs > self.backbone.epochs:
            raise ValueError("backbone warmup_epochs cannot exceed epochs")
        if self.joint.warmup_epochs > self.joint.epochs:
            raise ValueError("joint warmup_epochs cannot exceed epochs")
        return self


def load_config(path: Path, *, split_seed: int | None = None) -> TrainerConfig:
    source = path.expanduser().resolve(strict=True)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    raw["run"]["output_dir"] = Path(raw["run"]["output_dir"])
    if raw["data"].get("manifest") is not None:
        raw["data"]["manifest"] = Path(raw["data"]["manifest"])
    if raw["data"].get("rights_attestation") is not None:
        raw["data"]["rights_attestation"] = Path(raw["data"]["rights_attestation"])
    raw["model"]["widths"] = tuple(raw["model"]["widths"])
    raw["backbone"]["discrete_rotations_deg"] = tuple(raw["backbone"]["discrete_rotations_deg"])
    raw["evaluation"]["rotation_angles_degrees"] = tuple(
        raw["evaluation"]["rotation_angles_degrees"]
    )
    config = TrainerConfig.model_validate(raw)
    if split_seed is not None:
        if config.data.split.strategy != "random_by_signer":
            raise ValueError(
                'a split seed override requires [data.split] strategy = "random_by_signer"'
            )
        config = config.model_copy(
            update={
                "data": config.data.model_copy(
                    update={"split": config.data.split.model_copy(update={"seed": split_seed})}
                )
            }
        )
    base = source.parent
    updates: dict[str, object] = {
        "run": config.run.model_copy(update={"output_dir": _resolve(base, config.run.output_dir)})
    }
    if config.data.kind == "local_manifest":
        assert config.data.manifest is not None and config.data.rights_attestation is not None
        updates["data"] = config.data.model_copy(
            update={
                "manifest": _resolve(base, config.data.manifest),
                "rights_attestation": _resolve(base, config.data.rights_attestation),
            }
        )
    return config.model_copy(update=updates)


def _resolve(base: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return (expanded if expanded.is_absolute() else base / expanded).resolve()


def canonical_config(config: TrainerConfig) -> bytes:
    payload = config.model_dump(mode="json")
    payload["run"]["output_dir"] = "<OUTPUT_DIR>"
    if payload["data"]["kind"] == "local_manifest":
        payload["data"]["manifest"] = "<DATASET_MANIFEST>"
        payload["data"]["rights_attestation"] = "<RIGHTS_ATTESTATION>"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def config_sha256(config: TrainerConfig) -> str:
    return hashlib.sha256(canonical_config(config)).hexdigest()
