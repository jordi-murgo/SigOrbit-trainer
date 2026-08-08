"""Strict, local-only dataset manifests."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

_ALLOWED_MEDIA = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetManifest(StrictModel):
    schema_version: Literal[1]
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    revision: str = Field(min_length=1, max_length=256)
    records: str = Field(pattern=r"^[^/\\][^\\]*\.jsonl$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_pose: Literal["upright"]


class SampleRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9:._-]{0,191}$")
    image: str = Field(min_length=1, max_length=512)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(ge=1, le=32768)
    height: int = Field(ge=1, le=32768)
    signer_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9:._-]{0,191}$")
    split: Literal["train", "validation", "test"]
    kind: Literal["genuine", "forgery"]
    source_group: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9:._-]{0,191}$")
    canonical_angle_degrees: float = Field(default=0.0, ge=-180, le=180, allow_inf_nan=False)


class RightsAttestation(StrictModel):
    schema_version: Literal[1]
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertion: Literal["I_ASSERT_AUTHORIZED_USE_OF_THIS_DATASET"]
    permitted_purpose: Literal["research-only", "internal-approved", "commercial-approved"]
    authorization_reference: str = Field(min_length=1, max_length=256)
    approved_at: datetime


class ValidatedManifest(StrictModel):
    root: Path
    manifest_path: Path
    manifest: DatasetManifest
    manifest_sha256: str
    records_path: Path
    records_sha256: str
    records: tuple[SampleRecord, ...]
    split_counts: dict[str, int]
    signer_counts: dict[str, int]
    attestation_sha256: str
    permitted_purpose: str
    verify_on_open: bool
    max_file_bytes: int
    max_pixels: int


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"file exceeds allowed size: {path.name}")
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    manifest_path: Path,
    attestation_path: Path,
    *,
    verify_hashes: bool,
    include_forgeries: bool,
    require_disjoint: bool,
    max_files: int,
    max_file_bytes: int,
    max_pixels: int,
    minimum_train_samples: int,
) -> ValidatedManifest:
    _reject_any_symlink(manifest_path.expanduser().absolute())
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent.resolve(strict=True)
    _reject_symlink_chain(root, manifest_path)
    with manifest_path.open("rb") as handle:
        manifest = DatasetManifest.model_validate(tomllib.load(handle))
    manifest_digest = sha256_file(manifest_path, 1024 * 1024)

    records_rel = _safe_relative(manifest.records)
    records_path = _resolve_beneath(root, records_rel)
    records_digest = sha256_file(records_path, 256 * 1024 * 1024)
    if records_digest != manifest.records_sha256:
        raise ValueError("samples.jsonl digest does not match dataset.toml")

    _reject_any_symlink(attestation_path.expanduser().absolute())
    attestation_path = attestation_path.resolve(strict=True)
    if not attestation_path.is_file():
        raise ValueError("rights attestation must be a regular, non-symlink file")
    attestation_bytes = attestation_path.read_bytes()
    if len(attestation_bytes) > 64 * 1024:
        raise ValueError("rights attestation is too large")
    attestation = RightsAttestation.model_validate_json(attestation_bytes)
    if attestation.dataset_manifest_sha256 != manifest_digest:
        raise ValueError("rights attestation is not bound to this dataset manifest")

    records = _read_records(records_path, max_files)
    _validate_records(
        root,
        records,
        verify_hashes=verify_hashes,
        include_forgeries=include_forgeries,
        require_disjoint=require_disjoint,
        max_file_bytes=max_file_bytes,
        max_pixels=max_pixels,
        minimum_train_samples=minimum_train_samples,
    )
    return ValidatedManifest(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_digest,
        records_path=records_path,
        records_sha256=records_digest,
        records=tuple(records),
        split_counts=dict(Counter(r.split for r in records)),
        signer_counts={
            split: len({r.signer_id for r in records if r.split == split})
            for split in ("train", "validation", "test")
        },
        attestation_sha256=hashlib.sha256(attestation_bytes).hexdigest(),
        permitted_purpose=attestation.permitted_purpose,
        verify_on_open=verify_hashes,
        max_file_bytes=max_file_bytes,
        max_pixels=max_pixels,
    )


def _read_records(path: Path, max_files: int) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(line) > 64 * 1024:
                raise ValueError(f"manifest record line {line_number} is too long")
            if not line.strip():
                raise ValueError(f"blank manifest record at line {line_number}")
            records.append(SampleRecord.model_validate_json(line))
            if len(records) > max_files:
                raise ValueError("dataset contains more files than allowed")
    if not records:
        raise ValueError("dataset has no records")
    return records


def _validate_records(
    root: Path,
    records: list[SampleRecord],
    *,
    verify_hashes: bool,
    include_forgeries: bool,
    require_disjoint: bool,
    max_file_bytes: int,
    max_pixels: int,
    minimum_train_samples: int,
) -> None:
    sample_ids: set[str] = set()
    paths: set[str] = set()
    content_hashes: set[str] = set()
    signers: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    samples_per_signer: Counter[str] = Counter()
    for record in records:
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        relative = _safe_relative(record.image)
        normalized = relative.as_posix()
        folded = normalized.casefold()
        if folded in paths:
            raise ValueError(f"duplicate or case-colliding image path: {normalized}")
        if record.image_sha256 in content_hashes:
            raise ValueError(f"duplicate image content: {record.sample_id}")
        if record.kind == "forgery" and not include_forgeries:
            raise ValueError("forgeries are disabled by configuration")
        if abs(record.canonical_angle_degrees) > 1e-6:
            raise ValueError("this recipe requires upright zero-angle source images")
        if record.width * record.height > max_pixels:
            raise ValueError(f"declared pixels exceed limit: {record.sample_id}")
        image_path = _resolve_beneath(root, relative)
        _validate_image(image_path, record, verify_hashes, max_file_bytes, max_pixels)
        sample_ids.add(record.sample_id)
        paths.add(folded)
        content_hashes.add(record.image_sha256)
        signers[record.split].add(record.signer_id)
        source_splits[record.source_group].add(record.split)
        if record.split == "train" and record.kind == "genuine":
            samples_per_signer[record.signer_id] += 1

    if require_disjoint:
        split_names = ("train", "validation", "test")
        for index, left in enumerate(split_names):
            for right in split_names[index + 1 :]:
                overlap = signers[left] & signers[right]
                if overlap:
                    raise ValueError(f"signer leakage between {left} and {right}")
    if any(len(splits) > 1 for splits in source_splits.values()):
        raise ValueError("source_group leakage across splits")
    for signer, count in samples_per_signer.items():
        if count < minimum_train_samples:
            raise ValueError(f"train signer has fewer than K genuine samples: {signer}")
    if len(signers["train"]) < 2 or len(signers["validation"]) < 2:
        raise ValueError("train and validation each require at least two signers")
    validation_counts = Counter(r.signer_id for r in records if r.split == "validation")
    if any(count < 2 for count in validation_counts.values()):
        raise ValueError("every validation signer requires at least two samples")


def _validate_image(
    path: Path,
    record: SampleRecord,
    verify_hashes: bool,
    max_file_bytes: int,
    max_pixels: int,
) -> None:
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_file_bytes:
        raise ValueError(f"invalid image file size/type: {record.sample_id}")
    with path.open("rb") as handle:
        payload = handle.read(max_file_bytes + 1)
    if len(payload) > max_file_bytes:
        raise ValueError(f"image exceeds byte limit: {record.sample_id}")
    if verify_hashes and hashlib.sha256(payload).hexdigest() != record.image_sha256:
        raise ValueError(f"image digest mismatch: {record.sample_id}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != _ALLOWED_MEDIA[record.media_type]:
                raise ValueError(f"media type mismatch: {record.sample_id}")
            if getattr(image, "n_frames", 1) != 1 or getattr(image, "is_animated", False):
                raise ValueError(f"animated/multipage image rejected: {record.sample_id}")
            if image.width != record.width or image.height != record.height:
                raise ValueError(f"image dimensions mismatch: {record.sample_id}")
            if image.width * image.height > max_pixels:
                raise ValueError(f"decoded pixels exceed limit: {record.sample_id}")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image encoding: {record.sample_id}") from exc


def open_validated_image(
    root: Path,
    record: SampleRecord,
    *,
    verify_hash: bool,
    max_file_bytes: int,
    max_pixels: int,
) -> Image.Image:
    relative = _safe_relative(record.image)
    path = _resolve_beneath(root, relative)
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_file_bytes:
        raise ValueError(f"invalid image file size/type: {record.sample_id}")
    with path.open("rb") as handle:
        payload = handle.read(max_file_bytes + 1)
    if len(payload) > max_file_bytes:
        raise ValueError(f"image exceeds byte limit: {record.sample_id}")
    if verify_hash and hashlib.sha256(payload).hexdigest() != record.image_sha256:
        raise ValueError(f"image digest changed after preflight: {record.sample_id}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != _ALLOWED_MEDIA[record.media_type]:
                raise ValueError(f"media type mismatch: {record.sample_id}")
            if getattr(image, "n_frames", 1) != 1 or getattr(image, "is_animated", False):
                raise ValueError(f"animated/multipage image rejected: {record.sample_id}")
            if image.width != record.width or image.height != record.height:
                raise ValueError(f"image dimensions changed after preflight: {record.sample_id}")
            if image.width * image.height > max_pixels:
                raise ValueError(f"decoded pixels exceed limit: {record.sample_id}")
            image.load()
            return image.convert("L").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image encoding: {record.sample_id}") from exc


def _safe_relative(value: str) -> PurePosixPath:
    if any(ord(char) < 32 for char in value) or "\\" in value:
        raise ValueError("unsafe manifest path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("manifest paths must be normalized relative POSIX paths")
    return relative


def _resolve_beneath(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("manifest path escapes dataset root")
    _reject_symlink_chain(root, candidate)
    return resolved


def _reject_any_symlink(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinked input paths are rejected")
        if not current.exists():
            break


def _reject_symlink_chain(root: Path, candidate: Path) -> None:
    current = root
    if current.is_symlink():
        raise ValueError("dataset root cannot be a symlink")
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinked dataset paths are rejected")


def create_rights_attestation(
    manifest_path: Path,
    output_path: Path,
    *,
    permitted_purpose: Literal["research-only", "internal-approved", "commercial-approved"],
    authorization_reference: str,
    assert_authorized: bool,
) -> Path:
    if not assert_authorized:
        raise ValueError("--assert-authorized-use is required")
    manifest_path = manifest_path.resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError("rights attestation output already exists")
    if output_path.is_relative_to(repository):
        raise ValueError("rights attestations must be stored outside the source repository")
    payload = RightsAttestation(
        schema_version=1,
        dataset_manifest_sha256=sha256_file(manifest_path, 1024 * 1024),
        assertion="I_ASSERT_AUTHORIZED_USE_OF_THIS_DATASET",
        permitted_purpose=permitted_purpose,
        authorization_reference=authorization_reference,
        approved_at=datetime.now().astimezone(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload.model_dump_json(indent=2).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output_path
