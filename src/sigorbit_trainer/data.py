"""Offline data sources and deterministic synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig, SamplerConfig
from .manifest import SampleRecord, ValidatedManifest, open_validated_image, validate_manifest


class ImageSource(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def records(self, split: str) -> tuple[SampleRecord, ...]: ...

    def open_image(self, record: SampleRecord) -> Image.Image: ...


@dataclass(frozen=True)
class LocalManifestSource:
    validated: ValidatedManifest

    @property
    def fingerprint(self) -> str:
        return self.validated.manifest_sha256

    def records(self, split: str) -> tuple[SampleRecord, ...]:
        return tuple(
            record
            for record in self.validated.records
            if record.split == split and record.kind == "genuine"
        )

    def open_image(self, record: SampleRecord) -> Image.Image:
        return open_validated_image(
            self.validated.root,
            record,
            verify_hash=self.validated.verify_on_open,
            max_file_bytes=self.validated.max_file_bytes,
            max_pixels=self.validated.max_pixels,
        )


@dataclass(frozen=True)
class SyntheticSource:
    train_signers: int
    validation_signers: int
    samples_per_signer: int
    width: int
    height: int
    seed: int

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()
        return "synthetic:" + hashlib.sha256(payload).hexdigest()

    def records(self, split: str) -> tuple[SampleRecord, ...]:
        if split == "train":
            count, offset = self.train_signers, 0
        elif split == "validation":
            count, offset = self.validation_signers, self.train_signers
        else:
            return ()
        result: list[SampleRecord] = []
        for signer_index in range(count):
            global_signer = signer_index + offset
            for sample_index in range(self.samples_per_signer):
                token = f"synthetic-v1:{self.seed}:{split}:{global_signer}:{sample_index}"
                digest = hashlib.sha256(token.encode()).hexdigest()
                result.append(
                    SampleRecord(
                        sample_id=f"synthetic:{digest[:24]}",
                        image=f"synthetic/{digest}.png",
                        image_sha256=digest,
                        media_type="image/png",
                        width=self.width,
                        height=self.height,
                        signer_id=f"synthetic-{split}-{signer_index:04d}",
                        split=split,  # type: ignore[arg-type]
                        kind="genuine",
                        source_group=f"synthetic:{digest[:24]}",
                        canonical_angle_degrees=0.0,
                    )
                )
        return tuple(result)

    def open_image(self, record: SampleRecord) -> Image.Image:
        seed_material = f"{self.seed}:{record.signer_id}:{record.sample_id}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        signer_seed = int.from_bytes(hashlib.sha256(record.signer_id.encode()).digest()[:8], "big")
        signer_rng = random.Random(signer_seed)
        sample_rng = random.Random(sample_seed)
        image = Image.new("L", (self.width, self.height), 255)
        draw = ImageDraw.Draw(image)
        points: list[tuple[int, int]] = []
        phases = [signer_rng.uniform(0, math.tau) for _ in range(3)]
        for index in range(12):
            x = int((index + 1) * self.width / 13 + sample_rng.uniform(-1.5, 1.5))
            y_base = self.height * (0.5 + 0.18 * math.sin(index * 0.72 + phases[0]))
            y_base += self.height * 0.08 * math.sin(index * 1.37 + phases[1])
            y = int(y_base + sample_rng.uniform(-1.2, 1.2))
            points.append((x, max(2, min(self.height - 3, y))))
        draw.line(points, fill=25, width=max(1, self.height // 32), joint="curve")
        for loop in range(2):
            cx = int(self.width * (0.35 + loop * 0.3) + signer_rng.uniform(-5, 5))
            cy = int(self.height * 0.5 + signer_rng.uniform(-4, 4))
            radius = int(self.height * (0.10 + signer_rng.uniform(0.0, 0.04)))
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius * 2, cy + radius), outline=35, width=1
            )
        return image


@dataclass(frozen=True)
class DatasetBundle:
    source: ImageSource
    train_records: tuple[SampleRecord, ...]
    validation_records: tuple[SampleRecord, ...]
    class_map: dict[str, int]
    class_map_sha256: str
    summary: dict[str, object]


class SignatureDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        source: ImageSource,
        records: tuple[SampleRecord, ...],
        class_map: dict[str, int],
        transform: Callable[[Image.Image], Tensor],
        *,
        run_seed: int,
        stage: str,
        epoch: int,
    ) -> None:
        self.source = source
        self.records = records
        self.class_map = class_map
        self.transform = transform
        self.run_seed = run_seed
        self.stage = stage
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        import torch

        from .augment import stateless_seed

        record = self.records[index]
        seed = stateless_seed(self.run_seed, self.stage, self.epoch, record.sample_id)
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            tensor = self.transform(self.source.open_image(record))
        finally:
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
        return tensor, self.class_map[record.signer_id]


def load_data(config: DataConfig, sampler: SamplerConfig, seed: int) -> DatasetBundle:
    if config.include_forgeries:
        raise ValueError("forgery training is not implemented; include_forgeries must be false")
    if config.kind == "synthetic":
        assert config.synthetic is not None
        source: ImageSource = SyntheticSource(
            train_signers=config.synthetic.train_signers,
            validation_signers=config.synthetic.validation_signers,
            samples_per_signer=config.synthetic.samples_per_signer,
            width=config.synthetic.canvas_width,
            height=config.synthetic.canvas_height,
            seed=seed,
        )
        permitted_purpose = "synthetic-only"
        attestation_sha256 = None
    else:
        assert config.manifest is not None and config.rights_attestation is not None
        validated = validate_manifest(
            config.manifest,
            config.rights_attestation,
            verify_hashes=config.verify_hashes,
            include_forgeries=config.include_forgeries,
            require_disjoint=config.require_signer_disjoint_splits,
            max_files=config.max_files,
            max_file_bytes=config.max_file_bytes,
            max_pixels=config.max_pixels,
            minimum_train_samples=sampler.samples_per_person,
        )
        source = LocalManifestSource(validated)
        permitted_purpose = validated.permitted_purpose
        attestation_sha256 = validated.attestation_sha256

    train_records = source.records("train")
    validation_records = source.records("validation")
    if not train_records or not validation_records:
        raise ValueError("train and validation splits are required")
    train_signers = sorted({record.signer_id for record in train_records})
    validation_signers = {record.signer_id for record in validation_records}
    if config.require_signer_disjoint_splits and set(train_signers) & validation_signers:
        raise ValueError("train and validation signer sets overlap")
    if len(train_signers) < sampler.persons_per_batch:
        raise ValueError("persons_per_batch exceeds the number of training signers")
    counts = Counter(record.signer_id for record in train_records)
    if min(counts.values()) < sampler.samples_per_person:
        raise ValueError("each train signer must have at least K samples")
    validation_counts = Counter(record.signer_id for record in validation_records)
    if min(validation_counts.values()) < 2:
        raise ValueError("each validation signer must have at least two samples")

    class_map = {signer: index for index, signer in enumerate(train_signers)}
    class_bytes = json.dumps(class_map, sort_keys=True, separators=(",", ":")).encode()
    summary: dict[str, object] = {
        "source_fingerprint": source.fingerprint,
        "train_samples": len(train_records),
        "train_signers": len(train_signers),
        "validation_samples": len(validation_records),
        "validation_signers": len(validation_signers),
        "class_map_sha256": hashlib.sha256(class_bytes).hexdigest(),
        "attestation_sha256": attestation_sha256,
        "permitted_purpose": permitted_purpose,
    }
    return DatasetBundle(
        source=source,
        train_records=train_records,
        validation_records=validation_records,
        class_map=class_map,
        class_map_sha256=summary["class_map_sha256"],  # type: ignore[arg-type]
        summary=summary,
    )
