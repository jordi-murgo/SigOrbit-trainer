"""Offline data sources and deterministic synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig, SamplerConfig, SplitConfig
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
class ResplitSource:
    """Wraps a source and serves an in-memory writer-level repartition."""

    base: ImageSource
    reassigned: tuple[SampleRecord, ...]
    split_fingerprint: str

    @property
    def fingerprint(self) -> str:
        combined = f"{self.base.fingerprint}|split:{self.split_fingerprint}"
        return "sha256:" + hashlib.sha256(combined.encode()).hexdigest()

    def records(self, split: str) -> tuple[SampleRecord, ...]:
        return tuple(
            record
            for record in self.reassigned
            if record.split == split and record.kind == "genuine"
        )

    def open_image(self, record: SampleRecord) -> Image.Image:
        return self.base.open_image(record)


def _signer_order(signer_id: str, seed: int) -> str:
    """Stable per-signer ordering key; independent of manifest order."""
    return hashlib.sha256(f"sigorbit-split-v1:{seed}:{signer_id}".encode()).hexdigest()


def _allocate(total: int, policy: SplitConfig) -> tuple[int, int, int]:
    """Largest-remainder allocation with at least two signers for train/validation."""
    if total < 4:
        raise ValueError("random_by_signer requires at least four distinct signers")
    fractions = (policy.train_fraction, policy.validation_fraction, policy.test_fraction)
    exact = [total * fraction for fraction in fractions]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    # Guarantee a usable protocol: train and validation must both be populated.
    while counts[0] < 2:
        donor = 2 if counts[2] > 0 else 1
        if counts[donor] <= (2 if donor == 1 else 0):
            raise ValueError("split fractions leave too few signers for training")
        counts[donor] -= 1
        counts[0] += 1
    while counts[1] < 2:
        donor = 2 if counts[2] > 0 else 0
        if counts[donor] <= (2 if donor == 0 else 0):
            raise ValueError("split fractions leave too few signers for validation")
        counts[donor] -= 1
        counts[1] += 1
    return counts[0], counts[1], counts[2]


def apply_split_policy(
    records: tuple[SampleRecord, ...], policy: SplitConfig
) -> tuple[tuple[SampleRecord, ...], dict[str, object]]:
    """Reassign whole signers to splits in memory; the manifest is never rewritten."""
    if policy.strategy == "manifest":
        return records, {"strategy": "manifest"}

    signer_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        signer_groups[record.signer_id].append(record)
    # A source group crossing signers cannot be kept intact by a writer-level split.
    source_owners: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_owners[record.source_group].add(record.signer_id)
    straddling = sorted(group for group, owners in source_owners.items() if len(owners) > 1)
    if straddling:
        raise ValueError(
            "random_by_signer requires each source_group to belong to a single signer; "
            f"{len(straddling)} group(s) span multiple signers"
        )

    signers = sorted(signer_groups)
    ordered = sorted(signers, key=lambda signer: (_signer_order(signer, policy.seed), signer))
    train_count, validation_count, test_count = _allocate(len(ordered), policy)
    assignment: dict[str, str] = {}
    for index, signer in enumerate(ordered):
        if index < train_count:
            assignment[signer] = "train"
        elif index < train_count + validation_count:
            assignment[signer] = "validation"
        else:
            assignment[signer] = "test"

    reassigned = tuple(
        record.model_copy(update={"split": assignment[record.signer_id]}) for record in records
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {"seed": policy.seed, "assignment": assignment},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary: dict[str, object] = {
        "strategy": "random_by_signer",
        "seed": policy.seed,
        "signers": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
        "assignment_sha256": fingerprint,
    }
    return reassigned, summary


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
            enforce_split_layout=config.split.strategy == "manifest",
        )
        source = LocalManifestSource(validated)
        permitted_purpose = validated.permitted_purpose
        attestation_sha256 = validated.attestation_sha256

    all_records = tuple(
        record for split in ("train", "validation", "test") for record in source.records(split)
    )
    reassigned, split_summary = apply_split_policy(all_records, config.split)
    if config.split.strategy != "manifest":
        source = ResplitSource(
            base=source,
            reassigned=reassigned,
            split_fingerprint=str(split_summary["assignment_sha256"]),
        )
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
        "split": split_summary,
    }
    return DatasetBundle(
        source=source,
        train_records=train_records,
        validation_records=validation_records,
        class_map=class_map,
        class_map_sha256=summary["class_map_sha256"],  # type: ignore[arg-type]
        summary=summary,
    )
