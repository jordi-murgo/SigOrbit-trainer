from pathlib import Path

import pytest

from sigorbit_trainer.config import SamplerConfig, load_config
from sigorbit_trainer.data import LocalManifestSource, SyntheticSource, load_data
from sigorbit_trainer.manifest import validate_manifest


def test_synthetic_images_are_deterministic() -> None:
    source = SyntheticSource(2, 2, 2, 96, 48, 7)
    record = source.records("train")[0]
    assert source.open_image(record).tobytes() == source.open_image(record).tobytes()
    assert not set(r.signer_id for r in source.records("train")) & set(
        r.signer_id for r in source.records("validation")
    )


def test_manifest_validation(local_manifest: tuple[Path, Path]) -> None:
    manifest, attestation = local_manifest
    validated = validate_manifest(
        manifest,
        attestation,
        verify_hashes=True,
        include_forgeries=False,
        require_disjoint=True,
        max_files=100,
        max_file_bytes=1024 * 1024,
        max_pixels=4096,
        minimum_train_samples=2,
    )
    assert validated.split_counts == {"train": 4, "validation": 4}
    assert validated.signer_counts["train"] == 2


def test_manifest_rejects_symlink(local_manifest: tuple[Path, Path]) -> None:
    manifest, attestation = local_manifest
    image = next((manifest.parent / "images").rglob("*.png"))
    target = image.with_name("target.png")
    image.rename(target)
    image.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        validate_manifest(
            manifest,
            attestation,
            verify_hashes=False,
            include_forgeries=False,
            require_disjoint=True,
            max_files=100,
            max_file_bytes=1024 * 1024,
            max_pixels=4096,
            minimum_train_samples=2,
        )


def test_smoke_data_satisfies_pk() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.toml")
    bundle = load_data(config.data, SamplerConfig(persons_per_batch=2, samples_per_person=2), 0)
    assert len(bundle.class_map) == 2


def test_manifest_source_detects_post_preflight_mutation(
    local_manifest: tuple[Path, Path],
) -> None:
    manifest, attestation = local_manifest
    validated = validate_manifest(
        manifest,
        attestation,
        verify_hashes=True,
        include_forgeries=False,
        require_disjoint=True,
        max_files=100,
        max_file_bytes=1024 * 1024,
        max_pixels=4096,
        minimum_train_samples=2,
    )
    record = validated.records[0]
    (validated.root / record.image).write_bytes(b"not the approved image")
    with pytest.raises(ValueError, match="digest changed"):
        LocalManifestSource(validated).open_image(record)
