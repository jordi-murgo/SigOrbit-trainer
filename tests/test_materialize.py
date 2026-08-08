from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sigorbit_trainer.manifest import create_rights_attestation, validate_manifest
from sigorbit_trainer.materialize import import_hf_disk


def test_import_already_local_hf_dataset(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")

    def rows(offset: int) -> object:
        images = []
        labels = []
        for signer in range(2):
            for sample in range(2):
                image = Image.new("L", (48, 24), 255)
                draw = ImageDraw.Draw(image)
                draw.line(
                    [(2, 4 + signer), (20, 15 + sample), (45, 7 + signer + sample)],
                    fill=10 + offset + signer * 2 + sample,
                    width=2,
                )
                images.append(image)
                labels.append(offset + signer)
        return datasets.Dataset.from_dict({"image": images, "label": labels})

    source = tmp_path / "source"
    dataset = datasets.DatasetDict({"train": rows(0), "validation": rows(10)})
    dataset.save_to_disk(source)
    output = tmp_path / "materialized"
    result = import_hf_disk(
        source,
        output,
        dataset_id="authorized-test",
        revision="immutable-test-revision",
        assert_genuine_only=True,
    )
    manifest = Path(result["manifest"])
    attestation = create_rights_attestation(
        manifest,
        tmp_path / "rights.attestation.json",
        permitted_purpose="research-only",
        authorization_reference="pytest",
        assert_authorized=True,
    )
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
