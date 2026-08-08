from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sigorbit_trainer.config import RIGHTS_ASSERTION


@pytest.fixture
def local_manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    records = []
    index = 0
    for split, signer_count in (("train", 2), ("validation", 2)):
        for signer in range(signer_count):
            for sample in range(2):
                relative = Path("images") / split / f"s{signer}-{sample}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("L", (64, 32), 255)
                draw = ImageDraw.Draw(image)
                draw.line(
                    [(5, 8 + signer * 4), (30, 20 + sample), (58, 9 + signer + sample)],
                    fill=10 + index,
                    width=2,
                )
                image.save(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(
                    {
                        "sample_id": f"sample-{index}",
                        "image": relative.as_posix(),
                        "image_sha256": digest,
                        "media_type": "image/png",
                        "width": 64,
                        "height": 32,
                        "signer_id": f"{split}-signer-{signer}",
                        "split": split,
                        "kind": "genuine",
                        "source_group": f"source-{index}",
                        "canonical_angle_degrees": 0.0,
                    }
                )
                index += 1
    records_path = root / "samples.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    records_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest = root / "dataset.toml"
    manifest.write_text(
        "schema_version = 1\n"
        'dataset_id = "test-signatures"\n'
        'revision = "test-v1"\n'
        'records = "samples.jsonl"\n'
        f'records_sha256 = "{records_sha}"\n'
        'canonical_pose = "upright"\n'
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    attestation = tmp_path / "rights.attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_manifest_sha256": manifest_sha,
                "assertion": RIGHTS_ASSERTION,
                "permitted_purpose": "research-only",
                "authorization_reference": "test-only",
                "approved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    return manifest, attestation
