"""Optional offline adapters that materialize the core local-manifest contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


def import_hf_disk(
    source: Path,
    output: Path,
    *,
    dataset_id: str,
    revision: str,
    assert_genuine_only: bool,
    max_files: int = 100_000,
    max_pixels: int = 16_777_216,
) -> dict[str, Any]:
    if not assert_genuine_only:
        raise ValueError("--assert-genuine-only is required; inspect the source before importing")
    source = source.resolve(strict=True)
    output = output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError("materialization output already exists")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and materialization output must be separate")
    if output.is_relative_to(repository):
        raise ValueError("dataset materialization must be outside the source repository")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", dataset_id):
        raise ValueError("dataset_id must match the manifest identifier pattern")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9:._-]{0,255}", revision):
        raise ValueError("revision must be an immutable identifier without quotes or newlines")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("install sigorbit-trainer[hf] for local HF materialization") from exc
    loaded = load_from_disk(str(source))
    if not isinstance(loaded, DatasetDict):
        raise ValueError("expected a local DatasetDict")
    required = {"train", "validation"}
    if not required.issubset(loaded):
        raise ValueError("local DatasetDict requires train and validation splits")
    for split in loaded:
        columns = set(loaded[split].column_names)
        if not {"image", "label"}.issubset(columns):
            raise ValueError(f"split {split} lacks image/label columns")

    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    records: list[dict[str, Any]] = []
    content_hashes: set[str] = set()
    try:
        for split in ("train", "validation", "test"):
            if split not in loaded:
                continue
            rows = loaded[split]
            for index, row in enumerate(rows):
                if len(records) >= max_files:
                    raise ValueError("source exceeds max_files")
                image = _coerce_image(row["image"])
                if image.width * image.height > max_pixels:
                    raise ValueError(f"source image exceeds max_pixels at {split}/{index}")
                sample_token = hashlib.sha256(
                    f"{dataset_id}:{revision}:{split}:{index}".encode()
                ).hexdigest()
                relative = Path("images") / split / sample_token[:2] / f"{sample_token}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.convert("L").save(destination, format="PNG", optimize=False)
                destination.chmod(0o600)
                image_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if image_digest in content_hashes:
                    raise ValueError(f"duplicate decoded image content at {split}/{index}")
                content_hashes.add(image_digest)
                label_token = hashlib.sha256(
                    f"{dataset_id}:{revision}:writer:{row['label']}".encode()
                ).hexdigest()
                records.append(
                    {
                        "sample_id": f"sha256:{sample_token}",
                        "image": relative.as_posix(),
                        "image_sha256": image_digest,
                        "media_type": "image/png",
                        "width": image.width,
                        "height": image.height,
                        "signer_id": f"writer-{label_token[:24]}",
                        "split": split,
                        "kind": "genuine",
                        "source_group": f"source-{sample_token}",
                        "canonical_angle_degrees": 0.0,
                    }
                )
        records_path = output / "samples.jsonl"
        records_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        records_path.chmod(0o600)
        records_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
        manifest_path = output / "dataset.toml"
        manifest_path.write_text(
            "schema_version = 1\n"
            f'dataset_id = "{dataset_id}"\n'
            f'revision = "{revision}"\n'
            'records = "samples.jsonl"\n'
            f'records_sha256 = "{records_sha}"\n'
            'canonical_pose = "upright"\n',
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)
        return {
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "records": len(records),
            "splits": {split: len(loaded[split]) for split in loaded},
            "rights_attestation_created": False,
        }
    except BaseException:
        import shutil

        shutil.rmtree(output, ignore_errors=True)
        raise


def _coerce_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        value.load()
        return value.copy()
    if isinstance(value, dict) and value.get("bytes") is not None:
        import io

        with Image.open(io.BytesIO(value["bytes"])) as image:
            image.load()
            return image.copy()
    raise ValueError("unsupported local HF image representation")
