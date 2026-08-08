#!/usr/bin/env python3
"""Fail closed if tracked/package files contain training artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

DENIED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".npy",
    ".npz",
    ".db",
    ".sqlite",
    ".arrow",
    ".parquet",
    ".pkl",
    ".pickle",
    ".joblib",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
}
DENIED_ROOTS = {"data", "datasets", "runs", "outputs", "checkpoints", "artifacts"}
MAX_TRACKED_BYTES = 2 * 1024 * 1024


def check_name(name: str) -> None:
    normalized = name.replace("\\", "/").lstrip("./")
    parts = tuple(part for part in normalized.split("/") if part)
    if Path(normalized).suffix.lower() in DENIED_SUFFIXES:
        raise SystemExit(f"prohibited artifact extension: {name}")
    if parts and parts[0].lower() in DENIED_ROOTS:
        raise SystemExit(f"prohibited artifact root: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git executable not found")
    tracked = subprocess.run(  # noqa: S603 -- resolved local Git executable, fixed arguments
        [git, "ls-files", "-z"], check=True, capture_output=True
    ).stdout.split(b"\0")
    for raw in tracked:
        if not raw:
            continue
        path = Path(raw.decode())
        check_name(path.as_posix())
        if path.stat().st_size > MAX_TRACKED_BYTES:
            raise SystemExit(f"tracked file exceeds size limit: {path}")
        if path.is_file() and path.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec"
        ):
            raise SystemExit(f"Git LFS pointer rejected: {path}")
    for archive in args.archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as package:
                for name in package.namelist():
                    check_name(name)
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as package:
                for member in package.getmembers():
                    check_name(member.name)
        else:
            raise SystemExit(f"unknown distribution archive: {archive}")
    print("release contents: code-only")


if __name__ == "__main__":
    main()
