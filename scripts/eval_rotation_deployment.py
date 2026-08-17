#!/usr/bin/env python3
"""Rotation evaluation in the shape a deployment actually sees.

`sigorbit_trainer.metrics.rotation_retrieval`, which the trainer reports at the
end of a run, rotates the *whole gallery* -- queries and references together. A
rotation-equivariant model satisfies that almost for free, so it reports ~99.8%
at every angle and tells you very little. Enrolment stores clean references and
the query arrives at whatever angle the page was scanned, so only the asymmetric
case predicts field accuracy.

Two rotation protocols, because they answer different questions:

  model-canvas  pad to a square first, then rotate in place. No ink leaves the
                frame, so this isolates orientation.
  raw-expand    rotate with `expand=True`, then resize to the model input. Ink
                is preserved but framing, aspect and scale all move with the
                angle. This is what a real re-photographed note looks like.

Do NOT rotate a signature inside its own rectangular frame without padding: a
wide signature loses its ends, top-1 collapses (4.8% at 90 degrees in our
measurements) and recovers at 180 degrees where a rectangle fits its own bounds
again. That measures cropping, not orientation.

Usage:
  eval_rotation_deployment.py --checkpoint CKPT --manifest DATASET.toml
  eval_rotation_deployment.py ... --split test --angles 0,15,30,45,90,180
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sigorbit import load_model
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="dataset.toml directory root")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--angles", default="0,5,10,15,20,30,45,60,90,135,180")
    parser.add_argument("--input-size", type=int, default=257)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pad_to_square(image: Image.Image) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(image, ((side - image.size[0]) // 2, (side - image.size[1]) // 2))
    return canvas


def main() -> int:
    args = parse_args()
    angles = [float(a) for a in args.angles.split(",")]
    device = torch.device(args.device)

    model, info = load_model(args.checkpoint, device=device)
    model.eval()

    root = args.manifest.parent if args.manifest.is_file() else args.manifest
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    records = [r for r in samples if r["split"] == args.split and r["kind"] == "genuine"]
    if not records:
        raise SystemExit(f"no genuine records in split {args.split!r}")

    normalise = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def prepare(image: Image.Image) -> torch.Tensor:
        return normalise(image.resize((args.input_size, args.input_size), Image.BICUBIC))

    @torch.no_grad()
    def embed(images: list[Image.Image]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(images), args.batch_size):
            batch = torch.stack([prepare(im) for im in images[start : start + args.batch_size]])
            output = model(batch.to(device))
            if isinstance(output, tuple):
                output = output[0]
            chunks.append(F.normalize(output.float(), dim=1).cpu())
        return torch.cat(chunks)

    originals = [Image.open(root / r["image"]).convert("L") for r in records]
    signer_ids = [r["signer_id"] for r in records]
    references = embed(originals).numpy()

    signers = sorted(set(signer_ids))
    per_signer: dict[str, list[int]] = defaultdict(list)
    for index, signer in enumerate(signer_ids):
        per_signer[signer].append(index)
    truth = np.array([signers.index(s) for s in signer_ids])
    rows = np.arange(len(truth))

    def retrieve(queries: torch.Tensor) -> tuple[float, float]:
        similarity = queries.numpy() @ references.T
        # A query must never match its own clean embedding.
        np.fill_diagonal(similarity, -np.inf)
        by_signer = np.stack([similarity[:, per_signer[s]].max(axis=1) for s in signers], axis=1)
        genuine = by_signer[rows, truth]
        impostor = by_signer.copy()
        impostor[rows, truth] = -np.inf
        margins = genuine - impostor.max(axis=1)
        ranks = (np.argsort(-by_signer, axis=1) == truth[:, None]).argmax(axis=1) + 1
        return float((ranks == 1).mean() * 100.0), float(np.median(margins))

    squares = [pad_to_square(im) for im in originals]
    print(f"checkpoint {info.sha256[:16]}  split {args.split}  n={len(records)}")
    print("references are clean; only the query is rotated\n")
    print(f"{'angle':>6}  {'model-canvas':>20}  {'raw-expand':>20}")
    print(f"{'':>6}  {'top-1    margin':>20}  {'top-1    margin':>20}")
    for angle in angles:
        canvas = [im.rotate(angle, resample=Image.BICUBIC, fillcolor=255) for im in squares]
        expand = [
            im.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=255) for im in originals
        ]
        canvas_top1, canvas_margin = retrieve(embed(canvas))
        expand_top1, expand_margin = retrieve(embed(expand))
        print(
            f"{angle:>6.0f}  {canvas_top1:>7.1f}%  {canvas_margin:+.4f}  "
            f"{expand_top1:>11.1f}%  {expand_margin:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
