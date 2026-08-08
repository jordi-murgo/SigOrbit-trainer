"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import TrainerConfig, config_sha256, load_config
from .engine import evaluate_checkpoint, resume_training, run_training
from .manifest import create_rights_attestation, validate_manifest
from .materialize import import_hf_disk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigorbit-train", description="Offline SigOrbit training")
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="emit machine-readable output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="configuration operations")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_validate = config_sub.add_parser("validate", help="validate strict TOML")
    config_validate.add_argument("path", type=Path)
    config_schema = config_sub.add_parser("schema", help="write JSON Schema")
    config_schema.add_argument("--output", type=Path)

    dataset_parser = subparsers.add_parser("dataset", help="dataset operations")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_sub.add_parser("validate", help="validate a local manifest")
    dataset_validate.add_argument("manifest", type=Path)
    dataset_validate.add_argument("--attestation", required=True, type=Path)
    dataset_validate.add_argument("--p", type=int, default=8)
    dataset_validate.add_argument("--k", type=int, default=4)
    dataset_validate.add_argument("--no-hash-verification", action="store_true")
    dataset_import = dataset_sub.add_parser(
        "import-hf-disk", help="materialize an already-local HF DatasetDict"
    )
    dataset_import.add_argument("source", type=Path)
    dataset_import.add_argument("output", type=Path)
    dataset_import.add_argument("--dataset-id", required=True)
    dataset_import.add_argument("--revision", required=True)
    dataset_import.add_argument("--assert-genuine-only", action="store_true")
    dataset_attest = dataset_sub.add_parser(
        "attest", help="create an untracked assertion bound to a local manifest"
    )
    dataset_attest.add_argument("manifest", type=Path)
    dataset_attest.add_argument("output", type=Path)
    dataset_attest.add_argument(
        "--purpose",
        choices=("research-only", "internal-approved", "commercial-approved"),
        required=True,
    )
    dataset_attest.add_argument("--authorization-reference", required=True)
    dataset_attest.add_argument("--assert-authorized-use", action="store_true")

    run_parser = subparsers.add_parser("run", help="start a new three-stage run")
    run_parser.add_argument("config", type=Path)
    resume_parser = subparsers.add_parser("resume", help="resume at an epoch boundary")
    resume_parser.add_argument("config", type=Path)
    resume_parser.add_argument("--checkpoint", required=True, type=Path)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate an exported checkpoint on a manifest split"
    )
    evaluate_parser.add_argument("config", type=Path)
    evaluate_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_parser.add_argument("--allow-test-split", action="store_true")

    checkpoint_parser = subparsers.add_parser("checkpoint", help="recovery checkpoint operations")
    checkpoint_sub = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_inspect = checkpoint_sub.add_parser(
        "inspect", help="verify and inspect recovery state"
    )
    checkpoint_inspect.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "config" and args.config_command == "validate":
            config = load_config(args.path)
            return _emit({"valid": True, "config_sha256": config_sha256(config)}, args.json_output)
        if args.command == "config" and args.config_command == "schema":
            schema = TrainerConfig.model_json_schema()
            if args.output:
                args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
                return _emit({"written": str(args.output)}, args.json_output)
            print(json.dumps(schema, indent=2, sort_keys=True))
            return 0
        if args.command == "dataset" and args.dataset_command == "validate":
            validated = validate_manifest(
                args.manifest,
                args.attestation,
                verify_hashes=not args.no_hash_verification,
                include_forgeries=False,
                require_disjoint=True,
                max_files=100_000,
                max_file_bytes=16 * 1024 * 1024,
                max_pixels=16_777_216,
                minimum_train_samples=args.k,
            )
            if validated.signer_counts.get("train", 0) < args.p:
                raise ValueError("P exceeds manifest training signer count")
            return _emit(
                {
                    "valid": True,
                    "manifest_sha256": validated.manifest_sha256,
                    "split_counts": validated.split_counts,
                    "signer_counts": validated.signer_counts,
                    "permitted_purpose": validated.permitted_purpose,
                },
                args.json_output,
            )
        if args.command == "dataset" and args.dataset_command == "attest":
            written = create_rights_attestation(
                args.manifest,
                args.output,
                permitted_purpose=args.purpose,
                authorization_reference=args.authorization_reference,
                assert_authorized=args.assert_authorized_use,
            )
            return _emit({"written": str(written)}, args.json_output)
        if args.command == "dataset" and args.dataset_command == "import-hf-disk":
            materialized = import_hf_disk(
                args.source,
                args.output,
                dataset_id=args.dataset_id,
                revision=args.revision,
                assert_genuine_only=args.assert_genuine_only,
            )
            return _emit(materialized, args.json_output)
        if args.command == "run":
            config = load_config(args.config)
            run_result = run_training(config, source_config=args.config)
            return _emit(
                {
                    "status": "completed",
                    "output_dir": str(run_result.output_dir),
                    "checkpoint": str(run_result.checkpoint),
                    "sha256": run_result.sha256,
                    "metrics": run_result.metrics,
                },
                args.json_output,
            )
        if args.command == "resume":
            config = load_config(args.config)
            resumed_result = resume_training(config, checkpoint=args.checkpoint)
            return _emit(
                {
                    "status": "completed",
                    "output_dir": str(resumed_result.output_dir),
                    "checkpoint": str(resumed_result.checkpoint),
                    "sha256": resumed_result.sha256,
                    "metrics": resumed_result.metrics,
                },
                args.json_output,
            )
        if args.command == "evaluate":
            config = load_config(args.config)
            report = evaluate_checkpoint(
                config,
                checkpoint=args.checkpoint,
                split=args.split,
                allow_test_split=args.allow_test_split,
            )
            return _emit(report, args.json_output)
        if args.command == "checkpoint" and args.checkpoint_command == "inspect":
            loaded = load_checkpoint(args.path)
            return _emit(loaded.metadata.model_dump(mode="json"), args.json_output)
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {"error": type(exc).__name__, "message": str(exc)}
        if args.json_output:
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error("unhandled command")
    return 2


def _emit(payload: object, json_output: bool) -> int:
    if json_output:
        print(json.dumps(payload, sort_keys=True))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
