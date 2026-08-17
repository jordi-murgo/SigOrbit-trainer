import json
import shutil
from pathlib import Path

import pytest
from sigorbit import load_model

from sigorbit_trainer.config import load_config
from sigorbit_trainer.engine import (
    _joint_patience_exhausted,
    evaluate_checkpoint,
    resume_training,
    run_training,
)


def test_three_stage_synthetic_smoke(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(source)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "run"})}
    )
    result = run_training(config)
    assert result.checkpoint.is_file()
    assert result.sha256
    assert (result.output_dir / "checkpoints" / "best-backbone.json").is_file()
    assert (result.output_dir / "checkpoints" / "best-pose.json").is_file()
    assert (result.output_dir / "checkpoints" / "best-joint.json").is_file()
    checkpoint_root = result.output_dir / "checkpoints"
    retained = {
        json.loads(pointer.read_text())["checkpoint"] for pointer in checkpoint_root.glob("*.json")
    }
    directories = {path.name for path in checkpoint_root.iterdir() if path.is_dir()}
    assert directories == retained
    assert len(directories) <= 4
    model, info = load_model(
        result.checkpoint,
        expected_sha256=result.sha256,
        strict_release_config=False,
    )
    assert model.config.input_size == 33
    assert info.model_id == "sigorbit-synthetic-smoke"


def test_joint_patience_waits_for_minimum_epochs() -> None:
    policy = {"bad_epochs": 16, "min_epochs": 50, "patience": 16}
    assert not _joint_patience_exhausted(completed_epochs=49, **policy)
    assert _joint_patience_exhausted(completed_epochs=50, **policy)
    assert not _joint_patience_exhausted(
        completed_epochs=80, bad_epochs=15, min_epochs=50, patience=16
    )


def test_resume_from_latest_epoch_boundary(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(source)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "resume-run"})}
    )
    initial = run_training(config)
    run_path = initial.output_dir / "run.json"
    run_record = json.loads(run_path.read_text())
    run_record["status"] = "running"
    run_path.write_text(json.dumps(run_record))
    pointer = json.loads((initial.output_dir / "checkpoints" / "latest.json").read_text())
    checkpoint = initial.output_dir / "checkpoints" / pointer["checkpoint"]
    resumed = resume_training(config, checkpoint=checkpoint)
    assert resumed.metrics["global_step"] == initial.metrics["global_step"]
    assert resumed.checkpoint.is_file()


def test_mid_pipeline_resume_reproduces_the_uninterrupted_artifact(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(source)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "crash-run"})}
    )
    reference = run_training(config)

    # Roll the completed run back to its backbone boundary to emulate a crash.
    checkpoint_root = reference.output_dir / "checkpoints"
    backbone = sorted(
        path
        for path in checkpoint_root.iterdir()
        if path.name.startswith("stage=backbone-epoch=0000")
    )[0]
    metadata = json.loads((backbone / "checkpoint.json").read_text())
    for later in [path for path in checkpoint_root.iterdir() if path.is_dir() and path != backbone]:
        shutil.rmtree(later)
    pointer = json.dumps(
        {
            "checkpoint": backbone.name,
            "tensor_sha256": metadata["tensor_sha256"],
            "metadata_sha256": metadata["metadata_sha256"],
        }
    )
    (checkpoint_root / "latest.json").write_text(pointer)
    (checkpoint_root / "best-backbone.json").write_text(pointer)
    for stale in ("best-pose.json", "best-joint.json"):
        (checkpoint_root / stale).unlink(missing_ok=True)
    run_path = reference.output_dir / "run.json"
    run_record = json.loads(run_path.read_text())
    run_record["status"] = "running"
    run_path.write_text(json.dumps(run_record))

    resumed = resume_training(config, checkpoint=backbone)
    assert resumed.sha256 == reference.sha256
    assert resumed.metrics["global_step"] == reference.metrics["global_step"]
    assert resumed.metrics["clean"] == reference.metrics["clean"]


def test_resume_rejects_a_foreign_checkpoint(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(source)
    first = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "run-a"})}
    )
    second = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "run-b"})}
    )
    run_training(first)
    other = run_training(second)
    pointer = json.loads((other.output_dir / "checkpoints" / "latest.json").read_text())
    foreign = other.output_dir / "checkpoints" / pointer["checkpoint"]
    with pytest.raises(ValueError, match="configured run directory"):
        resume_training(first, checkpoint=foreign)


def test_per_angle_retrieval_and_test_split_guard(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(source)
    config = config.model_copy(
        update={"run": config.run.model_copy(update={"output_dir": tmp_path / "eval-run"})}
    )
    result = run_training(config)
    angles = [str(float(angle)) for angle in config.evaluation.rotation_angles_degrees]
    assert sorted(result.metrics["rotation_retrieval"]) == sorted(angles)
    for report in result.metrics["rotation_retrieval"].values():
        assert {"top1", "top5", "median_margin", "fragile_percent", "count"} <= set(report)

    evaluation = evaluate_checkpoint(
        config,
        checkpoint=result.checkpoint,
        split="validation",
        allow_test_split=False,
    )
    assert evaluation["split"] == "validation"
    assert evaluation["claim_status"] == "synthetic-smoke"

    with pytest.raises(ValueError, match="release-candidate only"):
        evaluate_checkpoint(
            config,
            checkpoint=result.checkpoint,
            split="test",
            allow_test_split=False,
        )
