import json
from pathlib import Path

import pytest
from sigorbit import load_model

from sigorbit_trainer.config import load_config
from sigorbit_trainer.engine import evaluate_checkpoint, resume_training, run_training


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
    model, info = load_model(
        result.checkpoint,
        expected_sha256=result.sha256,
        strict_release_config=False,
    )
    assert model.config.input_size == 33
    assert info.model_id == "sigorbit-synthetic-smoke"


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
