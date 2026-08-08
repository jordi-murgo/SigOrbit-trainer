from pathlib import Path

import pytest
from pydantic import ValidationError

from sigorbit_trainer.config import TrainerConfig, config_sha256, load_config


def test_smoke_config_is_strict_and_stable() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    first = load_config(path)
    second = load_config(path)
    assert first.schema_version == 1
    assert first.model.release_compatible is False
    assert config_sha256(first) == config_sha256(second)


def test_unknown_config_key_is_rejected() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(path)
    payload = config.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        TrainerConfig.model_validate(payload)


def test_release_architecture_cannot_drift() -> None:
    path = Path(__file__).parents[1] / "configs" / "c8-257-research.toml"
    config = load_config(path)
    with pytest.raises(ValidationError, match="257/C8"):
        config.model.__class__.model_validate({**config.model.model_dump(), "input_size": 129})


def test_non_finite_rotation_angle_is_rejected() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(path)
    payload = config.backbone.model_dump()
    payload["discrete_rotations_deg"] = (0.0, float("nan"))
    with pytest.raises(ValidationError, match="finite"):
        config.backbone.__class__.model_validate(payload)
