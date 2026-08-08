from pathlib import Path

import pytest
from pydantic import ValidationError

from sigorbit_trainer.config import SplitConfig, load_config
from sigorbit_trainer.data import apply_split_policy, load_data
from sigorbit_trainer.manifest import SampleRecord


def _records(signers: int, samples: int = 4, split: str = "train") -> tuple[SampleRecord, ...]:
    return tuple(
        SampleRecord(
            sample_id=f"s-{signer:03d}-{sample}",
            image=f"images/w{signer:03d}-{sample}.png",
            image_sha256=f"{signer:032d}{sample:032d}",
            media_type="image/png",
            width=64,
            height=32,
            signer_id=f"writer-{signer:03d}",
            split=split,  # type: ignore[arg-type]
            kind="genuine",
            source_group=f"src-{signer:03d}-{sample}",
            canonical_angle_degrees=0.0,
        )
        for signer in range(signers)
        for sample in range(samples)
    )


def test_random_split_is_writer_disjoint_and_deterministic() -> None:
    records = _records(20)
    policy = SplitConfig(strategy="random_by_signer", seed=1234)
    first, summary = apply_split_policy(records, policy)
    repeated, repeated_summary = apply_split_policy(records, policy)
    assert summary == repeated_summary
    assert [r.split for r in first] == [r.split for r in repeated]

    assignments: dict[str, set[str]] = {}
    for record in first:
        assignments.setdefault(record.signer_id, set()).add(record.split)
    # every signer lands wholly in a single split
    assert all(len(splits) == 1 for splits in assignments.values())
    per_split = {name: set() for name in ("train", "validation", "test")}
    for signer, splits in assignments.items():
        per_split[next(iter(splits))].add(signer)
    assert not per_split["train"] & per_split["validation"]
    assert not per_split["train"] & per_split["test"]
    assert not per_split["validation"] & per_split["test"]
    assert (len(per_split["train"]), len(per_split["validation"]), len(per_split["test"])) == (
        14,
        3,
        3,
    )


def test_random_split_changes_with_the_seed() -> None:
    records = _records(20)
    a, summary_a = apply_split_policy(records, SplitConfig(strategy="random_by_signer", seed=1))
    b, summary_b = apply_split_policy(records, SplitConfig(strategy="random_by_signer", seed=2))
    assert summary_a["assignment_sha256"] != summary_b["assignment_sha256"]
    assert [r.split for r in a] != [r.split for r in b]


def test_manifest_strategy_leaves_records_untouched() -> None:
    records = _records(6)
    unchanged, summary = apply_split_policy(records, SplitConfig())
    assert unchanged is records
    assert summary == {"strategy": "manifest"}


def test_random_split_rejects_a_source_group_spanning_signers() -> None:
    records = list(_records(8))
    records[0] = records[0].model_copy(update={"source_group": "shared"})
    records[4] = records[4].model_copy(update={"source_group": "shared"})
    with pytest.raises(ValueError, match="single signer"):
        apply_split_policy(tuple(records), SplitConfig(strategy="random_by_signer", seed=3))


def test_random_split_needs_enough_signers() -> None:
    with pytest.raises(ValueError, match="at least four distinct signers"):
        apply_split_policy(_records(3), SplitConfig(strategy="random_by_signer", seed=5))


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"sum to exactly 1\.0"):
        SplitConfig(strategy="random_by_signer", train_fraction=0.7, validation_fraction=0.7)


def test_seed_override_requires_the_random_strategy() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    with pytest.raises(ValueError, match="random_by_signer"):
        load_config(path, split_seed=42)


def test_split_seed_changes_the_dataset_fingerprint(tmp_path: Path) -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.toml"
    config = load_config(path)
    randomized = config.data.model_copy(
        update={"split": SplitConfig(strategy="random_by_signer", seed=11)}
    )
    other = config.data.model_copy(
        update={"split": SplitConfig(strategy="random_by_signer", seed=12)}
    )
    first = load_data(randomized, config.sampler, config.run.seed)
    second = load_data(other, config.sampler, config.run.seed)
    assert first.summary["source_fingerprint"] != second.summary["source_fingerprint"]
    assert first.summary["split"]["strategy"] == "random_by_signer"
