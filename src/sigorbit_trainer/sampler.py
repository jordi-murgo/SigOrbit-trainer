"""Deterministic P x K identity batch sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from .augment import stateless_seed


class PKBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        persons_per_batch: int,
        samples_per_person: int,
        *,
        seed: int,
        stage: str,
        epoch: int,
        start_batch: int = 0,
    ) -> None:
        groups: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            groups[int(label)].append(index)
        self.groups = dict(groups)
        self.persons = sorted(groups)
        self.persons_per_batch = persons_per_batch
        self.samples_per_person = samples_per_person
        self.seed = seed
        self.stage = stage
        self.epoch = epoch
        self.start_batch = start_batch
        self.batch_count = len(labels) // (persons_per_batch * samples_per_person)
        if len(self.persons) < persons_per_batch:
            raise ValueError("P exceeds available identities")
        if self.batch_count < 1:
            raise ValueError("dataset is smaller than one P*K batch")
        if not 0 <= start_batch <= self.batch_count:
            raise ValueError("invalid resume batch cursor")

    def __len__(self) -> int:
        return self.batch_count - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        seed = stateless_seed(self.seed, self.stage, self.epoch, "pk-sampler")
        rng = random.Random(seed)
        for batch_index in range(self.batch_count):
            selected = rng.sample(self.persons, self.persons_per_batch)
            batch: list[int] = []
            for person in selected:
                pool = self.groups[person]
                if len(pool) >= self.samples_per_person:
                    batch.extend(rng.sample(pool, self.samples_per_person))
                else:
                    batch.extend(rng.choice(pool) for _ in range(self.samples_per_person))
            if batch_index >= self.start_batch:
                yield batch
