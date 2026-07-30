from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaiazero.core import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class TrainingExample:
    observation: FloatArray
    legal_mask: BoolArray
    policy_target: FloatArray
    value_target: FloatArray


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    observations: FloatArray
    legal_masks: BoolArray
    policy_targets: FloatArray
    value_targets: FloatArray


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[TrainingExample] = []
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, example: TrainingExample) -> None:
        owned = TrainingExample(
            observation=example.observation.copy(),
            legal_mask=example.legal_mask.copy(),
            policy_target=example.policy_target.copy(),
            value_target=example.value_target.copy(),
        )
        if len(self._items) < self.capacity:
            self._items.append(owned)
            return
        self._items[self._cursor] = owned
        self._cursor = (self._cursor + 1) % self.capacity

    def extend(self, examples: list[TrainingExample]) -> None:
        for example in examples:
            self.append(example)

    def sample(self, batch_size: int) -> TrainingBatch:
        if not self._items:
            raise ValueError("cannot sample an empty replay buffer")
        size = min(batch_size, len(self._items))
        indices = self._rng.choice(len(self._items), size=size, replace=False)
        items = [self._items[int(index)] for index in indices]
        return TrainingBatch(
            observations=np.stack([item.observation for item in items]).astype(np.float32),
            legal_masks=np.stack([item.legal_mask for item in items]).astype(np.bool_),
            policy_targets=np.stack([item.policy_target for item in items]).astype(np.float32),
            value_targets=np.stack([item.value_target for item in items]).astype(np.float32),
        )

