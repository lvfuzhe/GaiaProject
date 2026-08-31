"""Small, checkpoint-friendly stochastic weight averaging implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class SWAConfig:
    enabled: bool = True
    start_after_samples: int = 200_000
    update_every_samples: int = 1_000
    snapshot_count: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.start_after_samples < 0:
            raise ValueError("start_after_samples must be non-negative")
        if self.update_every_samples < 1:
            raise ValueError("update_every_samples must be positive")
        if self.snapshot_count < 1:
            raise ValueError("snapshot_count must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SWAConfig":
        if value is None:
            return cls()
        return cls(
            enabled=bool(value.get("enabled", True)),
            start_after_samples=int(value.get("start_after_samples", 200_000)),
            update_every_samples=int(value.get("update_every_samples", 1_000)),
            snapshot_count=int(value.get("snapshot_count", 32)),
            device=str(value.get("device", "cpu")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SWAAccumulator:
    """Equal-weight rolling average of recent model snapshots.

    Parameters are copied to the configured device when captured, keeping the
    running trainer lightweight.  ``copy_to`` never mutates the accumulator.
    """

    def __init__(self, config: SWAConfig | None = None) -> None:
        self.config = config or SWAConfig()
        self._snapshots: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.snapshot_count
        )
        self.samples_seen = 0
        self.last_update_samples = -1

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    @property
    def active(self) -> bool:
        return bool(self._snapshots)

    def maybe_update(self, model: nn.Module, samples_seen: int) -> bool:
        self.samples_seen = max(self.samples_seen, int(samples_seen))
        if not self.config.enabled:
            return False
        if self.samples_seen < self.config.start_after_samples:
            return False
        if (
            self.last_update_samples >= 0
            and self.samples_seen - self.last_update_samples
            < self.config.update_every_samples
        ):
            return False
        device = torch.device(self.config.device)
        snapshot = {
            name: parameter.detach().to(device=device).clone()
            for name, parameter in model.state_dict().items()
        }
        self._snapshots.append(snapshot)
        self.last_update_samples = self.samples_seen
        return True

    def averaged_state_dict(self) -> dict[str, Tensor]:
        if not self._snapshots:
            raise ValueError("SWA has no captured snapshots")
        names = self._snapshots[0].keys()
        result: dict[str, Tensor] = {}
        for name in names:
            values = [snapshot[name] for snapshot in self._snapshots]
            first = values[0]
            if not torch.is_floating_point(first):
                result[name] = first.clone()
                continue
            accumulator = torch.zeros_like(first, dtype=torch.float32)
            for value in values:
                accumulator.add_(value.float())
            result[name] = (accumulator / len(values)).to(dtype=first.dtype)
        return result

    def copy_to(self, model: nn.Module) -> None:
        state = self.averaged_state_dict()
        destination = model.state_dict()
        for name, value in state.items():
            destination[name].copy_(value.to(destination[name].device))

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "samples_seen": self.samples_seen,
            "last_update_samples": self.last_update_samples,
            "snapshots": [
                {name: value.clone() for name, value in snapshot.items()}
                for snapshot in self._snapshots
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.samples_seen = int(state.get("samples_seen", 0))
        self.last_update_samples = int(state.get("last_update_samples", -1))
        self._snapshots.clear()
        for raw_snapshot in state.get("snapshots", []):
            if not isinstance(raw_snapshot, Mapping):
                raise ValueError("invalid SWA snapshot")
            self._snapshots.append(
                {
                    str(name): value.detach().to(self.config.device).clone()
                    for name, value in raw_snapshot.items()
                    if isinstance(value, Tensor)
                }
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "samples_seen": self.samples_seen,
            "snapshot_count": self.snapshot_count,
            "last_update_samples": self.last_update_samples,
            "start_after_samples": self.config.start_after_samples,
            "update_every_samples": self.config.update_every_samples,
        }
