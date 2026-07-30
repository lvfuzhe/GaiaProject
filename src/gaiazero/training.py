from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from gaiazero.model import PolicyValueNetwork, resolve_device
from gaiazero.replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class TrainMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    policy_entropy: float


TrainingObserver = Callable[[int, TrainMetrics], None]


class AlphaZeroTrainer:
    def __init__(self, model: PolicyValueNetwork, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_updates(
        self,
        replay: ReplayBuffer,
        updates: int,
        observer: TrainingObserver | None = None,
    ) -> TrainMetrics:
        if updates < 1:
            raise ValueError("updates must be positive")
        totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
        self.model.train()
        for update_index in range(updates):
            batch = replay.sample(self.config.batch_size)
            observations = torch.from_numpy(batch.observations).to(self.device)
            masks = torch.from_numpy(batch.legal_masks).to(self.device)
            policy_targets = torch.from_numpy(batch.policy_targets).to(self.device)
            value_targets = torch.from_numpy(batch.value_targets).to(self.device)

            logits, values = self.model(observations)
            masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
            log_policy = torch.log_softmax(masked_logits, dim=1)
            policy = torch.softmax(masked_logits, dim=1)
            policy_loss = -(policy_targets * log_policy).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            entropy = -(policy * log_policy).sum(dim=1).mean()
            totals["loss"] += float(loss.detach())
            totals["policy"] += float(policy_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            if observer is not None:
                observer(
                    update_index + 1,
                    TrainMetrics(
                        loss=float(loss.detach()),
                        policy_loss=float(policy_loss.detach()),
                        value_loss=float(value_loss.detach()),
                        policy_entropy=float(entropy.detach()),
                    ),
                )

        return TrainMetrics(
            loss=totals["loss"] / updates,
            policy_loss=totals["policy"] / updates,
            value_loss=totals["value"] / updates,
            policy_entropy=totals["entropy"] / updates,
        )
