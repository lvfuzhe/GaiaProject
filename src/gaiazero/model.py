from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from gaiazero.core import FloatArray, GameState


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    observation_size: int
    action_size: int
    num_players: int
    hidden_size: int = 256
    residual_blocks: int = 4


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


class PolicyValueNetwork(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Linear(config.observation_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(ResidualMLPBlock(config.hidden_size) for _ in range(config.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_players),
            nn.Tanh(),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.blocks(self.stem(observations))
        return self.policy_head(hidden), self.value_head(hidden)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkEvaluator:
    def __init__(self, model: PolicyValueNetwork, device: str | torch.device = "auto") -> None:
        self.model = model
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.model.to(self.device)

    def evaluate(self, state: GameState) -> tuple[FloatArray, FloatArray]:
        if state.is_terminal:
            return np.zeros(state.action_size, dtype=np.float32), state.returns()
        self.model.eval()
        observation = torch.from_numpy(state.observation()).to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(state.legal_action_mask()).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, values = self.model(observation)
            logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
            priors = torch.softmax(logits, dim=1)
        return (
            priors.squeeze(0).cpu().numpy().astype(np.float32),
            values.squeeze(0).cpu().numpy().astype(np.float32),
        )


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNetwork,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "network_config": asdict(model.config),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[PolicyValueNetwork, dict[str, Any]]:
    resolved = resolve_device(device) if isinstance(device, str) else device
    payload = torch.load(Path(path), map_location=resolved, weights_only=False)
    model = PolicyValueNetwork(NetworkConfig(**payload["network_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(resolved)
    return model, dict(payload.get("metadata", {}))
