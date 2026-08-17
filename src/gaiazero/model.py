from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from gaiazero.core import FloatArray, GameState


NNUE_ARCHITECTURE = "nnue"
KATAGO_ARCHITECTURE = "katago"
LEGACY_ARCHITECTURE = "legacy_mlp"


def architecture_for_players(num_players: int) -> str:
    if num_players == 2:
        return NNUE_ARCHITECTURE
    if num_players in (3, 4):
        return KATAGO_ARCHITECTURE
    raise ValueError("network architecture requires two to four players")


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    observation_size: int
    action_size: int
    num_players: int
    hidden_size: int = 256
    residual_blocks: int = 4
    architecture: str = "auto"

    def __post_init__(self) -> None:
        if self.observation_size < 1 or self.action_size < 1:
            raise ValueError("network dimensions must be positive")
        if self.hidden_size < 16:
            raise ValueError("hidden_size must be at least 16")
        if self.residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        expected = architecture_for_players(self.num_players)
        selected = expected if self.architecture == "auto" else self.architecture
        if selected not in (expected, LEGACY_ARCHITECTURE):
            raise ValueError(
                f"{self.num_players}-player models require {expected}, got {selected}"
            )
        object.__setattr__(self, "architecture", selected)


class SquaredClippedReLU(nn.Module):
    """NNUE-style bounded activation with stable integer-friendly support."""

    def forward(self, inputs: Tensor) -> Tensor:
        clipped = torch.clamp(inputs, 0.0, 1.0)
        return clipped * clipped


class NNUEPolicyValueBackbone(nn.Module):
    """Two-player feature accumulators with an antisymmetric value head.

    The observation is currently dense, so accumulators are recomputed for each
    state. Their first layer remains additive and can be cached incrementally
    when the rules engine exposes observation deltas in a future optimization.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        head_hidden = max(16, hidden // 2)
        self.player_zero_accumulator = nn.Linear(config.observation_size, hidden)
        self.player_one_accumulator = nn.Linear(config.observation_size, hidden)
        self.accumulator_activation = SquaredClippedReLU()
        self.hidden = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            SquaredClippedReLU(),
            nn.Linear(hidden, head_hidden),
            SquaredClippedReLU(),
        )
        self.policy_head = nn.Linear(head_hidden, config.action_size)
        self.value_head = nn.Linear(head_hidden, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        player_zero = self.accumulator_activation(
            self.player_zero_accumulator(observations)
        )
        player_one = self.accumulator_activation(
            self.player_one_accumulator(observations)
        )
        hidden = self.hidden(torch.cat((player_zero, player_one), dim=1))
        player_zero_value = torch.tanh(self.value_head(hidden))
        values = torch.cat((player_zero_value, -player_zero_value), dim=1)
        return self.policy_head(hidden), values


class KataGoResidualBlock(nn.Module):
    """Gated residual block adapted from KataGo's global-pooling tower."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        gate_hidden = max(16, hidden_size // 4)
        self.norm = nn.LayerNorm(hidden_size)
        self.expand = nn.Linear(hidden_size, hidden_size * 2)
        self.project = nn.Linear(hidden_size, hidden_size)
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm(inputs)
        value, gate = self.expand(normalized).chunk(2, dim=1)
        residual = self.project(nn.functional.silu(value) * torch.sigmoid(gate))
        return inputs + residual * self.global_gate(normalized)


class KataGoPolicyValueBackbone(nn.Module):
    """Multiplayer gated residual tower with separate policy and value heads."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        head_hidden = max(16, hidden // 2)
        self.stem = nn.Sequential(
            nn.Linear(config.observation_size, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *(KataGoResidualBlock(hidden) for _ in range(config.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.action_size),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, config.num_players),
        )

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.blocks(self.stem(observations))
        raw_values = self.value_head(hidden)
        centered_values = raw_values - raw_values.mean(dim=1, keepdim=True)
        return self.policy_head(hidden), torch.tanh(centered_values)


class LegacyResidualMLPBlock(nn.Module):
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


class LegacyPolicyValueBackbone(nn.Module):
    """Loader-only implementation for checkpoints created before model splitting."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(config.observation_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(
                LegacyResidualMLPBlock(config.hidden_size)
                for _ in range(config.residual_blocks)
            )
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


class PolicyValueNetwork(nn.Module):
    """Stable wrapper that dispatches to the player-count-specific network."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture == NNUE_ARCHITECTURE:
            self.network = NNUEPolicyValueBackbone(config)
        elif config.architecture == KATAGO_ARCHITECTURE:
            self.network = KataGoPolicyValueBackbone(config)
        elif config.architecture == LEGACY_ARCHITECTURE:
            self.network = LegacyPolicyValueBackbone(config)
        else:  # NetworkConfig validates this before construction.
            raise ValueError(f"unsupported network architecture {config.architecture}")

    @property
    def architecture(self) -> str:
        return self.config.architecture

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        return self.network(observations)


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
        "checkpoint_format": 2,
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
    config_payload = dict(payload["network_config"])
    legacy_checkpoint = "architecture" not in config_payload
    if legacy_checkpoint:
        config_payload["architecture"] = LEGACY_ARCHITECTURE
    model = PolicyValueNetwork(NetworkConfig(**config_payload))
    model_state = dict(payload["model_state"])
    if legacy_checkpoint and not any(key.startswith("network.") for key in model_state):
        model_state = {f"network.{key}": value for key, value in model_state.items()}
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint weights do not match {model.config.architecture} architecture"
        ) from error
    model.to(resolved)
    return model, dict(payload.get("metadata", {}))
