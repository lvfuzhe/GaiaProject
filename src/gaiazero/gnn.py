"""CPU-friendly graph hybrid network for the standard-v22 contract.

The implementation deliberately uses only stock PyTorch operators.  Nodes and
edges are padded to the configured maxima, so the same tensors can be exported
to ONNX and consumed by a future C++/TensorRT adapter without a custom
scatter/attention plugin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gaiazero.contracts import ACTION_TYPE_IDS, ACTION_TUPLE_SCHEMA_VERSION, RULES_VERSION


@dataclass(frozen=True, slots=True)
class GraphNetworkConfig:
    num_players: int
    node_feature_size: int = 16
    global_feature_size: int = 16
    player_feature_size: int = 16
    max_graph_nodes: int = 128
    max_graph_edges: int = 512
    relation_type_count: int = 16
    # ``None`` keeps small hand-written configs backwards-compatible by using
    # the node hidden width; production configs set this explicitly.
    relation_embedding_size: int | None = None
    hidden_size: int = 256
    hybrid_blocks: int = 12
    attention_heads: int = 8
    ffn_hidden_size: int = 512
    action_type_count: int = len(ACTION_TYPE_IDS)
    parameter_slot_count: int = 8
    argument_vocab_size: int = 128
    vp_buckets: int = 403
    dropout: float = 0.0
    network_config_id: str = "graph-hybrid-v1"

    def __post_init__(self) -> None:
        if self.num_players not in (2, 3, 4):
            raise ValueError("GraphNetworkConfig supports two to four players")
        positive = (
            "node_feature_size",
            "global_feature_size",
            "player_feature_size",
            "max_graph_nodes",
            "max_graph_edges",
            "relation_type_count",
            "hidden_size",
            "hybrid_blocks",
            "attention_heads",
            "ffn_hidden_size",
            "action_type_count",
            "parameter_slot_count",
            "argument_vocab_size",
            "vp_buckets",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.relation_embedding_size is None:
            object.__setattr__(self, "relation_embedding_size", self.hidden_size)
        elif int(self.relation_embedding_size) < 1:
            raise ValueError("relation_embedding_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class EdgeConditionedBlock(nn.Module):
    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_embedding = nn.Embedding(
            config.relation_type_count, config.relation_embedding_size
        )
        self.source_projection = nn.Linear(hidden, hidden, bias=False)
        self.edge_projection = nn.Linear(
            config.relation_embedding_size, hidden, bias=False
        )
        self.message_projection = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, config.ffn_hidden_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden, max(16, hidden // 4)),
            nn.SiLU(),
            nn.Linear(max(16, hidden // 4), hidden),
            nn.Sigmoid(),
        )

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        batch, node_count, hidden = nodes.shape
        normalized = self.node_norm(nodes)
        source_index = edge_index[..., 0].long().clamp(0, node_count - 1)
        destination_index = edge_index[..., 1].long().clamp(0, node_count - 1)
        source = torch.gather(
            normalized,
            1,
            source_index.unsqueeze(-1).expand(-1, -1, hidden),
        )
        destination_mask = torch.gather(node_mask, 1, destination_index)
        source_mask = torch.gather(node_mask, 1, source_index)
        active_edges = edge_mask.to(nodes.dtype) * source_mask * destination_mask
        relation = self.edge_embedding(
            edge_type.long().clamp(0, self.edge_embedding.num_embeddings - 1)
        )
        messages = self.message_projection(
            self.source_projection(source) + self.edge_projection(relation)
        )
        messages = messages * active_edges.unsqueeze(-1)

        # Dense one-hot routing avoids third-party scatter operators and has a
        # stable ONNX representation for fixed padded graph dimensions.
        routing = F.one_hot(destination_index, num_classes=node_count).to(nodes.dtype)
        routing = routing * active_edges.unsqueeze(-1)
        aggregate = torch.bmm(routing.transpose(1, 2), messages)
        degree = routing.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
        aggregate = aggregate / degree
        update = self.update(torch.cat((normalized, aggregate), dim=-1))
        return nodes + update * self.gate(normalized) * node_mask.unsqueeze(-1)


class GraphHybridNetwork(nn.Module):
    """Graph trunk with the three standard-v22 production heads."""

    def __init__(self, config: GraphNetworkConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.node_stem = nn.Sequential(
            nn.Linear(config.node_feature_size, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.global_projection = nn.Linear(config.global_feature_size, hidden)
        self.player_projection = nn.Sequential(
            nn.Linear(config.player_feature_size, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            EdgeConditionedBlock(config) for _ in range(config.hybrid_blocks)
        )
        self.context_norm = nn.LayerNorm(hidden)
        self.policy_type_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.action_type_count),
        )
        self.policy_argument_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.parameter_slot_count * config.argument_vocab_size),
        )
        self.pairwise_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.vp_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.vp_buckets),
        )

    @property
    def architecture_family(self) -> str:
        return "graph_hybrid"

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            "action_type_logits",
            "action_argument_logits",
            "pairwise_wdl_logits",
            "vp_belief_logits",
        )

    def _validate_inputs(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        player_features: Tensor,
        player_mask: Tensor,
    ) -> None:
        config = self.config
        # Shape checks are useful for eager callers but would become constants
        # in a traced ONNX graph and produce noisy tracer warnings.
        if torch.jit.is_tracing():
            return
        expected = {
            "node_features": (node_features, 3),
            "edge_index": (edge_index, 3),
            "edge_type": (edge_type, 2),
            "edge_mask": (edge_mask, 2),
            "node_mask": (node_mask, 2),
            "global_features": (global_features, 2),
            "player_features": (player_features, 3),
            "player_mask": (player_mask, 2),
        }
        for name, (value, rank) in expected.items():
            if value.ndim != rank:
                raise ValueError(f"{name} must have rank {rank}")
        if node_features.shape[1:] != (config.max_graph_nodes, config.node_feature_size):
            raise ValueError("node_features shape does not match GraphNetworkConfig")
        if edge_index.shape[1:] != (config.max_graph_edges, 2):
            raise ValueError("edge_index shape does not match GraphNetworkConfig")
        if edge_type.shape[1:] != (config.max_graph_edges,):
            raise ValueError("edge_type shape does not match GraphNetworkConfig")
        if edge_mask.shape[1:] != (config.max_graph_edges,):
            raise ValueError("edge_mask shape does not match GraphNetworkConfig")
        if node_mask.shape[1:] != (config.max_graph_nodes,):
            raise ValueError("node_mask shape does not match GraphNetworkConfig")
        if global_features.shape[1:] != (config.global_feature_size,):
            raise ValueError("global_features shape does not match GraphNetworkConfig")
        if player_features.shape[1:] != (config.num_players, config.player_feature_size):
            raise ValueError("player_features shape does not match GraphNetworkConfig")
        if player_mask.shape[1:] != (config.num_players,):
            raise ValueError("player_mask shape does not match GraphNetworkConfig")

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
        node_mask: Tensor,
        global_features: Tensor,
        player_features: Tensor,
        player_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate_inputs(
            node_features,
            edge_index,
            edge_type,
            edge_mask,
            node_mask,
            global_features,
            player_features,
            player_mask,
        )
        nodes = self.node_stem(node_features)
        for block in self.blocks:
            nodes = block(nodes, edge_index, edge_type, edge_mask, node_mask)
        node_weight = node_mask.to(nodes.dtype).unsqueeze(-1)
        pooled = (nodes * node_weight).sum(dim=1) / node_weight.sum(dim=1).clamp_min(1.0)
        context = self.context_norm(pooled + self.global_projection(global_features))

        player_hidden = self.player_projection(player_features)
        player_hidden = player_hidden * player_mask.to(player_hidden.dtype).unsqueeze(-1)
        pair_features: list[Tensor] = []
        for left in range(self.config.num_players):
            for right in range(left + 1, self.config.num_players):
                pair_features.append(
                    torch.cat((context, player_hidden[:, left], player_hidden[:, right]), dim=-1)
                )
        pairwise = self.pairwise_head(torch.stack(pair_features, dim=1))
        vp_context = context.unsqueeze(1).expand(-1, self.config.num_players, -1)
        vp = self.vp_head(torch.cat((vp_context, player_hidden), dim=-1))
        action_type = self.policy_type_head(context)
        action_argument = self.policy_argument_head(context).reshape(
            -1,
            self.config.parameter_slot_count,
            self.config.argument_vocab_size,
        )
        return action_type, action_argument, pairwise, vp


def graph_config_from_training(
    training_config: object,
    players: int,
    *,
    node_feature_size: int = 16,
    global_feature_size: int = 16,
    player_feature_size: int = 16,
) -> GraphNetworkConfig:
    """Create graph capacity from ``GaiaTrainingConfig`` without coupling it."""

    capacity = dict(getattr(training_config, "network_capacity"))
    observation = dict(getattr(training_config, "data")["observation_schema"])
    action = dict(getattr(training_config, "data")["action_schema"])
    network = dict(getattr(training_config, "data")["network"])
    vp = dict(network.get("vp_belief", {}))
    return GraphNetworkConfig(
        num_players=players,
        node_feature_size=node_feature_size,
        global_feature_size=global_feature_size,
        player_feature_size=player_feature_size,
        max_graph_nodes=int(observation.get("max_graph_nodes", 128)),
        max_graph_edges=int(observation.get("max_graph_edges", 512)),
        relation_type_count=int(observation.get("relation_type_count", 16)),
        relation_embedding_size=int(
            capacity.get(
                "relation_embedding_size", capacity.get("hidden_size", 256)
            )
        ),
        hidden_size=int(capacity.get("hidden_size", 256)),
        hybrid_blocks=int(capacity.get("hybrid_blocks", 12)),
        attention_heads=int(capacity.get("attention_heads", 8)),
        ffn_hidden_size=int(capacity.get("ffn_hidden_size", 512)),
        parameter_slot_count=int(action.get("parameter_slot_count", 8)),
        action_type_count=len(ACTION_TYPE_IDS),
        vp_buckets=int(vp.get("output_buckets", 403)),
        dropout=float(capacity.get("dropout", 0.0)),
        network_config_id=str(network.get("network_config_id", "graph-hybrid-v1")),
    )


def graph_inputs_from_state(
    state: Any,
    config: GraphNetworkConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Encode a ``GaiaState`` into the fixed padded graph input contract."""

    import numpy as np

    nodes = np.zeros(
        (config.max_graph_nodes, config.node_feature_size), dtype=np.float32
    )
    node_mask = np.zeros(config.max_graph_nodes, dtype=np.float32)
    active = [index for index, value in enumerate(state.active_planets) if value]
    for node_index, planet in enumerate(active[: config.max_graph_nodes]):
        node_mask[node_index] = 1.0
        values = (
            float(state.planet_q[planet]) / 16.0,
            float(state.planet_r[planet]) / 16.0,
            float(state.terrains[planet]) / 9.0,
            float(state.owners[planet]) / max(1, state.num_players),
            float(state.buildings[planet]) / 5.0,
            float(state.federated[planet]),
            float(state.gaiaformer_owner[planet]) / max(1, state.num_players),
            float(state.coexisting_mine_owner[planet]) / max(1, state.num_players),
        )
        nodes[node_index, : min(len(values), config.node_feature_size)] = values[
            : config.node_feature_size
        ]

    edge_index = np.zeros((config.max_graph_edges, 2), dtype=np.int64)
    edge_type = np.zeros(config.max_graph_edges, dtype=np.int64)
    edge_mask = np.zeros(config.max_graph_edges, dtype=np.float32)
    edge_count = 0
    for left, planet_left in enumerate(active[: config.max_graph_nodes]):
        for right, planet_right in enumerate(active[: config.max_graph_nodes]):
            if left == right:
                continue
            distance = abs(state.planet_q[planet_left] - state.planet_q[planet_right])
            distance += abs(state.planet_r[planet_left] - state.planet_r[planet_right])
            distance += abs(
                (state.planet_q[planet_left] + state.planet_r[planet_left])
                - (state.planet_q[planet_right] + state.planet_r[planet_right])
            )
            if distance // 2 > 1 or edge_count >= config.max_graph_edges:
                continue
            edge_index[edge_count] = (left, right)
            edge_type[edge_count] = int(state.terrains[planet_right]) % config.relation_type_count
            edge_mask[edge_count] = 1.0
            edge_count += 1

    observation = np.asarray(state.observation(), dtype=np.float32)
    global_features = np.zeros(config.global_feature_size, dtype=np.float32)
    global_features[: min(len(observation), config.global_feature_size)] = observation[
        : config.global_feature_size
    ]
    players = np.zeros(
        (config.num_players, config.player_feature_size), dtype=np.float32
    )
    player_mask = np.ones(config.num_players, dtype=np.float32)
    for player, info in enumerate(state.players):
        values = (
            info.credits / 30.0,
            info.ore / 15.0,
            info.knowledge / 15.0,
            info.qic / 10.0,
            info.vp / 150.0,
            info.bowl_one / 15.0,
            info.bowl_two / 15.0,
            info.bowl_three / 15.0,
            info.gaia_power / 15.0,
            info.gaiaformers / 3.0,
            info.federation_tokens / 6.0,
            info.tracks[0] / 5.0,
            info.tracks[1] / 5.0,
            info.tracks[2] / 5.0,
            float(info.passed),
            float(player == state.current_player),
        )
        players[player, : min(len(values), config.player_feature_size)] = values[
            : config.player_feature_size
        ]
    return tuple(
        torch.from_numpy(array).unsqueeze(0)
        for array in (
            nodes,
            edge_index,
            edge_type,
            edge_mask,
            node_mask,
            global_features,
            players,
            player_mask,
        )
    )


def save_graph_checkpoint(
    path: str | Path,
    model: GraphHybridNetwork,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    swa: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_format": "graph-hybrid-v1",
        "network_config": asdict(model.config),
        "model_state": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if swa is not None:
        payload["swa_state"] = swa.state_dict()
        if hasattr(swa, "metadata"):
            payload["metadata"]["swa"] = swa.metadata()
    torch.save(payload, destination)


def load_graph_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GraphHybridNetwork, dict[str, Any], dict[str, Any] | None]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_format") != "graph-hybrid-v1":
        raise ValueError("not a graph-hybrid checkpoint")
    model = GraphHybridNetwork(GraphNetworkConfig(**payload["network_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    swa = payload.get("swa_state")
    return model, dict(payload.get("metadata", {})), dict(swa) if isinstance(swa, dict) else None
