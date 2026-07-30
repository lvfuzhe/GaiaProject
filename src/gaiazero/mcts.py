from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from gaiazero.core import FloatArray, GameState, PolicyValueEvaluator


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if not 0 <= self.root_noise_fraction <= 1:
            raise ValueError("root_noise_fraction must be between zero and one")


@dataclass(slots=True)
class SearchEdge:
    prior: float
    num_players: int
    visit_count: int = 0
    value_sum: FloatArray = field(init=False)
    child: SearchNode | None = None

    def __post_init__(self) -> None:
        self.value_sum = np.zeros(self.num_players, dtype=np.float32)

    def q_value(self, player: int) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player] / self.visit_count)


@dataclass(slots=True)
class SearchNode:
    state: GameState
    edges: dict[int, SearchEdge] = field(default_factory=dict)
    expanded: bool = False
    network_value: FloatArray | None = None

    @property
    def visit_count(self) -> int:
        return sum(edge.visit_count for edge in self.edges.values())


@dataclass(frozen=True, slots=True)
class SearchResult:
    policy: FloatArray
    visits: np.ndarray
    root_value: FloatArray


class PUCTSearch:
    """AlphaZero-style perfect-information MCTS with multiplayer backup.

    Values always retain absolute player order. Selection at a node uses the
    component belonging to that node's current player, so no two-player sign
    flip or zero-sum assumption is hidden in the implementation.
    """

    def __init__(self, evaluator: PolicyValueEvaluator, config: SearchConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def run(
        self,
        state: GameState,
        *,
        add_root_noise: bool = False,
        temperature: float = 1.0,
    ) -> SearchResult:
        if state.is_terminal:
            raise ValueError("cannot search a terminal state")
        root = SearchNode(state)
        root_value = self._expand(root)
        if add_root_noise:
            self._add_root_noise(root)

        for _ in range(self.config.simulations):
            node = root
            path: list[SearchEdge] = []
            while node.expanded and node.edges:
                _, edge = self._select(node)
                path.append(edge)
                if edge.child is None:
                    action = next(action for action, candidate in node.edges.items() if candidate is edge)
                    edge.child = SearchNode(node.state.apply(action))
                node = edge.child
                if node.state.is_terminal:
                    value = node.state.returns()
                    break
                if not node.expanded:
                    value = self._expand(node)
                    break
            else:
                if node.state.is_terminal:
                    value = node.state.returns()
                else:
                    value = self._expand(node)

            for edge in path:
                edge.visit_count += 1
                edge.value_sum += value

        visits = np.zeros(state.action_size, dtype=np.int64)
        for action, edge in root.edges.items():
            visits[action] = edge.visit_count
        policy = self._visits_to_policy(visits, state.legal_action_mask(), temperature)
        root_value = self._root_search_value(root, root_value)
        return SearchResult(policy=policy, visits=visits, root_value=root_value)

    def _expand(self, node: SearchNode) -> FloatArray:
        priors, value = self.evaluator.evaluate(node.state)
        priors = np.asarray(priors, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if priors.shape != (node.state.action_size,):
            raise ValueError(f"evaluator policy shape {priors.shape} does not match action space")
        if value.shape != (node.state.num_players,):
            raise ValueError(f"evaluator value shape {value.shape} does not match player count")
        if not np.all(np.isfinite(priors)) or not np.all(np.isfinite(value)):
            raise ValueError("evaluator returned non-finite values")

        legal = node.state.legal_actions()
        masked = np.maximum(priors, 0.0) * node.state.legal_action_mask()
        total = float(masked.sum())
        if total <= 0:
            masked[list(legal)] = 1.0 / len(legal)
        else:
            masked /= total
        node.edges = {
            action: SearchEdge(float(masked[action]), node.state.num_players)
            for action in legal
        }
        node.expanded = True
        node.network_value = value.copy()
        return value

    def _select(self, node: SearchNode) -> tuple[int, SearchEdge]:
        player = node.state.current_player
        parent_visits = node.visit_count
        exploration_scale = self.config.c_puct * sqrt(parent_visits + 1)
        best_action = -1
        best_edge: SearchEdge | None = None
        best_score = float("-inf")
        for action, edge in node.edges.items():
            score = edge.q_value(player) + exploration_scale * edge.prior / (1 + edge.visit_count)
            if score > best_score:
                best_action, best_edge, best_score = action, edge, score
        if best_edge is None:
            raise RuntimeError("selection reached an expanded node without edges")
        return best_action, best_edge

    def _add_root_noise(self, root: SearchNode) -> None:
        actions = tuple(root.edges)
        if not actions or self.config.root_noise_fraction == 0:
            return
        noise = self.rng.dirichlet(np.full(len(actions), self.config.dirichlet_alpha))
        fraction = self.config.root_noise_fraction
        for action, sample in zip(actions, noise, strict=True):
            edge = root.edges[action]
            edge.prior = (1 - fraction) * edge.prior + fraction * float(sample)

    @staticmethod
    def _visits_to_policy(visits: np.ndarray, legal_mask: np.ndarray, temperature: float) -> FloatArray:
        policy = np.zeros(visits.shape, dtype=np.float32)
        legal = np.flatnonzero(legal_mask)
        if temperature <= 1e-6:
            legal_visits = visits[legal]
            policy[int(legal[int(np.argmax(legal_visits))])] = 1.0
            return policy
        scaled = np.zeros(visits.shape, dtype=np.float64)
        scaled[legal] = np.power(visits[legal].astype(np.float64), 1.0 / temperature)
        total = float(scaled.sum())
        if total <= 0:
            policy[legal] = 1.0 / len(legal)
        else:
            policy = (scaled / total).astype(np.float32)
        return policy

    @staticmethod
    def _root_search_value(root: SearchNode, fallback: FloatArray) -> FloatArray:
        total = root.visit_count
        if total == 0:
            return fallback.copy()
        value = np.zeros(root.state.num_players, dtype=np.float32)
        for edge in root.edges.values():
            value += edge.value_sum
        return value / total

