from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaiazero.core import GameState, PolicyValueEvaluator
from gaiazero.mcts import PUCTSearch, SearchConfig
from gaiazero.replay import TrainingExample


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    temperature_moves: int = 24
    temperature: float = 1.0
    max_moves: int = 512
    add_root_noise: bool = True
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    examples: list[TrainingExample]
    final_state: GameState
    actions: tuple[int, ...]


def play_self_game(
    initial_state: GameState,
    evaluator: PolicyValueEvaluator,
    search_config: SearchConfig,
    config: SelfPlayConfig | None = None,
) -> SelfPlayResult:
    settings = config or SelfPlayConfig()
    rng = np.random.default_rng(settings.seed)
    search = PUCTSearch(evaluator, search_config)
    state = initial_state
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    actions: list[int] = []

    while not state.is_terminal:
        if len(actions) >= settings.max_moves:
            raise RuntimeError(f"self-play exceeded {settings.max_moves} moves")
        temperature = settings.temperature if len(actions) < settings.temperature_moves else 0.0
        result = search.run(
            state,
            add_root_noise=settings.add_root_noise,
            temperature=temperature,
        )
        pending.append((state.observation(), state.legal_action_mask(), result.policy))
        action = int(rng.choice(state.action_size, p=result.policy))
        actions.append(action)
        state = state.apply(action)

    outcome = state.returns()
    examples = [
        TrainingExample(
            observation=observation,
            legal_mask=legal_mask,
            policy_target=policy,
            value_target=outcome,
        )
        for observation, legal_mask, policy in pending
    ]
    return SelfPlayResult(examples=examples, final_state=state, actions=tuple(actions))

