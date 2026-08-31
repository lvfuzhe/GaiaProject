from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from gaiazero.core import GameState, PolicyValueEvaluator
from gaiazero.contracts import canonical_json
from gaiazero.mcts import PUCTSearch, SearchConfig, SearchResult
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
    # One pre-action row per training position plus one terminal row.  The
    # fields are plain JSON-compatible dictionaries so writers can serialize
    # them without importing the game implementation.
    trajectory: tuple[dict[str, Any], ...] = ()


SelfPlayObserver = Callable[[int, GameState, int, GameState, SearchResult], None]


def play_self_game(
    initial_state: GameState,
    evaluator: PolicyValueEvaluator,
    search_config: SearchConfig,
    config: SelfPlayConfig | None = None,
    observer: SelfPlayObserver | None = None,
) -> SelfPlayResult:
    settings = config or SelfPlayConfig()
    rng = np.random.default_rng(settings.seed)
    search = PUCTSearch(evaluator, search_config)
    state = initial_state
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    actions: list[int] = []
    trajectory: list[dict[str, Any]] = []

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
        action_tuple = (
            state.action_tuple(action).to_dict()
            if hasattr(state, "action_tuple")
            else None
        )
        state_hash_value = (
            state.state_hash()
            if hasattr(state, "state_hash")
            else None
        )
        trajectory.append(
            {
                "position_index": len(actions),
                "semantic_turn_index": len(actions),
                "round": int(getattr(state, "round_number", 0)),
                "player_to_move": int(state.current_player),
                "action_id": action,
                "action_tuple": action_tuple,
                "legal_action_tuples": [
                    state.action_tuple(candidate).to_dict()
                    for candidate in state.legal_actions()
                ]
                if hasattr(state, "action_tuple")
                else [],
                "policy_visit_targets_by_tuple": [
                    {
                        "action_tuple": state.action_tuple(candidate).to_dict(),
                        "visit_count": int(result.visits[candidate]),
                        "visit_probability": float(result.policy[candidate]),
                    }
                    for candidate in state.legal_actions()
                ]
                if hasattr(state, "action_tuple")
                else [],
                "state_hash": state_hash_value,
                "state_json": canonical_json(state),
                "state": state.snapshot()
                if hasattr(state, "snapshot")
                else None,
            }
        )
        actions.append(action)
        next_state = state.apply(action)
        if observer is not None:
            observer(len(actions), state, action, next_state, result)
        state = next_state

    # A terminal row makes the raw shard independently replayable and allows
    # consumers to verify the final hash without applying the last action.
    trajectory.append(
        {
            "position_index": len(actions),
            "semantic_turn_index": len(actions),
            "round": int(getattr(state, "round_number", 0)),
            "player_to_move": int(state.current_player),
            "action_id": None,
            "action_tuple": None,
            "legal_action_tuples": [],
            "policy_visit_targets_by_tuple": [],
            "state_hash": state.state_hash()
            if hasattr(state, "state_hash")
            else None,
            "state_json": canonical_json(state),
            "state": state.snapshot()
            if hasattr(state, "snapshot")
            else None,
        }
    )

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
    return SelfPlayResult(
        examples=examples,
        final_state=state,
        actions=tuple(actions),
        trajectory=tuple(trajectory),
    )
