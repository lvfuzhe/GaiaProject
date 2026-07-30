from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from gaiazero.core import GameState, PolicyValueEvaluator
from gaiazero.mcts import PUCTSearch, SearchConfig


@dataclass(frozen=True, slots=True)
class ArenaGameResult:
    final_state: GameState
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArenaSummary:
    games: int
    mean_value: float
    first_places: int
    draws: int
    mean_score: float


def play_arena_game(
    initial_state: GameState,
    evaluators: Sequence[PolicyValueEvaluator],
    search_config: SearchConfig,
    max_moves: int = 512,
) -> ArenaGameResult:
    if len(evaluators) != initial_state.num_players:
        raise ValueError("one evaluator is required for each player")
    state = initial_state
    actions: list[int] = []
    searches = [
        PUCTSearch(evaluator, replace(search_config, seed=search_config.seed + player))
        for player, evaluator in enumerate(evaluators)
    ]
    while not state.is_terminal:
        if len(actions) >= max_moves:
            raise RuntimeError(f"arena game exceeded {max_moves} moves")
        result = searches[state.current_player].run(state, add_root_noise=False, temperature=0.0)
        action = int(np.argmax(result.policy))
        actions.append(action)
        state = state.apply(action)
    return ArenaGameResult(final_state=state, actions=tuple(actions))


def evaluate_against(
    state_factory: Callable[[int], GameState],
    challenger: PolicyValueEvaluator,
    baseline: PolicyValueEvaluator,
    *,
    num_players: int,
    games: int,
    search_config: SearchConfig,
) -> ArenaSummary:
    if games < 1:
        raise ValueError("games must be positive")
    values: list[float] = []
    scores: list[float] = []
    first_places = 0
    draws = 0
    for game in range(games):
        challenger_seat = game % num_players
        evaluators = [baseline] * num_players
        evaluators[challenger_seat] = challenger
        result = play_arena_game(
            state_factory(game),
            evaluators,
            replace(search_config, seed=search_config.seed + game * num_players),
        )
        outcome = result.final_state.returns()
        final_scores = result.final_state.final_scores()  # type: ignore[attr-defined]
        values.append(float(outcome[challenger_seat]))
        scores.append(float(final_scores[challenger_seat]))
        best = max(final_scores)
        winners = sum(score == best for score in final_scores)
        if final_scores[challenger_seat] == best:
            if winners == 1:
                first_places += 1
            else:
                draws += 1
    return ArenaSummary(
        games=games,
        mean_value=float(np.mean(values)),
        first_places=first_places,
        draws=draws,
        mean_score=float(np.mean(scores)),
    )

