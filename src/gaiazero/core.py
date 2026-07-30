from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@runtime_checkable
class GameState(Protocol):
    """Immutable state contract consumed by search and self-play."""

    @property
    def num_players(self) -> int: ...

    @property
    def current_player(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    @property
    def observation_size(self) -> int: ...

    @property
    def is_terminal(self) -> bool: ...

    def legal_actions(self) -> tuple[int, ...]: ...

    def legal_action_mask(self) -> BoolArray: ...

    def apply(self, action: int) -> Self: ...

    def observation(self) -> FloatArray: ...

    def returns(self) -> FloatArray: ...


class PolicyValueEvaluator(Protocol):
    """Returns an action prior and an absolute-player value vector."""

    def evaluate(self, state: GameState) -> tuple[FloatArray, FloatArray]: ...

